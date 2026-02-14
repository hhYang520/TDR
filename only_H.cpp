//
// Created by Yang on 2026/1/20.
//


#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <climits>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace bs {

using namespace std;

//图的存储，采用CSR方式
vector<int> source_offset;
vector<int> targets;
vector<int> target_offset;
vector<int> labels;

// V=vertices, TE=topology edges, LE=labeled edges, L=labels
int num_V = 0, num_TE = 0, num_LE = 0, num_L = 0;

int vis_cur = 1, cur = 0;
int *s;      // an empty stack, to record the vertices during DFS
int s_pos;   // record the top position of stack s
int *SCC;    // the vertices of strongly connected components except root vertex
int SCC_pos; // record the top position of stack SCC
const int vertex_hash_len = 8;// the number of hash bit for vertices
int label_hash_len = 3; // the number of hash bit for labels
const int UINT_bits = sizeof(uint32_t) * 8; // the number of bits in an uint32_t

//调用剪枝函数的次数
int prune_group_vertices = 0, prune_group_labels = 0, prune_forward = 0,
    prune_all_vertices = 0, prune_all_labels = 0, prune_interval = 0;
//成功剪枝的次数
int prune_group_vertices_success = 0, prune_group_labels_success = 0,
    prune_forward_success = 0, prune_all_vertices_success = 0,
    prune_all_labels_success = 0, prune_interval_success = 0;
unsigned long long AND_visited_count = 0, OR_visited_count = 0, NOT_visited_count = 0;

//为每个顶点存储的索引
struct VertexIndex {
  bool root{};       //是否是根顶点
  // the time of pushing into stack and popping out stack
  int t_in{}, t_out{};

  // Hash of all vertices and labels that u can reach
  vector<uint32_t> vertices_all;
  vector<uint32_t> labels_all;

  //后继顶点动态分组的组数；对于叶子顶点，记录自身的哈希值
  int group{};
  //记录各分组的所有出标签
  vector<uint32_t> labels_group;
  //后续顶点的动态分组
  vector<uint32_t> vertices_group;
};
vector<VertexIndex> vertices;

//单条PCR查询
struct Query {
  int origin;
  int destination;
  int pattern;
  vector<uint32_t> bits_label;
  unordered_set<int> ids_label;
  vector<int> ids_sequence;
  bool outcome;
};
vector<Query> queries;

inline int vertex_hash(const int &parent) {
  //叶子顶点的哈希
  if (parent == -1) {
    static int leaf = 0;
    leaf++;
    //尽可能使不同叶子顶点的哈希值差异大一些
    int x = leaf;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
  } else {
    static int c = -1;
    static int x = 0;
    if (c != parent) {
      c = parent;
      x = parent;
    }
    x++;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
  }
}

//标签哈希表
static struct LabelHashTable {
  std::vector<uint8_t> p;
  std::vector<uint8_t> h;

  void init(int numL, int L) {
    p.resize(numL);
    h.resize(numL);
    for (int l = 0; l < numL; ++l) {
      int blk = l >> 5;          // /32
      p[l] = blk % L;
      h[l] = l & (UINT_bits - 1);
    }
  }
} lht;

// 构建索引时的过程态
struct BuildState {
  std::vector<int> vis;         //顶点是否已访问
  std::vector<int> next;        //需要处理的下一个邻居
  std::vector<int> low;         // Tarjan
  std::vector<uint32_t> hash_value;  //顶点的哈希值

  void init(int n) {
    vis.assign(n, 0);
    next.assign(n, 0);
    low.assign(n, 0);
    hash_value.assign(n, 0);
  }
} bst;

//查询时的过程态
struct QueryState {
  std::vector<int> vis;       //顶点是否已访问
  std::vector<int> next;    //需要处理的下一个邻居顶点
  std::vector<int> gid;       //需要处理的组号

  void init(const int &n) {
    vis.assign(n, 0);
    next.resize(n);
    gid.resize(n);
  }
} qs;

//在whole Path和两个sequences查询时需要的过程态
struct QueryStateSupplement {
  std::vector<int> l_next;        //需要处理的下一个邻居标签
  std::vector<bool> topology;     //是否拓扑可达

  void init(const int &n) {
    l_next.resize(n);
    topology.resize(n);
  }
} qsSup;

// 路径级信息存储，位集的形式
struct PathStateBits {
  vector<uint32_t> path_bits; //每个顶点在当前路径上的标签信息，位集的形式存储

  void init(const int &n, const int &L) {
    path_bits.assign(n * L, 0u);
  }

  inline uint32_t *pl(const int &v) {
    return &path_bits[(size_t) v * (size_t) label_hash_len];
  }
  [[nodiscard]] inline const uint32_t *pl(const int &v) const {
    return &path_bits[(size_t) v * (size_t) label_hash_len];
  }
  [[nodiscard]] inline const bool is_subset(const int &v, const vector<uint32_t> &tps) const {
    const int vps = (size_t) v * (size_t) label_hash_len;
    for (int i = 0; i < label_hash_len; ++i) {
      // 如果 vps 不包含 tps 的所有位
      if ((path_bits[vps + i] & tps[i]) != tps[i]) {
        return false;
      }
    }
    return true;
  }

} ps;

// 路径级信息存储，whole Path类查询时精确的标签存储
// 当图中的标签个数大于label_hash_len*UINT_bits时执行
struct PathStateExact {
  vector<int> path_label_seq;              // dfs中当前路径上的标签
  unordered_map<int, uint8_t> ids_hash;    // 当前查询集中标签的哈希值
  vector<int> first_matched_pos;           // 在dfs中第一次匹配标签的位置
  uint8_t target_bits{};                     // 当前查询集idsL的总哈希值
  uint8_t current_bits{};                    // 当前路径上匹配标签的总哈希值
  vector<vector<uint8_t>> visited_vertex;  //记录已访问顶点上的标签集

  inline void init(const unordered_set<int> &idsL) {
    int n = (int) idsL.size();
    ids_hash.clear();
    ids_hash.reserve(n * 2 + 1);
    first_matched_pos.clear();
    first_matched_pos.resize(n);
    target_bits = 0;
    int pos = INT_MAX;
    uint8_t index = 0;
    for (auto &l: idsL) {
      ids_hash.emplace(l, index);
      first_matched_pos[index] = pos;
      target_bits |= (1u << index);
      index++;
    }
    current_bits = 0;
    path_label_seq.clear();
    path_label_seq.resize(num_V);
    // 完全释放所有内存的最有效方法
    vector<vector<uint8_t>>().swap(visited_vertex);
    // 然后重新分配
    visited_vertex.resize(num_V);
  }

  inline bool should_push_label(const int &l, const int &p) {
    uint8_t index = ids_hash[l];
    if (first_matched_pos[index] > p) {
      return true;
    } else {
      return false;
    }
  }

  inline void push_label(const int &l, const bool &flag, const int &p) {
    uint8_t index = ids_hash[l];
    if (flag) {
      first_matched_pos[index] = p;
      path_label_seq.push_back(l);
      current_bits |= (1u << index);
    } else {
      path_label_seq.push_back(0);
    }
  }

  inline bool should_revisit(const int &uid, const int &l, const bool &flag) {
    uint8_t new_bits = current_bits;
    if (flag) new_bits |= (1u << ids_hash[l]);
    for (auto &saved_bits: visited_vertex[uid]) {
      // 如果新的路径上没有新的匹配标签出现，则不必重新处理该顶点
      if ((saved_bits & new_bits) == new_bits) {
        return false;
      }
    }
    visited_vertex[uid].push_back(new_bits);
    return true;
  }

  inline void pop_label(const int &uid) {
    visited_vertex[uid].push_back(current_bits);
    const int l = path_label_seq.back();
    if (l != 0) {
      uint8_t index = ids_hash[l];
      first_matched_pos[index] = INT_MAX;
      //将current_bits的第index位 重置为0
      current_bits &= ~(1u << index);
    }
  }
} psExact;

//sequences类查询的过程态
struct SequenceQueryState {
  uint8_t need_pos;                   //记录sequence类查询时需要匹配的标签在idsL中的位置
  int num;                            //idsSeq中标签的个数
  vector<vector<int>> matched_label;  //顶点中匹配的标签
  vector<int> alternative;            //sequence1类查询中备选的标签
  void init(const std::vector<int> &idsSeq) {
    num = (int) idsSeq.size();
    need_pos = 0;
    // 完全释放所有内存的最有效方法
    vector<vector<int>>().swap(matched_label);
    matched_label.resize(num_V);
    alternative.assign(num_V, -1);
  }
  inline void push_label_seq1(const int &uid, const int &l, const std::vector<int> &idsSeq) {
    matched_label[uid].push_back(l);
    if (l == idsSeq[need_pos]) {
      //need_pos的上界是数组idsSeq的末尾
      need_pos = (need_pos < num - 1) ? (need_pos + 1) : need_pos;
    }
  }
  inline void push_label_seq2(const int &uid, const int &l) {
    matched_label[uid].push_back(l);
    // 实现循环递增
    need_pos = (need_pos + 1) % num;
  }
  inline void pop_label_seq1(const int &uid, const std::vector<int> &idsSeq) {
    //起始点的时候不必检查
    if (matched_label[uid].empty()) return;
    int current_label = matched_label[uid].back();
    //check_pos的上界是数组idsSeq的起始
    int check_pos = (need_pos > 0) ? (need_pos - 1) : need_pos;
    if (current_label != idsSeq[check_pos] && current_label != idsSeq[num - 1]) {
      need_pos = check_pos;
    }
  }
  inline void pop_label_seq2() {
    // 实现循环递减
    need_pos = (need_pos - 1 + num) % num;
  }
  inline bool should_revisit(const int &uid, const int &l) {
    for (auto old_match: matched_label[uid]) {
      if (old_match == l) {
        return false;
      }
    }
    return true;
  }
  inline bool accept(const int &uid, const std::vector<int> &idsSeq) {
    return matched_label[uid].back() == idsSeq[num - 1];
  }
} sqs;

// 查询模式枚举
enum class Mode { WholePath, AllLabel, Sequence1, Sequence2 };
enum class LabelMatch { AND, OR, NOT, LCR };

// ==================== 标签匹配策略 ====================
template<LabelMatch LM>
struct LabelMatcher {
  // 顶点剪枝：基于顶点标签和查询标签
  static bool prune_by_bits(const uint32_t &vertex_bits,
                            const uint32_t &path_bits,
                            const uint32_t &required_bits) {
    if constexpr (LM == LabelMatch::AND) {
      return ((vertex_bits | path_bits) & required_bits) != required_bits;
    } else if constexpr (LM == LabelMatch::OR) {
      return ((vertex_bits | path_bits) & required_bits) == 0;
    } else if constexpr (LM == LabelMatch::NOT) {
      return (vertex_bits & required_bits) == vertex_bits;
    } else if constexpr (LM == LabelMatch::LCR) {
      return (vertex_bits & required_bits) == 0;
    }
  }

  // 边可行性：检查边上的标签
  static bool feasible_bits(int &l_next, const int &end_pos,
                            uint32_t *v_ps,
                            const vector<uint32_t> &bitsL) {
    for (int i = l_next; i < end_pos; ++i) {
      int tl = labels[i];
      const uint8_t p = lht.p[tl];
      const uint8_t h = lht.h[tl];
      bool label_in_query = ((bitsL[p] & (1u << h)) != 0);
      if constexpr (LM == LabelMatch::NOT) {
        // NOT：不能包含任何查询标签
        if (!label_in_query) {
          return true;
        }
      } else if constexpr (LM == LabelMatch::LCR) {
        // LCR：需要包含查询标签
        if (label_in_query) {
          return true;
        }
      } else {
        bool label_in_path = ((v_ps[p] & (1u << h)) != 0);
        //对于AND和OR来说，存在新的未匹配的标签
        if (label_in_query && !label_in_path) {
          l_next = i + 1;
          v_ps[p] |= (1u << h);
          return true;
        }
      }
    }
    if constexpr(LM == LabelMatch::AND || LM == LabelMatch::OR) {
      l_next = end_pos;
    }
    return false;
  }
  static bool feasible_exact(int &l_next, const int &end_pos,
                             const std::unordered_set<int> &idsL,
                             int &tl, const int &pos) {
    for (int i = l_next; i < end_pos; ++i) {
      tl = labels[i];
      bool label_in_query = (idsL.find(tl) != idsL.end());
      if constexpr (LM == LabelMatch::NOT) {
        // NOT：不能包含任何查询标签
        if (!label_in_query) {
          return true;
        }
      } else if constexpr (LM == LabelMatch::LCR) {
        // LCR：需要包含查询标签
        if (label_in_query) {
          return true;
        }
      } else {
        bool label_in_path = psExact.should_push_label(tl, pos);
        //对于AND和OR来说，存在新的未匹配的标签
        if (label_in_query && !label_in_path) {
          l_next = i + 1;
          return true;
        }
      }
    }
    if constexpr(LM == LabelMatch::AND || LM == LabelMatch::OR) {
      l_next = end_pos;
    }
    return false;
  }

  // 路径接受：基于路径标签
  static bool accept_bits(const uint32_t &path_bits,
                          const uint32_t &required_bits) {
    if constexpr (LM == LabelMatch::AND) {
      return path_bits == required_bits;
    } else if constexpr (LM == LabelMatch::OR) {
      return path_bits != 0;
    } else {
      return true;
    }
  }
  static bool accept_exact(const uint8_t &current_bits,
                           const uint8_t &require_bits) {
    if constexpr (LM == LabelMatch::AND) {
      return current_bits == require_bits;
    } else if constexpr (LM == LabelMatch::OR) {
      return current_bits != 0;
    } else {
      return true;
    }
  }
};

// ==================== 核心搜索器模板 ====================
template<Mode M, LabelMatch LM, bool UseExact = false>
class ReachabilitySearcher {
 public:
  bool search(int uid, int vid,
              const std::vector<uint32_t> &bitsL,
              const std::unordered_set<int> &idsL,
              const std::vector<int> &idsSeq) {
    return search_impl(uid, vid, bitsL, idsL, idsSeq);
  }

 private:
  using matcher = LabelMatcher<LM>;
  // ==================== 剪枝判断 ====================
  // 顶点包含性剪枝，true表示能够剪枝，false表示不能剪枝
  [[nodiscard]] bool vertex_prune(const VertexIndex &U, const VertexIndex &V) const {
    prune_all_vertices++;
    // 目标顶点是叶子
    if (V.group < 0) {
      int blk = (-V.group) & (vertex_hash_len - 1);
      int bit = (-V.group) & (UINT_bits - 1);
      bool flag = (U.vertices_all[blk] & (1u << bit)) == 0;
      if (flag) prune_all_vertices_success++;
      return flag;
    }

    // 目标顶点不是叶子
    for (int i = 0; i < vertex_hash_len; ++i) {
      if ((U.vertices_all[i] & V.vertices_all[i]) != V.vertices_all[i]) {
        prune_all_vertices_success++;
        return true;
      }
    }
    return false;
  }

  // 标签包含性剪枝，针对whole path和all labels类的查询。true表示能够剪枝，false表示不能剪枝
  [[nodiscard]] bool label_prune_not_sequence(const int &uid, const std::vector<uint32_t> &bitsL) const {
    prune_all_labels++;
    const VertexIndex &U = vertices[uid];
    for (int i = 0; i < label_hash_len; ++i) {
      if (!matcher::prune_by_bits(U.labels_all[i], ps.pl(uid)[i], bitsL[i])) {
        return false;
      }
    }
    prune_all_labels_success++;
    return true;
  }

  // 标签包含性剪枝，针对两类sequence的查询。true表示能够剪枝，false表示不能剪枝
  [[nodiscard]] bool label_prune_sequences(const VertexIndex &U,
                                           const int &needed_pos,
                                           const std::vector<int> &idsSeq) const {
    prune_all_labels++;
    int n = sqs.num;
    //若没有包含剩余所有元素，则剪枝
    for (int i = needed_pos; i < n; ++i) {
      int l = idsSeq[i];
      const int p = lht.p[l];
      const int h = lht.h[l];
      if ((U.labels_all[p] & (1u << h)) == 0) {
        prune_all_labels_success++;
        return true;
      }
    }
    return false;
  }

  // 根据分组顶点的剪枝策略，true表示能够剪枝，false表示不能剪枝
  [[nodiscard]] bool group_vertex_prune(const VertexIndex &U,
                                        const VertexIndex &V,
                                        const int &gid) const {
    prune_group_vertices++;
    // 目标顶点是叶子
    if (V.group < 0) {
      int blk = (-V.group) & (vertex_hash_len - 1);
      int bit = (-V.group) & (UINT_bits - 1);
      uint32_t temp = U.vertices_group[gid * vertex_hash_len + blk];
      bool flag = (temp & (1u << bit)) == 0;
      if (flag) prune_group_vertices_success++;
      return flag;
    }

    // 目标顶点不是叶子
    for (int i = 0; i < vertex_hash_len; ++i) {
      uint32_t temp = U.vertices_group[gid * vertex_hash_len + i];
      if ((temp & V.vertices_all[i]) != V.vertices_all[i]) {
        prune_group_vertices_success++;
        return true;
      }
    }
    return false;
  }

  // 针对whole path类查询的分组标签剪枝策略，true表示能够剪枝，false表示不能剪枝
  [[nodiscard]] bool group_label_prune(const int &uid,
                                       const std::vector<uint32_t> &bitsL,
                                       const int &gid) const {
    prune_group_labels++;
    const VertexIndex &U = vertices[uid];
    int index = gid * label_hash_len;
    for (int i = 0; i < label_hash_len; ++i) {
      uint32_t vertex_bits = U.labels_group[index + i];
      //任意一个标签哈希位不能剪枝，则整体都不能剪枝
      if (!matcher::prune_by_bits(vertex_bits, ps.pl(uid)[i], bitsL[i])) {
        return false;
      }
    }
    prune_group_labels_success++;
    return true;
  }

  // 标签接受判断 (WholePath)
  [[nodiscard]] bool label_accept_check(const int &uid, const std::vector<uint32_t> &bitsL) const {
    if constexpr (UseExact) {
      if (matcher::accept_exact(psExact.current_bits, psExact.target_bits))
        return true;
    } else {
      for (int i = 0; i < bitsL.size(); ++i) {
        if (!matcher::accept_bits(ps.pl(uid)[i], bitsL[i])) {
          return false;
        }
      }
      return true;
    }
    return true;
  }

  int feasible_sequence1(const int &uid, const std::vector<int> &idsSeq, int &l1, int &l2) {
    bool flag1 = false, flag2 = false;
    //check_pos的上界是数组idsSeq的起始
    int check_pos = (sqs.need_pos > 0) ? (sqs.need_pos - 1) : sqs.need_pos;
    //如果匹配到末尾了，则只检查need_pos的标签
    int endLabel = -1;
    if (!sqs.matched_label[uid].empty()) endLabel = sqs.matched_label[uid].back();
    bool endPos = (endLabel == idsSeq[sqs.num - 1]);
    for (int i = target_offset[qs.next[uid]]; i < target_offset[qs.next[uid] + 1]; ++i) {
      int tl = labels[i];
      if (tl == idsSeq[sqs.need_pos]) {
        l1 = tl;
        flag1 = true;
        continue;
      }
      if (endPos) continue;
      if (tl == idsSeq[check_pos]) {
        l2 = tl;
        flag2 = true;
      }
    }
    if (flag1 && flag2) return 3;
    if (flag1) return 1;
    if (flag2) return 2;
    return 0;
  }
  bool feasible_sequence2(const int &uid, const std::vector<int> &idsSeq) {
    for (int i = target_offset[qs.next[uid]]; i < target_offset[qs.next[uid] + 1]; ++i) {
      int tl = labels[i];
      if (tl == idsSeq[sqs.need_pos]) {
        return true;
      }
    }
    return false;
  }

  // ==================== 核心搜索实现 ====================
  bool search_impl(int uid, int vid,
                   const std::vector<uint32_t> &bitsL,
                   const std::unordered_set<int> &idsL,
                   const std::vector<int> &idsSeq) {
    const VertexIndex &V = vertices[vid];
    if (V.root) return false;

    // 初始化栈和起始顶点
    s[s_pos] = uid;
    bool isAdd = true;

    if constexpr (M == Mode::WholePath) {
      //初始顶点的path_bit置零
      std::memset(ps.pl(uid), 0u, label_hash_len * sizeof(uint32_t));
      if constexpr (UseExact) psExact.init(idsL);
    }
    if constexpr(M == Mode::Sequence1 || M == Mode::Sequence2) {
      sqs.init(idsSeq);
    }

    while (s_pos != -1) { //当栈为空时，遍历结束
      int top_uid = s[s_pos];
      VertexIndex &top_u = vertices[top_uid];
      if constexpr(M == Mode::Sequence1) {
        sqs.pop_label_seq1(top_uid, idsSeq);
      }

      if (isAdd) {
        isAdd = false;
        if constexpr(LM == LabelMatch::AND) {
          AND_visited_count++;
        } else if constexpr(LM == LabelMatch::OR) {
          OR_visited_count++;
        } else if constexpr(LM == LabelMatch::NOT) {
          NOT_visited_count++;
        }
        if constexpr(M != Mode::AllLabel) {
          qsSup.topology[top_uid] = true;
        }

        //找到具体路径则可达
        if (top_uid == vid) {
          if constexpr (M == Mode::AllLabel) {
            // AllLabel 模式到达目标顶点即成功
            return true;
          } else if constexpr(M == Mode::WholePath) {
            if (label_accept_check(top_uid, bitsL)) {
              return true;
            }
          } else {
            if (sqs.accept(vid, idsSeq)) {
              return true;
            }
          }
          // 标记为已访问并弹出
          qs.vis[top_uid] = vis_cur + 1;
          --s_pos;
          if constexpr(M == Mode::WholePath && UseExact) {
            psExact.pop_label(top_uid);
          }
          if constexpr(M == Mode::Sequence2) {
            sqs.pop_label_seq2();
          }
          continue;
        }

        //若栈顶元素为叶子顶点，则剪枝
        if (top_u.group < 0) {
          if constexpr(M != Mode::AllLabel) {
            qsSup.topology[top_uid] = false;
          }
          if constexpr(M == Mode::WholePath && UseExact) {
            psExact.pop_label(top_uid);
          }
          if constexpr(M == Mode::Sequence2) {
            sqs.pop_label_seq2();
          }
          qs.vis[top_uid] = vis_cur + 1;
          --s_pos;
          continue;
        }

        //根据interval进行可达性判断
        prune_interval++;
        if (top_u.t_out < V.t_out) {
          if constexpr(M != Mode::AllLabel) {
            qsSup.topology[top_uid] = false;
          }
          if constexpr(M == Mode::WholePath && UseExact) {
            psExact.pop_label(top_uid);
          }
          if constexpr(M == Mode::Sequence2) {
            sqs.pop_label_seq2();
          }
          prune_interval_success++;
          qs.vis[top_uid] = vis_cur + 1;
          --s_pos;
          continue;
        }
        if constexpr (M == Mode::WholePath) {
          if (top_u.t_in <= V.t_in && label_accept_check(top_uid, bitsL)) return true;
        }

        //根据栈顶元素的标签进行可达性判断
        if constexpr(M == Mode::Sequence1 || M == Mode::Sequence2) {
          if (label_prune_sequences(top_u, sqs.need_pos, idsSeq)) {
            qs.vis[top_uid] = vis_cur + 1;
            if constexpr(M == Mode::Sequence2) {
              sqs.pop_label_seq2();
            }
            --s_pos;
            continue;
          }
        } else {
          if (label_prune_not_sequence(top_uid, bitsL)) {
            if constexpr(M == Mode::WholePath && UseExact) {
              psExact.pop_label(top_uid);
            }
            qs.vis[top_uid] = vis_cur + 1;
            --s_pos;
            continue;
          }
        }

        //根据栈顶元素的顶点包含关系进行可达性判断
        if (vertex_prune(top_u, V)) {
          if constexpr(M != Mode::AllLabel) {
            qsSup.topology[top_uid] = false;
          }
          if constexpr(M == Mode::WholePath && UseExact) {
            psExact.pop_label(top_uid);
          }
          if constexpr(M == Mode::Sequence2) {
            sqs.pop_label_seq2();
          }
          qs.vis[top_uid] = vis_cur + 1;
          --s_pos;
          continue;
        }

        qs.vis[top_uid] = vis_cur;
        if (top_u.group > 1) {
          qs.gid[top_uid] = -1;
          qs.next[top_uid] = source_offset[top_uid + 1];
        } else {
          //只有1组时不需要通过分组剪枝
          qs.gid[top_uid] = 1;
          qs.next[top_uid] = source_offset[top_uid];
        }
      }

      //当通过标签无法判断时，需要进行遍历
      bool finished = true;
      //根据分组进行剪枝
      while (qs.gid[top_uid] < top_u.group) {
        //原来的分组已经处理完
        if (qs.next[top_uid] >= source_offset[top_uid + 1]) {
          qs.gid[top_uid]++;
          if (qs.gid[top_uid] == top_u.group) break;
          // 分组标签剪枝
          if constexpr (M == Mode::WholePath) {
            if (group_label_prune(top_uid, bitsL, qs.gid[top_uid])) continue;
          }

          // 分组顶点剪枝
          if (group_vertex_prune(top_u, V, qs.gid[top_uid])) {
            continue;
          }
          //若该组不能被剪枝，则依次访问该组的顶点
          qs.next[top_uid] = source_offset[top_uid] + qs.gid[top_uid];
          finished = false;
          break;
        } else {
          finished = false;
          break;
        }
      }
      // 依次处理分组中的每个顶点
      while (qs.next[top_uid] < source_offset[top_uid + 1]) {
        //如果路径上的标签不在给定标签集中，则进行剪枝
        int tid = targets[qs.next[top_uid]];
        VertexIndex &t = vertices[tid];
        if constexpr(M == Mode::AllLabel) {
          if (qs.vis[tid] < vis_cur) {
            bool flag;
            if constexpr (UseExact) {
              int temp;
              flag = matcher::feasible_exact(target_offset[qs.next[top_uid]],
                                             target_offset[qs.next[top_uid] + 1],
                                             idsL, temp, 0);
            } else {
              flag = matcher::feasible_bits(target_offset[qs.next[top_uid]],
                                            target_offset[qs.next[top_uid] + 1], {}, bitsL);
            }
            if (flag) {
              isAdd = true;
              s[++s_pos] = tid;
              finished = false;
              break;
            }
          }
          qs.next[top_uid] += top_u.group;
        } else {
          //如果后继顶点未访问，则入栈
          if (qs.vis[tid] < vis_cur) {
            if constexpr(M == Mode::WholePath) {
              qsSup.l_next[top_uid] = target_offset[qs.next[top_uid]];
              if constexpr (UseExact) {
                int tl = 0;
                // 把 top_u 的 path_label 复制给 t
                memcpy(ps.pl(tid), ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                bool flag = matcher::feasible_exact(qsSup.l_next[top_uid],
                                                    target_offset[qs.next[top_uid] + 1],
                                                    idsL, tl, s_pos);
                psExact.push_label(tl, flag, s_pos);
                if (flag) {
                  const uint8_t p = lht.p[tl];
                  const uint8_t h = lht.h[tl];
                  ps.pl(tid)[p] |= (1u << h);
                }
              } else {
                memcpy(ps.pl(tid), ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                bool flag = matcher::feasible_bits(qsSup.l_next[top_uid],
                                                   target_offset[qs.next[top_uid] + 1],
                                                   ps.pl(tid), bitsL);
              }
              isAdd = true;
              s[++s_pos] = tid;
              finished = false;
              break;
            } else if constexpr (M == Mode::Sequence1) {
              int l1 = -1, l2 = -1;
              const int r = feasible_sequence1(top_uid, idsSeq, l1, l2);
              if (r == 3) {
                sqs.push_label_seq1(tid, l1, idsSeq);
                sqs.alternative[tid] = l2;
              } else if (r == 2) {
                sqs.push_label_seq1(tid, l2, idsSeq);
              } else if (r == 1) {
                sqs.push_label_seq1(tid, l1, idsSeq);
              }
              if (r != 0) {
                isAdd = true;
                s[++s_pos] = tid;
                finished = false;
                break;
              }
            } else {
              if (feasible_sequence2(top_uid, idsSeq)) {
                isAdd = true;
                s[++s_pos] = tid;
                sqs.push_label_seq2(tid, idsSeq[sqs.need_pos]);
                finished = false;
                break;
              }
            }
          } else if (qs.vis[tid] > vis_cur && qsSup.topology[tid]) {
            if constexpr(M == Mode::WholePath) {
              if constexpr (UseExact) {
                int tl = 0;
                bool flag = matcher::feasible_exact(qsSup.l_next[top_uid],
                                                    target_offset[qs.next[top_uid] + 1],
                                                    idsL, tl, s_pos);
                if (psExact.should_revisit(tid, tl, flag)) {
                  isAdd = true;
                  s[++s_pos] = tid;
                  finished = false;
                  // 把 top_u 的 path_label 复制给 t
                  memcpy(ps.pl(tid), ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                  const uint8_t p = lht.p[tl];
                  const uint8_t h = lht.h[tl];
                  ps.pl(tid)[p] |= (1u << h);
                  break;
                }
              } else {
                vector<uint32_t> tps(label_hash_len);
                memcpy(&tps[0], ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                bool flag = matcher::feasible_bits(qsSup.l_next[top_uid],
                                                   target_offset[qs.next[top_uid] + 1],
                                                   &tps[0], bitsL);
                //如果新路径存在新标签，则t重新入栈
                if (!ps.is_subset(tid, tps)) {
                  memcpy(ps.pl(tid), &tps[0], sizeof(uint32_t) * label_hash_len);
                  isAdd = true;
                  s[++s_pos] = tid;
                  finished = false;
                  break;
                }
              }
            } else if constexpr(M == Mode::Sequence1) {
              //回溯边
              if (sqs.alternative[tid] != -1) {
                int l = sqs.alternative[tid];
                sqs.alternative[tid] = -1;
                if (sqs.should_revisit(tid, l)) {
                  sqs.push_label_seq1(tid, l, idsSeq);
                  isAdd = true;
                  s[++s_pos] = tid;
                  finished = false;
                  qs.next[top_uid] += top_u.group;
                  break;
                }
              } else {
                int l1 = -1, l2 = -1;
                const int r = feasible_sequence1(top_uid, idsSeq, l1, l2);
                bool flag = false;
                if (r == 3) {
                  if (sqs.should_revisit(tid, l1)) {
                    sqs.push_label_seq1(tid, l1, idsSeq);
                    flag = true;
                    if (sqs.should_revisit(tid, l2)) sqs.alternative[tid] = l2;
                  } else if (sqs.should_revisit(tid, l2)) {
                    sqs.push_label_seq1(tid, l2, idsSeq);
                    flag = true;
                  }
                } else if (r == 2) {
                  if (sqs.should_revisit(tid, l2)) {
                    sqs.push_label_seq1(tid, l2, idsSeq);
                    flag = true;
                  }
                } else if (r == 1) {
                  if (sqs.should_revisit(tid, l1)) {
                    sqs.push_label_seq1(tid, l1, idsSeq);
                    flag = true;
                  }
                }
                if (flag) {
                  isAdd = true;
                  s[++s_pos] = tid;
                  finished = false;
                  break;
                }
              }

            } else {
              if (feasible_sequence2(top_uid, idsSeq) && sqs.should_revisit(tid, idsSeq[sqs.need_pos])) {
                isAdd = true;
                s[++s_pos] = tid;
                sqs.push_label_seq2(tid, idsSeq[sqs.need_pos]);
                finished = false;
                break;
              }
            }
          }
          qs.next[top_uid] += top_u.group;
          if constexpr(M == Mode::WholePath) {
            //非回溯边检查边上所有标签
            if (qs.next[top_uid] < source_offset[top_uid + 1]) qsSup.l_next[top_uid] = target_offset[qs.next[top_uid]];
          }
        }
      }

      if (finished) {
        --s_pos; //弹出栈顶元素
        qs.vis[top_uid] = vis_cur + 1;
        isAdd = false;
        if constexpr(M == Mode::WholePath && UseExact) {
          psExact.pop_label(top_uid);
        }
        if constexpr(M == Mode::Sequence2) {
          sqs.pop_label_seq2();
        }
      }
    }
    return false;
  }
};

// ==================== 统一的查询接口 ====================
class ReachableQuery {
 private:
  // 编译期分发：使用函数模板和 if constexpr
  template<int Ptn>
  bool searcher(int uid, int vid,
                const std::vector<uint32_t> &bitsL,
                const std::unordered_set<int> &idsL,
                const std::vector<int> &idsSeq) {
    // 编译器会为每个 Ptn 生成特化版本
    if constexpr (Ptn == 1) {
      static ReachabilitySearcher<Mode::WholePath, LabelMatch::AND, false> searcher;
      return searcher.search(uid, vid, bitsL, {}, {});
    } else if constexpr (Ptn == 2) {
      static ReachabilitySearcher<Mode::WholePath, LabelMatch::OR, false> searcher;
      return searcher.search(uid, vid, bitsL, {}, {});
    } else if constexpr (Ptn == 3) {
      static ReachabilitySearcher<Mode::AllLabel, LabelMatch::NOT, false> searcher;
      return searcher.search(uid, vid, bitsL, {}, {});
    } else if constexpr (Ptn == 4) {
      static ReachabilitySearcher<Mode::AllLabel, LabelMatch::LCR, false> searcher;
      return searcher.search(uid, vid, bitsL, {}, {});
    } else if constexpr (Ptn == 5) {
      static ReachabilitySearcher<Mode::WholePath, LabelMatch::AND, true> searcher;
      return searcher.search(uid, vid, bitsL, idsL, {});
    } else if constexpr (Ptn == 6) {
      static ReachabilitySearcher<Mode::WholePath, LabelMatch::OR, true> searcher;
      return searcher.search(uid, vid, bitsL, idsL, {});
    } else if constexpr (Ptn == 7) {
      static ReachabilitySearcher<Mode::AllLabel, LabelMatch::NOT, true> searcher;
      return searcher.search(uid, vid, bitsL, idsL, {});
    } else if constexpr (Ptn == 8) {
      static ReachabilitySearcher<Mode::AllLabel, LabelMatch::LCR, true> searcher;
      return searcher.search(uid, vid, bitsL, idsL, {});
    } else if constexpr (Ptn == 9) {
      static ReachabilitySearcher<Mode::Sequence1, LabelMatch::AND, false> searcher;
      return searcher.search(uid, vid, {}, {}, idsSeq);
    } else if constexpr (Ptn == 10) {
      static ReachabilitySearcher<Mode::Sequence2, LabelMatch::AND, false> searcher;
      return searcher.search(uid, vid, {}, {}, idsSeq);
    } else {
      // 编译期错误或运行时异常
      throw std::invalid_argument("Invalid ptn value");
    }
  }

 public:
  // 运行时分发 - 编译器可能优化为直接跳转
  __attribute__((always_inline))
  bool search_dispatch(int ptn, int uid, int vid,
                       const std::vector<uint32_t> &bitsL,
                       const std::unordered_set<int> &idsL,
                       const std::vector<int> &idsSeq) {
    // 使用 switch 让编译器生成高效跳转表
    switch (ptn) {
      case 1: return searcher<1>(uid, vid, bitsL, idsL, idsSeq);
      case 2: return searcher<2>(uid, vid, bitsL, idsL, idsSeq);
      case 3: return searcher<3>(uid, vid, bitsL, idsL, idsSeq);
      case 4: return searcher<4>(uid, vid, bitsL, idsL, idsSeq);
      case 5: return searcher<5>(uid, vid, bitsL, idsL, idsSeq);
      case 6: return searcher<6>(uid, vid, bitsL, idsL, idsSeq);
      case 7: return searcher<7>(uid, vid, bitsL, idsL, idsSeq);
      case 8: return searcher<8>(uid, vid, bitsL, idsL, idsSeq);
      case 9: return searcher<9>(uid, vid, bitsL, idsL, idsSeq);
      case 10: return searcher<10>(uid, vid, bitsL, idsL, idsSeq);
      default:throw std::invalid_argument("Invalid ptn value: " + std::to_string(ptn));
    }
  }

//  // 便捷方法：针对常用模式
//  static bool whole_path_and(int uid, int vid, const std::vector<uint32_t> &bitsL) {
//    return search_impl<1>(uid, vid, bitsL, {}, {});
//  }
//
//  bool whole_path_or(int uid, int vid, const std::vector<uint32_t> &bitsL) {
//    return search_impl<2>(uid, vid, bitsL, {}, {});
//  }
//
//  bool whole_path_and_exact(int uid, int vid,
//                            const std::unordered_set<int> &idsL) {
//    return search_impl<5>(uid, vid, {}, idsL, {});
//  }
//
//  bool sequence1(int uid, int vid, const std::vector<int> &idsSeq) {
//    return search_impl<9>(uid, vid, {}, {}, idsSeq);
//  }
};

// read graph file and show the distribution of labels
bool read_graph(const string &filename) {
  clock_t start_time, end_time;
  start_time = clock();
  ifstream file;
  file.open(filename, ios::in);
  string str;

  if (!file.is_open()) {
    cout << "Failed to read graph file!" << endl;
    return false;
  }
  file >> str >> num_V >> str >> num_TE >> str >> num_LE >> str >> num_L;

  // initialize vertices
  vertices.resize(num_V);
  // 整数除法向上取整 (a + b - 1) / b
//  if (num_L < label_hash_len * UINT_bits) label_hash_len = (num_L + 32 - 1) / 32;
  if (num_L < (label_hash_len << 5)) label_hash_len = (num_L + 32 - 1) >> 5;
  for (auto &vtx: vertices) {
    vtx.t_in = 0;
    vtx.t_out = 0;
    vtx.root = true;
  }
  bst.init(num_V);

  // count the number of each label
  vector<int> each_label_num;
  each_label_num.assign(num_L, 0);
  lht.init(num_L, label_hash_len);

  source_offset.assign(num_V + 1, 0);
  targets.resize(num_TE);
  target_offset.assign(num_TE + 1, 0);
  labels.resize(num_LE);

  // the index of target vertices and associated labels
  int u, v, v_pos = 0, l_pos = 0;
  while (file >> u >> v >> str) {
    targets[v_pos] = v;
    vertices[v].root = false;
    stringstream ss(str);
    while (getline(ss, str, ',')) {
      int l = stoi(str);
      labels[l_pos] = l;
      each_label_num[l]++;
      l_pos++;
      target_offset[v_pos + 1] = l_pos;
    }
    v_pos++;
    source_offset[u + 1] = v_pos;
  }
  file.close();
  for (int i = 1; i < num_V + 1; ++i) {
    if (source_offset[i] == 0) source_offset[i] = source_offset[i - 1];
  }
  for (int i = 1; i < num_TE + 1; ++i) {
    if (target_offset[i] == 0) target_offset[i] = target_offset[i - 1];
  }

  // show the distribution of labels
  cout << "### Label Distribution ###" << endl;
  for (int i = 0; i < num_L; i++) {
    cout << "Label" << i << ": " << each_label_num[i] << "/" << num_LE << " = "
         << each_label_num[i] / (double) num_TE << endl;
  }

  cout << endl;

  end_time = clock();
  double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
  printf("read time(graph): %.3fs\n", total_time);

  return true;
}

void DFS_out(const int &start) {
  s[s_pos] = start; //起始点入栈
  bool isAdd = true;
  SCC_pos = 0;
  SCC[SCC_pos] = start;
  while (s_pos != -1) { //当栈为空时，遍历结束
    int top_vid = s[s_pos];
    VertexIndex &top_v = vertices[top_vid];
    if (isAdd) {
      isAdd = false;
      bst.vis[top_vid] = vis_cur;
      bst.next[top_vid] = source_offset[top_vid];
      top_v.t_in = ++cur;
      bst.low[top_vid] = cur;
    }
    bool finished = true;
    int max_neighbor = source_offset[top_vid + 1];
    while (bst.next[top_vid] < max_neighbor) {
      int vid = targets[bst.next[top_vid]];
      VertexIndex &v = vertices[vid];
      if (bst.vis[vid] < vis_cur) { //当后继顶点未被访问时，将后继顶点入栈
        ++s_pos;
        s[s_pos] = vid;
        ++SCC_pos;
        SCC[SCC_pos] = vid;
        isAdd = true;
        finished = false;
        break;
      } else if (bst.vis[vid] == vis_cur) {
        bst.low[top_vid] = min(bst.low[top_vid], bst.low[vid]);
      }
      bst.next[top_vid]++;
    }

    if (finished) { //表示该顶点的所有后继顶点都已经处理完，可以出栈
      int n = max_neighbor - source_offset[top_vid];
      if (n == 0) {
        bst.hash_value[top_vid] = vertex_hash(-1);
        top_v.group = (int) -bst.hash_value[top_vid];
      } else {
        //初始化
        top_v.vertices_all.assign(vertex_hash_len, 0);
        top_v.labels_all.assign(label_hash_len, 0);
        int group = 1;
        if (n > 3) group = (int) round(sqrt(n));
        top_v.group = group;
        //分配内存并初始化
        if (group > 1) {
          top_v.labels_group.assign(group * label_hash_len, 0);
          top_v.vertices_group.assign(group * vertex_hash_len, 0);
        }

        int order = 0;
        for (int i = source_offset[top_vid]; i < max_neighbor; i++) {
          int vid = targets[i], hv, index = order % group;
          order++;
          VertexIndex &v = vertices[vid];
          if (bst.hash_value[vid] == 0) {
            hv = vertex_hash(top_vid);
            bst.hash_value[vid] = hv;
          } else {
            hv = (int) bst.hash_value[vid];
          }
          for (int j = target_offset[i]; j < target_offset[i + 1]; ++j) {
            const uint8_t p = lht.p[labels[j]];
            const uint8_t h = lht.h[labels[j]];
            top_v.labels_all[p] |= (1u << h);
            if (group > 1) top_v.labels_group[index * label_hash_len + p] |= (1u << h);
          }
          // p=hv%vertex_hash_len, h=hv % UINT_bits;
          int p = hv & (vertex_hash_len - 1), h = hv & (UINT_bits - 1);
          top_v.vertices_all[p] |= (1u << h);
          if (group > 1) top_v.vertices_group[index * vertex_hash_len + p] |= (1u << h);
          //如果后继顶点在栈中，则跳过
          if (v.vertices_all.empty()) continue;
          //合并后继顶点的索引信息
          if (source_offset[vid] != source_offset[vid + 1]) {
            for (int j = 0; j < vertex_hash_len; ++j) {
              top_v.vertices_all[j] |= v.vertices_all[j];
            }
            for (int j = 0; j < label_hash_len; j++) {
              top_v.labels_all[j] |= v.labels_all[j];
            }
          }
        }
      }
      top_v.t_out = ++cur;

      //根据low判断顶点是否在环中
      if (top_v.t_in == bst.low[top_vid]) {
        while (true) {
          int tid = SCC[SCC_pos];
          VertexIndex &t = vertices[tid];
          SCC_pos--;
          bst.vis[tid] = vis_cur + 1;
          if (tid == top_vid) break;
          t.t_in = top_v.t_in;
          t.t_out = top_v.t_out;
          for (int i = 0; i < label_hash_len; ++i) {
            t.labels_all[i] = top_v.labels_all[i];
          }
          for (int j = 0; j < vertex_hash_len; ++j) {
            t.vertices_all[j] = top_v.vertices_all[j];
          }
        }
      }
      s_pos--;
    }
  }
}

// 构建索引并计算构建时间
void index_construction(const string &total_name) {
  clock_t start_time, end_time;
  start_time = clock();

  //优先处理根顶点所在的最大子图
  for (int u = 0; u < num_V; ++u) {
    //根顶点
    if (vertices[u].root) {
      s_pos = 0;
      DFS_out(u);
    }
  }

  //处理只有环的子图，即子图中没有顶点为根顶点
  for (int u = 0; u < num_V; ++u) {
    if (bst.vis[u] != vis_cur + 1) {
      s_pos = 0;
      DFS_out(u);
    }
  }

  //为顶点构建两维索引
  for (int uid = 0; uid < num_V; ++uid) {
    VertexIndex &u = vertices[uid];
    if (u.group > 1) {
      int order = 0;
      for (int i = source_offset[uid]; i < source_offset[uid + 1]; i++) {
        int vid = targets[i], index = order % u.group;
        order++;
        VertexIndex &v = vertices[vid];
        //合并非叶子顶点的索引信息
        if (source_offset[vid] != source_offset[vid + 1]) {
          //后继顶点的顶点信息
          for (int j = 0; j < vertex_hash_len; ++j) {
            u.vertices_group[index * vertex_hash_len + j] |= v.vertices_all[j];
          }
          //后继顶点的标签信息
          for (int j = 0; j < label_hash_len; j++) {
            u.labels_group[index * label_hash_len + j] |= v.labels_all[j];
          }
        }
      }
    }
  }

  end_time = clock();
  double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
  const int one_hour = 1 * 3600 * 1000, one_minute = 1 * 60 * 1000;
  cout << "indexTime: " << total_time << " ms";
  if (total_time >= one_hour)
    cout << " = " << total_time / one_hour << " h" << endl;
  else if (total_time >= one_minute)
    cout << " = " << total_time / one_minute << " min" << endl;
  else
    cout << endl;

  long long index_size = 0;
  //计算索引大小
  for (auto &v: vertices) {
    index_size += (sizeof(v.t_in) + sizeof(v.t_out));
    index_size += (long long) sizeof(uint32_t) * label_hash_len;
    index_size += (long long) sizeof(int) * vertex_hash_len;
    if (v.group < 0)
      continue;
    if (v.group == 1) continue;
    index_size += (long long) sizeof(uint32_t) * v.group * vertex_hash_len;
    index_size += (long long) sizeof(uint32_t) * v.group * label_hash_len;
  }
  printf("indexSpace: %.3fMB\n", double(index_size) / (1024 * 1024));
  cout << endl;

  ofstream total_file;
  total_file.open(total_name, ios::app);
  total_file.setf(ios::fixed, ios::floatfield);
  total_file.precision(3);
  total_file << "indexTime: " << total_time << " ms";
  if (total_time >= one_hour)
    total_file << " = " << total_time / one_hour << " h" << endl;
  else if (total_time >= one_minute)
    total_file << " = " << total_time / one_minute << " min" << endl;
  else
    total_file << endl;
  total_file << "indexSpace: " << double(index_size) / (1024 * 1024) << " MB" << endl
             << endl;
  total_file.close();
}

//读取查询语句并计算读取时间
bool read_queries(const string &filename) {
  clock_t start_time, end_time;
  start_time = clock();

  ifstream file;
  file.open(filename, ios::in);
  if (!file.is_open()) {
    cout << "Failed to read query file!" << endl;
    return false;
  }
  queries.clear();
  Query q{};
  string str;
  while (file >> q.origin >> q.destination >> q.pattern >> str) {
    q.bits_label.assign(label_hash_len, 0);
    q.ids_label.clear();
    q.ids_sequence.clear();
    stringstream ss(str);
    while (getline(ss, str, ',')) {
      int l = stoi(str);
      if (q.pattern == 9) {
        if (q.ids_sequence.empty() || l != q.ids_sequence.back()) q.ids_sequence.push_back(l);
      } else if (q.pattern == 10) {
        q.ids_sequence.push_back(l);
      } else {
        q.ids_label.insert(l);
      }
      const uint8_t p = lht.p[l];
      const uint8_t h = lht.h[l];
      q.bits_label[p] |= (1u << h);
    }
    q.outcome = false;
    queries.push_back(q);
  }
  file.close();

  end_time = clock();
  double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;

  auto found = filename.rfind('/'); //返回指定字符最后一次出现的位置
  string name;
  // 实现的功能：若input字符串中含有/,则将result定义为最后一个/之后的部分，否则直接将input赋给result
  if (found != std::string::npos) { // 若rfind查找失败则返回npos
    // begin返回一个迭代器，指向字符串的第一个元素
    // end()返回一个迭代器，指向字符串的末尾（最后一个字符的下一个位置）
    string temp(filename.begin() + found + 1, filename.end());
    name = temp;
  } else {
    name = filename;
  }
  cout << "The time to read " << name << " is: " << total_time << endl;
  return true;
}

void run_queries(const string &total_name) {
  unordered_map<int, unordered_map<int, pair<double, int>>> true_query, false_query;
  clock_t start_time, end_time;

  unordered_map<int, int> true_count;
  for (auto &q: queries) {
    int u = q.origin;
    int v = q.destination;
    int L = (int) q.ids_label.size();
    if (q.pattern > 8) L = (int) q.ids_sequence.size();
//    cout << u << " " << v << " " << q.pattern << " " << L << endl;
    start_time = clock();
    if (u == v) {
      q.outcome = true;
    } else {
      s_pos = 0;
      vis_cur += 2;
      ReachableQuery reach;
      q.outcome = reach.search_dispatch(q.pattern, u, v, q.bits_label, q.ids_label, q.ids_sequence);
    }
    end_time = clock();
    if (q.outcome) {
      if (true_query.find(q.pattern) == true_query.end()) {
        true_count[q.pattern] = 0;
      }
      if (true_query.find(q.pattern) == true_query.end()
          || true_query[q.pattern].find(L) == true_query[q.pattern].end()) {
        true_query[q.pattern][L].first = 0;
        true_query[q.pattern][L].second = 0;
      }
      true_query[q.pattern][L].first += (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
      true_query[q.pattern][L].second++;
      true_count[q.pattern]++;
    } else {
      if (false_query.find(q.pattern) == false_query.end()
          || false_query[q.pattern].find(L) == false_query[q.pattern].end()) {
        false_query[q.pattern][L].first = 0;
        false_query[q.pattern][L].second = 0;
      }
      false_query[q.pattern][L].first += (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
      false_query[q.pattern][L].second++;
    }
  }

  //生出输出文件
  ofstream total_file;
  total_file.open(total_name, ios::app);
  if (!total_file.is_open()) {
    cout << "Failed to create runtime total_file!" << endl;
    return;
  }
  total_file.setf(ios::fixed, ios::floatfield);
  total_file.precision(3);
  total_file << "true-queries runtime:" << endl;
  for (auto &p: true_query) {
    total_file << "pattern: " << p.first << ":" << endl;
    for (auto &t: p.second) {
      total_file << "|L|=" << t.first << ": " << t.second.first << " / " << t.second.second
                 << " = " << t.second.first / t.second.second << ";    ";
    }
    total_file << endl;
  }
  total_file << "false-queries runtime:" << endl;
  for (auto &p: false_query) {
    total_file << "pattern: " << p.first << ":" << endl;
    for (auto &t: p.second) {
      total_file << "|L|=" << t.first << ": " << t.second.first << " / " << t.second.second
                 << " = " << t.second.first / t.second.second << ";    ";
    }
    total_file << endl;
  }
  total_file << endl;
  total_file.close();
}

} // namespace bs

int main(int argc, char *argv[]) {
  using namespace bs;

//  if (argc < 3) {
//    cerr << "command: ./TDR graphFile queryFile" << endl;
//    return 1;
//  }

  string base_path = "G:\\backup_PCR_2026.01.30\\formatGraphs\\";
  string graphs[6] = {"citeseer", "wikitalk", "dblp", "webBerkStan", "Youtube", "superuser"};
  for (auto &name: graphs) {
    //用户输入的图文件
    string input = base_path + name;
    cout << name << ":" << endl;

    //读图
    cout << "********* start to read graph! *********" << endl;
    if (!read_graph(input))
      return 2;

    s = new int[num_V];
    SCC = new int[num_V];

    string query_name = base_path + name + "_pcr";
    string reach_mode = query_name.substr(query_name.length() - 3);
    string total_name = "2static_H_" + reach_mode;
    ofstream total_file;
    total_file.open(total_name, ios::app);
    total_file << name << ":" << endl;
    total_file.close();

    //构建索引
    index_construction(total_name);

    if (!read_queries(query_name))
      return 3;
    cout << "********* start to answer queries! *********" << endl;
    qs.init(num_V);
    if (reach_mode != "lcr") qsSup.init(num_V);
    if (reach_mode == "pcr") ps.init(num_V, label_hash_len);
    AND_visited_count = 0;
    OR_visited_count = 0;
    NOT_visited_count = 0;
    run_queries(total_name);
    cout << "********* Finish! *********" << endl;

    cout << "********* Pruning Strategy Efficiency *********" << endl;
    if (prune_interval != 0)
      cout << "interval: " << prune_interval_success << "/" << prune_interval << "="
           << (double) prune_interval_success / prune_interval << endl;
    if (prune_all_vertices != 0)
      cout << "all vertices: " << prune_all_vertices_success << "/" << prune_all_vertices << "="
           << (double) prune_all_vertices_success / prune_all_vertices << endl;
    if (prune_all_labels != 0)
      cout << "all labels: " << prune_all_labels_success << "/" << prune_all_labels << "="
           << (double) prune_all_labels_success / prune_all_labels << endl;
    if (prune_group_vertices != 0)
      cout << "group vertices: " << prune_group_vertices_success << "/" << prune_group_vertices << "="
           << (double) prune_group_vertices_success / prune_group_vertices << endl;
    if (prune_group_labels != 0)
      cout << "group labels: " << prune_group_labels_success << "/" << prune_group_labels << "="
           << (double) prune_group_labels_success / prune_group_labels << endl;
    if (prune_forward != 0)
      cout << "forward: " << prune_forward_success << "/" << prune_forward << "="
           << (double) prune_forward_success / prune_forward << endl;

    cout << "total_visited_count: " << AND_visited_count << " " << OR_visited_count << " " << NOT_visited_count << endl;
    total_file.open(total_name, ios::app);
    total_file << "total_visited_count: " << AND_visited_count << " " << OR_visited_count << " " << NOT_visited_count
               << endl;
    total_file.close();
  }

  //删除动态数组实现的栈
  delete[] s;
  delete[] SCC;

  return 0;
}
