/*
 * Created by Yang on 2024/4/21.
 * 该版本能够正确处理标签数量大于32的图。
 * 删除了Node结构体中label_all字段，表示无法再根据标签进行剪枝。
 * 合并AND和OR到reachable_by_whole_path函数，合并NOT和IN到reachable_by_all_label函数。
 * 创建函数对象类Reachable，将模式作为参数，从而调用对应的判断函数
 */

#include <algorithm>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <vector>
#include <fstream>
#include "sstream"
#include <unordered_map>
#include <unordered_set>
//#include <map>

using namespace std;

#ifndef K
#define K 5
#endif
#ifndef D
#define D (320 * K)
#endif

namespace bs {

using namespace std;

struct VertexIndex {
  //N_I表示能到达该点的一跳点，N_O表示该点能到达的一跳点，L_O表示该点出边上的标签
  vector<int> neighbor_in;
  vector<int> neighbor_out;
  vector<int> label_out;
  int vis{};
  //记录访问N_I或N_O的位置
  int next{};
  //AND和OR模式下，记录已访问路径上的标记
  unordered_set<int> path_label;
};
vector<VertexIndex> vertices;
struct Query {
  int origin;
  int destination;
  int pattern;
  unordered_set<int> bits_label;
  vector<int> sequence_label;
  bool outcome;
};
vector<Query> queries;
//map<string, Label> labels;
//map<string, Label>::iterator iter;
int vis_cur = 1, cur = 0;
int *s;//定义一个空栈
int s_pos;//定义栈顶元素的位置
vector<int> SCC;//存储强连通分量除根顶点外其他顶点
unsigned long long AND_visited_count = 0, OR_visited_count = 0, NOT_visited_count = 0, seq_visited_count = 0;

//使用函数对象类实现 根据参数调用不同函数并多次执行
class Reachable {
 public:
  const int &uid;
  const int &vid;
  unordered_set<int> &idsL;
  vector<int> &idsSeq;
  bool operator()(int param) const {
    switch (param) {
      case 1: return reachable_by_whole_path(uid, vid, idsL, true);
      case 2: return reachable_by_whole_path(uid, vid, idsL, false);
      case 3: return reachable_by_all_label(uid, vid, idsL, true);
      case 4: return reachable_by_all_label(uid, vid, idsL, false);
      case 5: return reachable_by_sequence1(uid, vid, idsSeq);
      case 6: return reachable_by_sequence2(uid, vid, idsSeq);
      default:std::cout << "Invalid parameter" << std::endl;
        break;
    }
    return false;
  }
 private:
  //AND模式和OR模式,ptn=true时为AND模式，ptn=false时为OR模式
  //函数名意为整体路径约束可达，要求整个路径的标签满足查询标签集的约束
  static bool reachable_by_whole_path(const int &u,
                                      const int &vid,
                                      const unordered_set<int> &label_ids,
                                      const bool ptn) {
    //v入度为0，则一定不可达
    if (vertices[vid].neighbor_in.empty()) {
      return false;
    }

    //将v替换成其等价点
    VertexIndex &v = vertices[vid];

    s[s_pos] = u;
    //清空起始顶点的path_label，防止之前的查询语句中已经访问过起始顶点
    vertices[u].path_label.clear();
    bool isAdd = true;
    while (s_pos != -1) {//当栈为空时，遍历结束
      VertexIndex &m = vertices[s[s_pos]];//返回栈顶元素, 但不删除该元素
      int n = (int) m.neighbor_out.size();

      if (isAdd) {
        isAdd = false;
        m.next = 0;
        m.vis = vis_cur;
        if (ptn) {
          AND_visited_count++;
        } else {
          OR_visited_count++;
        }


        //找到具体路径且标签满足条件则可达，否则，重新查找路径
        if (s[s_pos] == vid) {
          if (ptn ? m.path_label == label_ids : !m.path_label.empty()) {
            return true;
          }
          --s_pos;//弹出栈顶元素, 但不返回其值
          m.vis = vis_cur + 1;
          continue;
        }
      }

      //当通过标签无法判断时，需要进行遍历
      bool finished = true;
      while (m.next < n) {
        //如果路径上的标签不在给定标签集中，则进行剪枝
        int tid = m.neighbor_out[m.next];
        VertexIndex &t = vertices[tid];
        m.next++;
        if (t.vis == vis_cur) continue;

        int l = m.label_out[m.next];
        if (t.vis < vis_cur) {
          //如果后继顶点未被访问，则将后继顶点入栈
          isAdd = true;
          finished = false;
          s[++s_pos] = tid;
          //后继顶点t的path_label将在当前顶点m的path_label的基础上
          t.path_label = m.path_label;
          //若边m-t上的标签l在查询标签集里，则将l加入到t的path_label中，否则，不加入
          if (label_ids.find(l) != label_ids.end()) t.path_label.insert(l);
          break;
        } else {
          //检查边m-t上的标签l是否有新增标签，若有，则需要重新入栈
          if (label_ids.find(l) != label_ids.end() && t.path_label.find(l) == t.path_label.end()) {
            t.path_label = m.path_label;
            t.path_label.insert(l);
            isAdd = true;
            finished = false;
            s[++s_pos] = tid;
            break;
          }
        }
      }
      if (finished) {
        m.vis = vis_cur + 1;
        isAdd = false;
        --s_pos;//弹出栈顶元素
      }
    }
    return false;
  }

  //NOT模式和IN模式,ptn=true时为NOT模式，ptn=false时为IN模式
  //函数名意为全路径标签约束可达，要求路径上所有标签都要满足查询标签集的约束
  static bool reachable_by_all_label(const int &u,
                                     const int &vid,
                                     const unordered_set<int> &label_ids,
                                     const bool ptn) {
    //v入度为0，则一定不可达
    if (vertices[vid].neighbor_in.empty()) {
      return false;
    }

    //将v替换成其等价点
    VertexIndex &v = vertices[vid];

    s[s_pos] = u;
    bool isAdd = true;
    while (s_pos != -1) {//当栈为空时，遍历结束
      VertexIndex &m = vertices[s[s_pos]];//返回栈顶元素, 但不删除该元素
      int n = (int) m.neighbor_out.size();

      if (isAdd) {
        isAdd = false;
        m.next = 0;
        m.vis = vis_cur;
        if (ptn) {
          NOT_visited_count++;
        }

        //找到具体路径则可达
        if (s[s_pos] == vid) {
          return true;
        }
      }

      //当通过标签无法判断时，需要进行遍历
      bool finished = true;
      while (m.next < n) {
        int tid = m.neighbor_out[m.next];
        VertexIndex &t = vertices[tid];
        int tl = m.label_out[m.next];
        m.next++;
        //只有边m-t上的标签不在给定标签集中，并且后继顶点t未被访问过，才将t入栈
        if ((ptn ? label_ids.find(tl) == label_ids.end() : label_ids.find(tl) != label_ids.end()) && t.vis != vis_cur) {
          isAdd = true;
          finished = false;
          s[++s_pos] = tid;
          break;
        }
      }
      if (finished) {
        isAdd = false;
        --s_pos;//弹出栈顶元素
      }
    }

    return false;
  }

  /*
   * 正则路径查询中标签顺序约束为a+b+c+……形式的查询
   */
  static bool reachable_by_sequence1(const int &uid, const int &vid, const vector<int> &idsSeq) {
    const VertexIndex &v = vertices[vid];
    //v入度为0，则一定不可达
    if (v.neighbor_in.empty()) {
      return false;
    }

    //查询集的标签数量
    int ql_num = (int) idsSeq.size();
    //记录等待匹配的标签位置
    int needed_pos = 0;
    vector<int> path_labels;
    unordered_map<int, unordered_set<int>> saved_labels;

    s[s_pos] = uid;
    bool isAdd = true;
    while (s_pos != -1) { //当栈为空时，遍历结束
      VertexIndex &m = vertices[s[s_pos]];//返回栈顶元素, 但不删除该元素
//      cout<<endl<<s[s_pos]<<":";
      if (path_labels.empty()) needed_pos = 0;
      else {
        int l = path_labels.back();
        int current_pos = (needed_pos > 0) ? (needed_pos - 1) : needed_pos;
        if (l != idsSeq[current_pos] && l != idsSeq[ql_num - 1]) {
          needed_pos = current_pos;
        }
      }
      int n = (int) m.neighbor_out.size();

      if (isAdd) {
        isAdd = false;
        seq_visited_count++;
//        cout<<path_labels.size()<<" "<<s_pos<<endl;
//        for (auto a: path_labels) {
//          cout << a << " ";
//        }
//        cout << endl;
//        for (int i = 0; i < s_pos; ++i) {
//          cout<<s[i]<<",";
//        }
//        cout<<endl;
        //找到具体路径则可达
        if (s[s_pos] == vid) {
          if (path_labels.back() == idsSeq[ql_num - 1])
            return true;
          m.vis = vis_cur + 1;
          --s_pos;//弹出栈顶元素
          path_labels.pop_back();
          continue;
        }
        m.next = 0;
        m.vis = vis_cur;
      }

      //当通过标签无法判断时，需要进行遍历
      bool finished = true;
      while (m.next < n) {
        int tid = m.neighbor_out[m.next];
        VertexIndex &t = vertices[tid];
        int tl = m.label_out[m.next];
        m.next++;
        if (t.vis >= vis_cur) continue;
//        cout<<tid<<" ";
        if (tl == idsSeq[needed_pos]) {
          bool revisited = true;
          if (t.vis > vis_cur) {
            for (auto &l: saved_labels[tid]) {
              if (l == tl) {
                revisited = false;
                break;
              }
            }
          }
          if (!revisited) continue;
          isAdd = true;
          finished = false;
          //need_pos的上界是数组idsSeq的末尾
          needed_pos = (needed_pos < ql_num - 1) ? (needed_pos + 1) : needed_pos;
          s[++s_pos] = tid;
          path_labels.push_back(tl);
          saved_labels[tid].insert(tl);
          break;
        }
        if (!path_labels.empty() && path_labels.back() == idsSeq[ql_num - 1]) continue;
        int current_pos = (needed_pos > 0) ? (needed_pos - 1) : needed_pos;
        if (tl == idsSeq[current_pos]) {
          bool revisited = true;
          if (t.vis > vis_cur) {
            for (auto &l: saved_labels[tid]) {
              if (l == tl) {
                revisited = false;
                break;
              }
            }
          }
          if (!revisited) continue;
          isAdd = true;
          finished = false;
          s[++s_pos] = tid;
          path_labels.push_back(tl);
          saved_labels[tid].insert(tl);
          break;
        }
      }
      if (finished) {
        isAdd = false;
        m.vis = vis_cur + 1;
        --s_pos;//弹出栈顶元素
        if (!path_labels.empty()) path_labels.pop_back();
      }
    }
    return false;
  }
  /*
    * 正则路径查询中标签顺序约束为(abc……)+形式的查询
    */
  static bool reachable_by_sequence2(const int &uid, const int &vid, const vector<int> &idsSeq) {
    const VertexIndex &v = vertices[vid];
    //v入度为0，则一定不可达
    if (v.neighbor_in.empty()) {
      return false;
    }

    //查询集的标签数量
    int ql_num = (int) idsSeq.size();
    //记录等待匹配的标签位置
    int needed_pos = 0;
    vector<int> path_labels;
    unordered_map<int, unordered_set<int>> saved_labels;

    s[s_pos] = uid;
    bool isAdd = true;
    while (s_pos != -1) { //当栈为空时，遍历结束
      VertexIndex &m = vertices[s[s_pos]];//返回栈顶元素, 但不删除该元素
      int n = (int) m.neighbor_out.size();
//      cout<<s_pos<<" "<<needed_pos<<endl;

      if (isAdd) {
        isAdd = false;
        m.next = 0;
        m.vis = vis_cur;
        seq_visited_count++;

        //找到具体路径则可达
        if (s[s_pos] == vid) {
          if (path_labels.back() == idsSeq[ql_num - 1])
            return true;
          m.vis = vis_cur + 1;
          --s_pos;//弹出栈顶元素
          path_labels.pop_back();
          // 实现循环递减
          needed_pos = (needed_pos - 1 + ql_num) % ql_num;
          continue;
        }
      }

      //当通过标签无法判断时，需要进行遍历
      bool finished = true;
      while (m.next < n) {
        int tid = m.neighbor_out[m.next];
        VertexIndex &t = vertices[tid];
        int tl = m.label_out[m.next];
        m.next++;
        if (t.vis == vis_cur) continue;

        if (tl == idsSeq[needed_pos]) {
          bool revisited = true;
          if (t.vis > vis_cur) {
            for (auto &l: saved_labels[tid]) {
              if (l == tl) {
                revisited = false;
                break;
              }
            }
          }
          if (!revisited) continue;
          isAdd = true;
          finished = false;
          // 实现循环递增
          needed_pos = (needed_pos + 1) % ql_num;
          s[++s_pos] = tid;
          path_labels.push_back(tl);
          saved_labels[tid].insert(tl);
          break;
        }
      }
      if (finished) {
        isAdd = false;
        m.vis = vis_cur + 1;
        --s_pos;//弹出栈顶元素
        if (!path_labels.empty()) path_labels.pop_back();
        // 实现循环递减
        needed_pos = (needed_pos - 1 + ql_num) % ql_num;
      }
    }
    return false;
  }

};

//读取图并计算读取时间
bool read_graph(const string &filename) {
  clock_t start_time, end_time;
  start_time = clock();

  ifstream file;
  file.open(filename, ios::in);
  if (!file.is_open()) {
    cout << "Failed to read graph file!" << endl;
    return false;
  }
  // V=vertices, TE=topology edges, LE=labeled edges, L=labels
  string str;
  int num_V = 0, num_TE = 0, num_LE = 0, num_L = 0;
  file >> str >> num_V >> str >> num_TE >> str >> num_LE >> str >> num_L;

  // initialize vertices
  vertices.clear();
  vertices.resize(num_V);//resize(n)的作用是调整容器的大小，使其包含n个元素
  for (auto &node: vertices) {//初始化
    node.next = 0;
    node.vis = 0;
  }

  int u, v, v_pos = 0, l_pos = 0;
  while (file >> u >> v >> str) {
    stringstream ss(str);
    while (getline(ss, str, ',')) {
      int lid = stoi(str);
      vertices[u].neighbor_out.push_back(v);
      vertices[u].label_out.push_back(lid);
      vertices[v].neighbor_in.push_back(u);
    }
  }
  file.close();

  end_time = clock();
  double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
  printf("read time(graph): %.3fs\n", total_time);

  s = new int[vertices.size()];
  return true;
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
  Query q{};
  string str;
  queries.clear();
  while (file >> q.origin >> q.destination >> q.pattern >> str) {
    q.bits_label.clear();
    q.sequence_label.clear();
    if (q.pattern == 9) q.pattern = 5;
    else if (q.pattern == 10) q.pattern = 6;
    else if (q.pattern > 4) q.pattern -= 4;
    stringstream ss(str);
    while (getline(ss, str, ',')) {
      int l = stoi(str);
      if (q.pattern == 5) {
        if (q.sequence_label.empty() || l != q.sequence_label.back())
          q.sequence_label.push_back(l);
      } else if (q.pattern == 6) {
        q.sequence_label.push_back(l);
      } else q.bits_label.insert(l);
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
  if (found != std::string::npos) {// 若rfind查找失败则返回npos
    // begin返回一个迭代器，指向字符串的第一个元素
    // end()返回一个迭代器，指向字符串的末尾（最后一个字符的下一个位置）
    string temp(filename.begin() + found + 1, filename.end());
    name = temp;
  } else {
    name = filename;
  }
  cout << "The time to read " << name << " is: " << total_time << endl;
//  printf("read time(query): %.3fs\n", total_time);

  return true;
}

void run_queries(const string &filename) {
  double total_time;
  unordered_map<int, unordered_map<int, pair<double, int>>> true_query, false_query;
  bool time_out;
  clock_t start_time, end_time;
//  ofstream file;
//  file.open(filename, ios::app);
//  if (!file.is_open()) {
//    cout << "Failed to create runtime file!" << endl;
//    return;
//  }
//  file.setf(ios::fixed, ios::floatfield);
//  file.precision(3);
  const int cutoff = 12 * 3600 * 1000;//定义临界值为12小时
  const int one_hour = 1 * 3600 * 1000, one_minute = 1 * 60 * 1000;
  time_out = false;
  total_time = 0;
  for (auto &q: queries) {
    int u = q.origin;
    int v = q.destination;
//    cout << u << " " << v << " " << q.pattern << endl;
    start_time = clock();
    if (u == v) {
      q.outcome = true;
    } else {
      s_pos = 0;
      vis_cur += 2;
      Reachable reach{u, v, q.bits_label, q.sequence_label};
      q.outcome = reach(q.pattern);
    }
    end_time = clock();
//    cout << u << " " << v << " " << q.outcome << endl;
    total_time += (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
    int L = (int) q.bits_label.size();
    if (q.outcome) {
      if (true_query.find(q.pattern) == true_query.end()
          || true_query[q.pattern].find(L) == true_query[q.pattern].end()) {
        true_query[q.pattern][L].first = 0;
        true_query[q.pattern][L].second = 0;
      }
      true_query[q.pattern][L].first += (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
      true_query[q.pattern][L].second++;
    } else {
      if (false_query.find(q.pattern) == false_query.end()
          || false_query[q.pattern].find(L) == false_query[q.pattern].end()) {
        false_query[q.pattern][L].first = 0;
        false_query[q.pattern][L].second = 0;
      }
      false_query[q.pattern][L].first += (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
      false_query[q.pattern][L].second++;
    }
    if (total_time - cutoff > 0) {
      time_out = true;
      break;
    }
  }
//  if (time_out) {
//    cout << "time out! " << endl;
//    file << "time out! " << endl;
//  } else {
//    file << "true-queries runtime:" << endl;
//    for (auto &p: true_query) {
//      file << "pattern: " << p.first << ":" << endl;
//      for (auto &t: p.second) {
//        file << "|L|=" << t.first << ": " << t.second.first << " / " << t.second.second
//             << " = " << t.second.first / t.second.second << ";    ";
//      }
//      file << endl;
//    }
//    file << "false-queries runtime:" << endl;
//    for (auto &p: false_query) {
//      file << "pattern: " << p.first << ":" << endl;
//      for (auto &t: p.second) {
//        file << "|L|=" << t.first << ": " << t.second.first << " / " << t.second.second
//             << " = " << t.second.first / t.second.second << ";    ";
//      }
//      file << endl;
//    }
//  }
//  file << endl;
//  file.close();
}

}

int main(int argc, char *argv[]) {
  using namespace bs;

//  if (argc < 2) {
//    cerr << "command: ./DFS filename queryFile" << endl;
//    return 1;
//  }

  //用户输入的图文件
  string base_path = "G:\\backup_PCR_2026.01.30\\formatGraphs\\";
  string graphs[6] = {"citeseer", "wikitalk", "dblp", "webBerkStan", "Youtube", "superuser"};
//  string graphs[3] = {"webBerkStan", "Youtube", "superuser"};
  for (auto &name: graphs) {
    string input = base_path + name;
    cout << name << ":" << endl;

    string query_name = base_path + name + "_rpq";
    string reach_mode = query_name.substr(query_name.length() - 3);
    string result_name = "1total_DFS_" + reach_mode;
    ofstream outfile;
    outfile.open(result_name, ios::app);
    outfile << name << ":" << endl;
    outfile.close();

    //读图
    cout << "********* start to read graph! *********" << endl;
    if (!read_graph(input))
      return 2;

    if (!read_queries(query_name))
      return 3;
    cout << "********* start to answer queries! *********" << endl;
    AND_visited_count = 0;
    OR_visited_count = 0;
    NOT_visited_count = 0;
    seq_visited_count = 0;
    run_queries(result_name);
//    cout << "total_visited_count: " << AND_visited_count << " " << OR_visited_count << " " << NOT_visited_count << endl;
    outfile.open(result_name, ios::app);
//    outfile << "total_visited_count: " << AND_visited_count << " " << OR_visited_count << " " << NOT_visited_count
//            << endl;
    outfile << "sequence visited count: " << seq_visited_count << endl;
    cout << "sequence visited count: " << seq_visited_count << endl;
    outfile.close();
    cout << "************* finish! *************" << endl << endl;
  }

  //删除动态数组实现的栈
  delete[]s;

  return 0;
}