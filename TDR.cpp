#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <climits>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <new>
#include <stdexcept>
#include <vector>

namespace bs {
    using namespace std;
    vector<int> source_offset;
    vector<int> targets;
    vector<int> target_offset;
    vector<int> labels;
    vector<int> graph_label_count;
    int num_V = 0, num_TE = 0, num_LE = 0, num_L = 0;

    int vis_cur = 1, cur = 0;
    int *s;
    int s_capacity = 0;
    int s_pos;
    int *SCC;
    int SCC_pos;
    constexpr int vertex_hash_len = 32;
    int label_hash_len = 1;
    constexpr int forward_default_k = 2;
    constexpr int UINT_bits = sizeof(uint32_t) * 8;
    bool build_group_label_index = true;
    bool build_forward_label_index = true;
    constexpr int output_max_pattern = 10;
    constexpr int output_max_label_length = 6;

    enum class IndexProfile { TDR, OnlyH, OnlyV };

    auto active_index_profile = IndexProfile::TDR;

    int vertex_hash_leaf = 0;
    int vertex_hash_parent = -1;
    int vertex_hash_value = 0;

    inline int vertex_hash_block(const uint32_t hv) {
        return static_cast<int>(hv & vertex_hash_len - 1);
    }

    inline int vertex_hash_bit(const uint32_t hv) {
        return static_cast<int>(hv >> 5 & UINT_bits - 1);
    }
    enum class QueryProfile { Baseline, Full, OnlyV, OnlyH };

    auto active_profile = QueryProfile::Full;
    long long query_edge_visits = 0;

    inline void ensure_search_stack_capacity(const int required) {
        if (required <= s_capacity) return;
        int next_capacity = std::max(required, std::max(16, s_capacity * 2));
        int *next = new int[next_capacity];
        if (s != nullptr && s_pos >= 0) {
            std::memcpy(next, s, static_cast<size_t>(s_pos + 1) * sizeof(int));
        }
        delete[] s;
        s = next;
        s_capacity = next_capacity;
    }

    inline bool enable_base_pruning() {
        return active_profile != QueryProfile::Baseline;
    }

    inline bool enable_group_pruning() {
        return build_group_label_index && (active_profile == QueryProfile::Full || active_profile ==
                                           QueryProfile::OnlyH);
    }

    inline bool enable_forward_pruning() {
        return build_forward_label_index && (active_profile == QueryProfile::Full || active_profile ==
                                             QueryProfile::OnlyV);
    }
    struct VisitRatioStats {
        long long baseline_edges = 0;
        long long profile_edges = 0;
        long long mismatches = 0;

        void add(const long long baseline, const long long profile, const bool same_outcome) {
            baseline_edges += baseline;
            profile_edges += profile;
            if (!same_outcome) {
                mismatches++;
            }
        }

        [[nodiscard]] double ratio() const {
            return baseline_edges == 0 ? 0.0 : static_cast<double>(profile_edges) / static_cast<double>(baseline_edges);
        }
    };
    using VertexBits = vector<uint32_t>;

    struct VertexIndex {
        bool root{};
        int t_in{}, t_out{};
        bool leaf{};

        union Storage {
            VertexBits vertices_all;
            uint32_t leaf_hash;

            Storage() {
            }

            ~Storage() {
            }
        } storage;

        vector<uint32_t> labels_all;

        VertexIndex() { new(&storage.vertices_all) VertexBits(); }
        ~VertexIndex() { if (!leaf) storage.vertices_all.~VertexBits(); }
        VertexIndex(const VertexIndex &other) { copy_from(other); }

        VertexIndex &operator=(const VertexIndex &other) {
            if (this == &other) return *this;
            if (!leaf) storage.vertices_all.~VertexBits();
            copy_from(other);
            return *this;
        }

        VertexIndex(VertexIndex &&other) noexcept { move_from(std::move(other)); }

        VertexIndex &operator=(VertexIndex &&other) noexcept {
            if (this == &other) return *this;
            if (!leaf) storage.vertices_all.~VertexBits();
            move_from(std::move(other));
            return *this;
        }

        void make_leaf(const uint32_t hash) {
            if (!leaf) storage.vertices_all.~VertexBits();
            leaf = true;
            storage.leaf_hash = hash;
            vector<uint32_t>().swap(labels_all);
        }

        void make_internal() {
            if (leaf) {
                new(&storage.vertices_all) VertexBits();
                leaf = false;
            }
        }

        VertexBits &vertices_all() { return storage.vertices_all; }
        const VertexBits &vertices_all() const { return storage.vertices_all; }
        uint32_t leaf_hash() const { return storage.leaf_hash; }

    private:
        void copy_from(const VertexIndex &other) {
            root = other.root;
            t_in = other.t_in;
            t_out = other.t_out;
            leaf = other.leaf;
            labels_all = other.labels_all;
            if (leaf) storage.leaf_hash = other.storage.leaf_hash;
            else new(&storage.vertices_all) VertexBits(other.storage.vertices_all);
        }

        void move_from(VertexIndex &&other) {
            root = other.root;
            t_in = other.t_in;
            t_out = other.t_out;
            leaf = other.leaf;
            labels_all = std::move(other.labels_all);
            if (leaf) storage.leaf_hash = other.storage.leaf_hash;
            else new(&storage.vertices_all) VertexBits(std::move(other.storage.vertices_all));
        }
    };

    struct VertexIndexH {
        int group{};
        vector<uint32_t> labels_group;
        vector<uint32_t> vertices_group;
        vector<int> vertices_descriptor, labels_descriptor;

        static uint32_t word(const vector<uint32_t> &data, const vector<int> &descriptor,
                             int width, int offset, bool vertex) {
            if (descriptor.empty()) return data[offset];
            int d = descriptor[offset / width];
            if (d == -1) return 0;
            if (d == -2) return vertex ? UINT32_MAX : label_universe_word(offset % width);
            return data[d + offset % width];
        }

        static uint32_t label_universe_word(int word);
    };

    struct VertexIndexV {
        int group{};
        vector<uint32_t> labels_forward;
        vector<uint8_t> forward_depth;
        vector<int> forward_offset;
        enum LayerStorage : int { EMPTY = -1, FULL = -2 };

        vector<int> forward_bitmap_offset;
    };

    vector<VertexIndex> indices_base;
    vector<VertexIndexH> indices_horizontal;
    vector<VertexIndexV> indices_vertical;
    vector<uint32_t> label_universe_bits;
    long long horizontal_dense_bytes = 0;

    uint32_t VertexIndexH::label_universe_word(int word) { return label_universe_bits[word]; }
    void compact_horizontal_words(vector<uint32_t> &words, vector<int> &descriptor,
                                  int width, bool vertex) {
        if (words.empty()) return;
        vector<uint32_t> packed;
        vector<int> offsets;
        std::map<vector<uint32_t>, int> stored;
        for (size_t start = 0; start < words.size(); start += width) {
            bool empty = true, full = true;
            for (int w = 0; w < width; ++w) {
                empty &= words[start + w] == 0;
                full &= words[start + w] == (vertex ? UINT32_MAX : label_universe_bits[w]);
            }
            if (empty) offsets.push_back(-1);
            else if (full) offsets.push_back(-2);
            else {
                vector<uint32_t> block(words.begin() + start, words.begin() + start + width);
                auto existing = stored.find(block);
                if (existing != stored.end()) offsets.push_back(existing->second);
                else {
                    offsets.push_back(static_cast<int>(packed.size()));
                    stored.emplace(std::move(block), static_cast<int>(packed.size()));
                    packed.insert(packed.end(), words.begin() + start, words.begin() + start + width);
                }
            }
        }
        if ((packed.size() + offsets.size()) * sizeof(uint32_t) >= words.size() * sizeof(uint32_t)) return;
        words.swap(packed);
        descriptor.swap(offsets);
    }

    struct QueryHorizontalWords {
        const VertexIndexH *index = nullptr;
        bool vertex = false;

        uint32_t operator[](int offset) const {
            return VertexIndexH::word(vertex ? index->vertices_group : index->labels_group,
                                      vertex ? index->vertices_descriptor : index->labels_descriptor,
                                      vertex ? vertex_hash_len : label_hash_len, offset, vertex);
        }
    };

    inline int forward_group_depth(const VertexIndexV &U, const int gid) {
        if (gid < 0 || gid >= U.group || U.forward_depth.empty()) return 0;
        return U.forward_depth[gid];
    }

    inline int forward_group_offset(const VertexIndexV &U, const int gid) {
        if (gid < 0 || gid >= U.group || U.forward_offset.empty()) return 0;
        return static_cast<int>(U.forward_offset[gid]) * label_hash_len;
    }

    inline int forward_total_words(const VertexIndexV &U) {
        return U.forward_offset.empty() ? 0 : static_cast<int>(U.forward_offset.back()) * label_hash_len;
    }

    inline int forward_layer_slot(const VertexIndexV &U, const int gid, const int layer) {
        if (gid < 0 || gid >= U.group || layer < 0 || U.forward_offset.empty()) return -1;
        return static_cast<int>(U.forward_offset[gid]) + layer;
    }

    inline bool forward_layer_is_dense(const VertexIndexV &U, const int gid, const int layer) {
        const int slot = forward_layer_slot(U, gid, layer);
        if (slot < 0 || layer >= forward_group_depth(U, gid)) return false;
        if (!U.forward_bitmap_offset.empty())
            return U.forward_bitmap_offset[slot] == VertexIndexV::FULL;
        if (U.labels_forward.empty() || label_universe_bits.empty()) return false;
        for (int word = 0; word < label_hash_len; ++word) {
            if ((U.labels_forward[slot * label_hash_len + word] & label_universe_bits[word]) !=
                label_universe_bits[word])
                return false;
        }
        return true;
    }
    inline uint32_t forward_layer_word(const VertexIndexV &U, const int gid,
                                       const int layer, const int word) {
        const int slot = forward_layer_slot(U, gid, layer);
        if (slot < 0 || layer >= forward_group_depth(U, gid)) return 0;
        if (U.forward_bitmap_offset.empty())
            return U.labels_forward[slot * label_hash_len + word];
        const int offset = U.forward_bitmap_offset[slot];
        if (offset == VertexIndexV::EMPTY) return 0;
        if (offset == VertexIndexV::FULL) return label_universe_bits[word];
        return U.labels_forward[offset + word];
    }

    inline bool label_bitset_empty(const uint32_t *bits) {
        if (bits == nullptr || label_universe_bits.empty()) return true;
        for (int i = 0; i < label_hash_len; ++i) {
            if ((bits[i] & label_universe_bits[i]) != 0) return false;
        }
        return true;
    }

    inline bool label_bitset_full(const uint32_t *bits) {
        if (bits == nullptr || label_universe_bits.empty()) return false;
        for (int i = 0; i < label_hash_len; ++i) {
            if ((bits[i] & label_universe_bits[i]) != label_universe_bits[i]) return false;
        }
        return true;
    }

    inline void or_words(uint32_t *dst, const uint32_t *src, const int n) {
        for (int i = 0; i < n; ++i) dst[i] |= src[i];
    }

    inline int group_remaining_edges(const int uid, const int gid) {
        const int begin = source_offset[uid];
        const int end = source_offset[uid + 1];
        const int first = begin + gid;
        if (gid < 0 || first >= end) return 0;
        int group = 1;
        if (enable_group_pruning() && uid < static_cast<int>(indices_horizontal.size()))
            group = std::max(1, indices_horizontal[uid].group);
        else if (enable_forward_pruning() && uid < static_cast<int>(indices_vertical.size()))
            group = std::max(1, indices_vertical[uid].group);
        return (end - 1 - first) / group + 1;
    }
    struct Query {
        int origin;
        int destination;
        int pattern;
        vector<uint32_t> bits_label;
        vector<int> ids_label;
        vector<int> ids_sequence;
        bool outcome;
    };

    vector<Query> queries;
    struct ProfileRuntimeStats {
        map<int, vector<pair<double, int> > > true_qtime;
        map<int, vector<pair<double, int> > > false_qtime;
        map<int, int> true_count;
        map<int, double> true_time;
        map<int, int> total_count;
        map<int, double> total_time;

        void add(const int pattern, const int L, const bool outcome, const double qtime) {
            map<int, vector<pair<double, int> > > &bucket = outcome ? true_qtime : false_qtime;
            if (bucket.find(pattern) == bucket.end()) {
                bucket[pattern] = vector<pair<double, int> >(output_max_label_length, {0.0, 0});
            }
            vector<pair<double, int> > &items = bucket[pattern];
            if (static_cast<int>(items.size()) < L) {
                cerr << "One query is out range of 6 labels." << endl;
                return;
            }
            items[L - 1].first += qtime;
            items[L - 1].second++;
            if (outcome) {
                true_count[pattern]++;
                true_time[pattern] += qtime;
            }
            total_count[pattern]++;
            total_time[pattern] += qtime;
        }
    };
    inline int mix_hash(int x) {
        uint32_t value = static_cast<uint32_t>(x);
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        value *= 0x846ca68bu;
        value ^= value >> 16;
        return static_cast<int>(value);
    }

    inline void reset_vertex_hash_state() {
        vertex_hash_leaf = 0;
        vertex_hash_parent = -1;
        vertex_hash_value = 0;
    }

    inline int vertex_hash(const int &vertex_id) {
        return mix_hash(vertex_id + 1);
    }
    template<class Fn>
    void for_each_label(const string &str, Fn &&fn) {
        int value = 0;
        bool has_digit = false;
        bool negative = false;
        for (const char c: str) {
            if (c >= '0' && c <= '9') {
                value = value * 10 + (c - '0');
                has_digit = true;
            } else if (c == '-') {
                negative = true;
            } else if (c == ',') {
                if (has_digit) fn(negative ? -value : value);
                value = 0;
                has_digit = false;
                negative = false;
            }
        }
        if (has_digit) fn(negative ? -value : value);
    }

    inline bool contains_label(const std::vector<int> &idsL, const int label) {
        return std::find(idsL.begin(), idsL.end(), label) != idsL.end();
    }
    inline void configure_index_components(const string & ,
                                           const IndexProfile profile = IndexProfile::TDR) {
        active_index_profile = profile;
        build_group_label_index = profile != IndexProfile::OnlyV;
        build_forward_label_index = profile != IndexProfile::OnlyH;
    }

    std::vector<int> horizontal_group_count;
    inline int degree_group_cap(const int out_degree) {
        if (out_degree <= 3) return 1;
        return static_cast<int>(std::round(std::sqrt(static_cast<double>(out_degree))));
    }

    inline int prepared_horizontal_group_count(const int uid, const int out_degree) {
        if (out_degree <= 0) return 0;
        if (active_index_profile == IndexProfile::OnlyV) return 1;
        if (uid >= 0 && uid < static_cast<int>(horizontal_group_count.size()))
            return horizontal_group_count[uid];
        return degree_group_cap(out_degree);
    }

    inline int popcount32(uint32_t bits) {
        int cnt = 0;
        while (bits != 0) {
            bits &= bits - 1;
            ++cnt;
        }
        return cnt;
    }
    static struct LabelHashTable {
        std::vector<int> p;
        std::vector<uint8_t> h;
        std::vector<uint32_t> mask;

        void init(const int numL, int ) {
            p.resize(numL);
            h.resize(numL);
            mask.resize(numL);
            for (int l = 0; l < numL; ++l) {
                const int blk = l >> 5;
                p[l] = blk;
                h[l] = l & UINT_bits - 1;
                mask[l] = 1u << h[l];
            }
        }
    } lht;

    inline int popcount64(uint64_t bits) {
        int cnt = 0;
        while (bits != 0) {
            bits &= bits - 1;
            ++cnt;
        }
        return cnt;
    }
    void prepare_horizontal_group_features(const std::vector<int> & ) {
        horizontal_group_count.assign(num_V, 1);
        for (int uid = 0; uid < num_V; ++uid) {
            horizontal_group_count[uid] = degree_group_cap(source_offset[uid + 1] - source_offset[uid]);
        }
    }
    inline void rebuild_forward_offsets(VertexIndexV &U) {
        U.forward_offset.assign(U.forward_depth.size() + 1, 0);
        int prefix = 0;
        for (int gid = 0; gid < static_cast<int>(U.forward_depth.size()); ++gid) {
            U.forward_offset[gid] = prefix;
            prefix += static_cast<int>(U.forward_depth[gid]);
        }
        U.forward_offset[U.forward_depth.size()] = prefix;
    }
    inline void prepare_forward_layout(VertexIndexV &U, const int group) {
        U.group = group;
        U.forward_depth.clear();
        U.forward_offset.clear();
        U.forward_bitmap_offset.clear();
        if (!build_forward_label_index || group <= 0) return;
        U.forward_depth.assign(group, forward_default_k);
        rebuild_forward_offsets(U);
    }
    void compact_forward_layers(VertexIndexV &U) {
        if (!build_forward_label_index || U.group <= 0 || U.labels_forward.empty() ||
            U.forward_depth.empty() || !U.forward_bitmap_offset.empty())
            return;

        const int layers = forward_total_words(U) / label_hash_len;
        std::vector<int> offsets(layers, VertexIndexV::EMPTY);
        size_t bitmap_words = 0;
        for (int slot = 0; slot < layers; ++slot) {
            const uint32_t *bits = &U.labels_forward[slot * label_hash_len];
            if (label_bitset_empty(bits)) continue;
            if (label_bitset_full(bits)) {
                offsets[slot] = VertexIndexV::FULL;
            } else {
                offsets[slot] = static_cast<int>(bitmap_words);
                bitmap_words += label_hash_len;
            }
        }
        if (bitmap_words * sizeof(uint32_t) + offsets.size() * sizeof(int) >=
            U.labels_forward.size() * sizeof(uint32_t))
            return;

        std::vector<uint32_t> bitmaps(bitmap_words);
        for (int slot = 0; slot < layers; ++slot) {
            if (offsets[slot] < 0) continue;
            std::copy_n(&U.labels_forward[slot * label_hash_len], label_hash_len,
                        &bitmaps[offsets[slot]]);
        }
        U.labels_forward.swap(bitmaps);
        U.forward_bitmap_offset.swap(offsets);
    }

    void finalize_forward_density_flags() {
        if (!build_forward_label_index) return;
        for (auto &v: indices_vertical) compact_forward_layers(v);
    }
    struct BuildState {
        std::vector<int> vis;
        std::vector<int> next;
        std::vector<int> low;
        std::vector<int> parent;
        std::vector<uint32_t> hash_value;

        void init(const int n) {
            vis.assign(n, 0);
            next.assign(n, 0);
            low.assign(n, 0);
            parent.assign(n, -1);
            hash_value.assign(n, 0);
        }
    } bst;
    struct QueryState {
        std::vector<int> vis;
        std::vector<int> next;
        std::vector<int> gid;
        std::vector<int> label_edge;
        std::vector<uint32_t> scratch_bits;

        void init(const int &n) {
            vis.assign(n, 0);
            next.resize(n);
            gid.resize(n);
            label_edge.assign(n, -1);
            scratch_bits.resize(label_hash_len);
        }
    } qs;

    inline void reset_query_marks_if_needed() {
        if (vis_cur < INT_MAX - 16) return;
        std::fill(qs.vis.begin(), qs.vis.end(), 0);
        vis_cur = 1;
    }
    struct QueryStateSupplement {
        std::vector<int> l_next;
        std::vector<uint8_t> topology;

        void init(const int &n) {
            l_next.resize(n);
            topology.assign(n, 0);
        }
    } qsSup;
    struct PathStateBits {
        vector<uint32_t> path_bits; 

        void init(const int &n, const int &L) {
            path_bits.assign(n * L, 0u);
        }

        uint32_t *pl(const int &v) {
            return &path_bits[(size_t) v * (size_t) label_hash_len];
        }

        [[nodiscard]] const uint32_t *pl(const int &v) const {
            return &path_bits[(size_t) v * (size_t) label_hash_len];
        }

        [[nodiscard]] bool is_subset(const int &v, const vector<uint32_t> &tps) const {
            const int vps = (size_t) v * (size_t) label_hash_len;
            for (int i = 0; i < label_hash_len; ++i) {
                if ((path_bits[vps + i] & tps[i]) != tps[i]) {
                    return false;
                }
            }
            return true;
        }
    } ps;
    struct PathStateExact {
        vector<int> path_label_seq; 
        vector<uint8_t> label_pos; 
        vector<int> touched_labels;
        vector<int> first_matched_pos; 
        uint8_t target_bits{}; 
        uint8_t current_bits{}; 
        vector<vector<uint8_t> > visited_vertex; 
        vector<int> touched_vertices; 

        inline void init(const std::vector<int> &idsL) {
            int n = (int) idsL.size();
            if (label_pos.size() != (size_t) num_L) {
                label_pos.assign(num_L, 255);
            } else {
                for (int l: touched_labels) label_pos[l] = 255;
            }
            touched_labels.clear();
            first_matched_pos.clear();
            first_matched_pos.resize(n);
            target_bits = 0;
            int pos = INT_MAX;
            uint8_t index = 0;
            for (auto &l: idsL) {
                label_pos[l] = index;
                touched_labels.push_back(l);
                first_matched_pos[index] = pos;
                target_bits |= (1u << index);
                index++;
            }
            current_bits = 0;
            path_label_seq.clear();
            if (visited_vertex.size() != (size_t) num_V) {
                visited_vertex.clear();
                visited_vertex.resize(num_V);
            } else {
                for (int uid: touched_vertices) visited_vertex[uid].clear();
            }
            touched_vertices.clear();
        }

        inline bool should_push_label(const int &l, const int &p) {
            uint8_t index = label_pos[l];
            if (index == 255) return false;
            return first_matched_pos[index] > p;
        }

        inline void push_label(const int &l, const bool &flag, const int &p) {
            if (flag) {
                uint8_t index = label_pos[l];
                first_matched_pos[index] = p;
                path_label_seq.push_back(l);
                current_bits |= (1u << index);
            } else {
                path_label_seq.push_back(-1);
            }
        }

        inline bool should_revisit(const int &uid, const int &l, const bool &flag) {
            uint8_t new_bits = current_bits;
            if (flag) {
                uint8_t index = label_pos[l];
                if (index == 255) return false;
                new_bits |= (1u << index);
            }
            for (auto &saved_bits: visited_vertex[uid]) {
                if ((saved_bits & new_bits) == new_bits) {
                    return false;
                }
            }
            if (visited_vertex[uid].empty()) touched_vertices.push_back(uid);
            visited_vertex[uid].push_back(new_bits);
            return true;
        }

        inline void pop_label(const int &uid) {
            if (visited_vertex[uid].empty()) touched_vertices.push_back(uid);
            visited_vertex[uid].push_back(current_bits);
            if (path_label_seq.empty()) return;
            const int l = path_label_seq.back();
            path_label_seq.pop_back();
            if (l >= 0) {
                uint8_t index = label_pos[l];
                first_matched_pos[index] = INT_MAX;
                current_bits &= ~(1u << index);
            }
        }
    } psExact;
    struct SequenceQueryState {
        int need_pos{}; 
        int num{}; 
        vector<vector<int> > matched_label; 
        vector<int> touched_vertices; 
        vector<int> alternative; 
        vector<int> touched_alternative; 
        void init(const std::vector<int> &idsSeq) {
            num = (int) idsSeq.size();
            need_pos = 0;
            if (matched_label.size() != (size_t) num_V) {
                matched_label.clear();
                matched_label.resize(num_V);
            } else {
                for (int uid: touched_vertices) matched_label[uid].clear();
            }
            touched_vertices.clear();
            if (alternative.size() != (size_t) num_V) {
                alternative.assign(num_V, -1);
            } else {
                for (int uid: touched_alternative) alternative[uid] = -1;
            }
            touched_alternative.clear();
        }

        inline void push_label_seq1(const int &uid, const int &l, const std::vector<int> &idsSeq) {
            if (matched_label[uid].empty()) touched_vertices.push_back(uid);
            matched_label[uid].push_back(l);
            if (l == idsSeq[need_pos]) {
                need_pos = (need_pos < num - 1) ? (need_pos + 1) : need_pos;
            }
        }

        inline void push_label_seq2(const int &uid, const int &l) {
            if (matched_label[uid].empty()) touched_vertices.push_back(uid);
            matched_label[uid].push_back(l);
            need_pos = (need_pos + 1) % num;
        }

        inline void pop_label_seq1(const int &uid, const std::vector<int> &idsSeq) {
            if (matched_label[uid].empty()) return;
            int current_label = matched_label[uid].back();
            int check_pos = (need_pos > 0) ? (need_pos - 1) : need_pos;
            if (current_label != idsSeq[check_pos] && current_label != idsSeq[num - 1]) {
                need_pos = check_pos;
            }
        }

        inline void pop_label_seq2() {
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
            return !matched_label[uid].empty() && matched_label[uid].back() == idsSeq[num - 1];
        }

        inline void set_alternative(const int &uid, const int &l) {
            if (alternative[uid] == -1) touched_alternative.push_back(uid);
            alternative[uid] = l;
        }

        inline int take_alternative(const int &uid) {
            int l = alternative[uid];
            alternative[uid] = -1;
            return l;
        }
    } sqs;
    enum class Mode { WholePath, AllLabel, Sequence1, Sequence2 };

    enum class LabelMatch { AND, OR, NOT, LCR };
    template<LabelMatch LM>
    struct LabelMatcher {
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
        static bool feasible_bits(int &l_next, const int &end_pos,
                                  uint32_t *v_ps,
                                  const vector<uint32_t> &bitsL) {
            for (int i = l_next; i < end_pos; ++i) {
                int tl = labels[i];
                const int p = lht.p[tl];
                const uint32_t mask = lht.mask[tl];
                bool label_in_query = ((bitsL[p] & mask) != 0);
                if constexpr (LM == LabelMatch::NOT) {
                    if (!label_in_query) {
                        return true;
                    }
                } else if constexpr (LM == LabelMatch::LCR) {
                    if (label_in_query) {
                        return true;
                    }
                } else {
                    bool label_in_path = ((v_ps[p] & mask) != 0);
                    if (label_in_query && !label_in_path) {
                        l_next = i + 1;
                        v_ps[p] |= mask;
                        return true;
                    }
                }
            }
            if constexpr (LM == LabelMatch::AND || LM == LabelMatch::OR) {
                l_next = end_pos;
            }
            return false;
        }

        static bool feasible_exact(int &l_next, const int &end_pos,
                                   const std::vector<int> &idsL,
                                   int &tl, const int &pos) {
            for (int i = l_next; i < end_pos; ++i) {
                tl = labels[i];
                bool label_in_query = contains_label(idsL, tl);
                if constexpr (LM == LabelMatch::NOT) {
                    if (!label_in_query) {
                        return true;
                    }
                } else if constexpr (LM == LabelMatch::LCR) {
                    if (label_in_query) {
                        return true;
                    }
                } else {
                    if (label_in_query) {
                        bool should_push = psExact.should_push_label(tl, pos);
                        if (should_push) {
                            l_next = i + 1;
                            return true;
                        }
                    }
                }
            }
            if constexpr (LM == LabelMatch::AND || LM == LabelMatch::OR) {
                l_next = end_pos;
            }
            return false;
        }
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
    struct QueryForwardWords {
        const VertexIndexV *index;

        uint32_t operator[](const int offset) const {
            if (index == nullptr) return 0;
            const int slot = offset / label_hash_len;
            return forward_layer_word(*index, slot / forward_default_k,
                                      slot % forward_default_k, offset % label_hash_len);
        }
    };

    struct QueryIndexView {
        bool root;
        int t_in, t_out, group;
        uint32_t leaf_hash;
        const vector<uint32_t> &vertices_all, &labels_all;
        QueryHorizontalWords vertices_group, labels_group;
        QueryForwardWords labels_forward;
    };

    inline QueryIndexView query_index_view(const int uid) {
        static const vector<uint32_t> empty;
        const VertexIndex &base = indices_base[uid];
        const VertexIndexH *h = enable_group_pruning() ? &indices_horizontal[uid] : nullptr;
        const VertexIndexV *v = enable_forward_pruning() ? &indices_vertical[uid] : nullptr;
        const int group = base.leaf ? -1 : (h != nullptr ? h->group : (v != nullptr ? v->group : 1));
        return {
            base.root, base.t_in, base.t_out, group, base.leaf ? base.leaf_hash() : 0,
            base.leaf ? empty : base.vertices_all(), base.labels_all,
            {h, true}, {h, false}, {v}
        };
    }
    template<Mode M, LabelMatch LM, bool UseExact = false>
    class ReachabilitySearcher {
    public:
        bool search(int uid, int vid,
                    const std::vector<uint32_t> &bitsL,
                    const std::vector<int> &idsL,
                    const std::vector<int> &idsSeq) {
            return search_impl(uid, vid, bitsL, idsL, idsSeq);
        }

    private:
        using matcher = LabelMatcher<LM>;
        [[nodiscard]] bool vertex_prune(const QueryIndexView &U, const QueryIndexView &V) const {
            if (!enable_base_pruning()) return false;
            if (V.group < 0) {
                uint32_t vh = V.leaf_hash;
                int blk = vertex_hash_block(vh);
                int bit = vertex_hash_bit(vh);
                bool flag = (U.vertices_all[blk] & (1u << bit)) == 0;
                return flag;
            }
            for (int i = 0; i < vertex_hash_len; ++i) {
                if ((U.vertices_all[i] & V.vertices_all[i]) != V.vertices_all[i]) {
                    return true;
                }
            }
            return false;
        }
        [[nodiscard]] bool label_prune_not_sequence(const int &uid, const std::vector<uint32_t> &bitsL) const {
            if (!enable_base_pruning()) return false;

            const QueryIndexView U = query_index_view(uid);
            for (int i = 0; i < label_hash_len; ++i) {
                uint32_t path_bits = 0u;
                if constexpr (M != Mode::AllLabel) path_bits = ps.pl(uid)[i];
                bool missing = matcher::prune_by_bits(U.labels_all[i], path_bits, bitsL[i]);
                if constexpr (LM == LabelMatch::AND) {
                    if (missing) return true;
                } else {
                    if (!missing) return false;
                }
            }

            return LM != LabelMatch::AND;
        }
        [[nodiscard]] bool label_prune_sequences(const QueryIndexView &U,
                                                 const int &needed_pos,
                                                 const std::vector<int> &idsSeq) const {
            if (!enable_base_pruning()) return false;

            int n = (int) idsSeq.size();
            for (int i = needed_pos; i < n; ++i) {
                int l = idsSeq[i];
                const int p = lht.p[l];
                if ((U.labels_all[p] & lht.mask[l]) == 0) {
                    return true;
                }
            }
            return false;
        }
        [[nodiscard]] bool group_vertex_prune(const QueryIndexView &U,
                                              const QueryIndexView &V,
                                              const int &gid) const {
            if (!enable_group_pruning()) return false;
            if (V.group < 0) {
                uint32_t vh = V.leaf_hash;
                int blk = vertex_hash_block(vh);
                int bit = vertex_hash_bit(vh);
                uint32_t temp = U.vertices_group[gid * vertex_hash_len + blk];
                bool flag = (temp & (1u << bit)) == 0;
                return flag;
            }
            for (int i = 0; i < vertex_hash_len; ++i) {
                uint32_t temp = U.vertices_group[gid * vertex_hash_len + i];
                if ((temp & V.vertices_all[i]) != V.vertices_all[i]) {
                    return true;
                }
            }
            return false;
        }
        [[nodiscard]] bool group_label_prune(const int &uid,
                                             const std::vector<uint32_t> &bitsL,
                                             const int &gid) const {
            if (!enable_group_pruning()) return false;

            const QueryIndexView U = query_index_view(uid);
            int index = gid * label_hash_len;
            for (int i = 0; i < label_hash_len; ++i) {
                uint32_t vertex_bits = U.labels_group[index + i];
                bool missing = matcher::prune_by_bits(vertex_bits, ps.pl(uid)[i], bitsL[i]);
                if constexpr (LM == LabelMatch::AND) {
                    if (missing) return true;
                } else {
                    if (!missing) return false;
                }
            }

            return LM != LabelMatch::AND;
        }
        [[nodiscard]] bool forward_label_prune_all_labels(const int &gid,
                                                          const QueryIndexView &U,
                                                          const std::vector<uint32_t> &bitsL) const {
            if (!enable_forward_pruning()) return false;

            int index = gid * (forward_default_k * label_hash_len);
            for (int i = 0; i < label_hash_len; ++i) {
                uint32_t vertex_bits = U.labels_forward[index + i];
                if (!matcher::prune_by_bits(vertex_bits, 0, bitsL[i])) {
                    return false;
                }
            }

            return true;
        }
        [[nodiscard]] bool forward_label_prune_sequence1(const int &gid, const QueryIndexView &U,
                                                         const std::vector<int> &idsSeq) const {
            if (!enable_forward_pruning()) return false;
            int index = gid * (forward_default_k * label_hash_len);
            int needed_pos = sqs.need_pos;
            const int needed_label = idsSeq[needed_pos];
            int p = lht.p[needed_label];
            if ((U.labels_forward[index + p] & lht.mask[needed_label]) != 0) {
                return false;
            }
            uint32_t mask = !!needed_pos;
            const int current_label = idsSeq[needed_pos - mask];
            p = lht.p[current_label];
            if ((U.labels_forward[index + p] & lht.mask[current_label]) != 0) {
                return false;
            }

            return true;
        }
        [[nodiscard]] bool forward_label_prune_sequence2(const int &gid, const QueryIndexView &U,
                                                         const std::vector<int> &idsSeq) const {
            if (!enable_forward_pruning()) return false;

            int needed_pos = sqs.need_pos;
            const int steps = std::min(forward_default_k, sqs.num - needed_pos);

            if (steps <= 0) {
                return false; 
            }
            for (int i = 0; i < steps; ++i) {
                int temp = gid * (forward_default_k * label_hash_len) + i * label_hash_len;
                int tl = idsSeq[needed_pos + i];
                const int p = lht.p[tl];
                if ((U.labels_forward[temp + p] & lht.mask[tl]) == 0) {
                    return true;
                }
            }
            return false;
        }
        [[nodiscard]] bool label_accept_check(const int &uid, const std::vector<uint32_t> &bitsL) const {
            if constexpr (UseExact) {
                return matcher::accept_exact(psExact.current_bits, psExact.target_bits);
            } else {
                if constexpr (LM == LabelMatch::OR) {
                    for (int i = 0; i < bitsL.size(); ++i)
                        if ((ps.pl(uid)[i] & bitsL[i]) != 0) return true;
                    return false;
                }
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
            int check_pos = (sqs.need_pos > 0) ? (sqs.need_pos - 1) : sqs.need_pos;
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
        bool search_impl(int uid, int vid,
                         const std::vector<uint32_t> &bitsL,
                         const std::vector<int> &idsL,
                         const std::vector<int> &idsSeq) {
            const QueryIndexView V = query_index_view(vid);
            if (V.root && uid != vid) return false;
            ensure_search_stack_capacity(s_pos + 1);
            s[s_pos] = uid;
            bool isAdd = true;

            if constexpr (M == Mode::WholePath) {
                std::memset(ps.pl(uid), 0u, label_hash_len * sizeof(uint32_t));
                if constexpr (UseExact) psExact.init(idsL);
            }
            if constexpr (M == Mode::Sequence1 || M == Mode::Sequence2) {
                sqs.init(idsSeq);
            }

            while (s_pos != -1) {
                int top_uid = s[s_pos];
                QueryIndexView top_u = query_index_view(top_uid);
                if constexpr (M == Mode::Sequence1) {
                    sqs.pop_label_seq1(top_uid, idsSeq);
                }

                if (isAdd) {
                    isAdd = false;
                    qs.label_edge[top_uid] = -1;
                    if constexpr (M != Mode::AllLabel) {
                        qsSup.topology[top_uid] = true;
                    }
                    if (top_uid == vid) {
                        if constexpr (M == Mode::AllLabel) {
                            return true;
                        } else if constexpr (M == Mode::WholePath) {
                            if (label_accept_check(top_uid, bitsL)) {
                                return true;
                            }
                        } else {
                            if (sqs.accept(vid, idsSeq)) {
                                return true;
                            }
                        }
                        qs.vis[top_uid] = vis_cur + 1;
                        --s_pos;
                        if constexpr (M == Mode::WholePath && UseExact) {
                            psExact.pop_label(top_uid);
                        }
                        if constexpr (M == Mode::Sequence2) {
                            sqs.pop_label_seq2();
                        }
                        continue;
                    }
                    if (top_u.group < 0) {
                        if constexpr (M != Mode::AllLabel) {
                            qsSup.topology[top_uid] = false;
                        }
                        if constexpr (M == Mode::WholePath && UseExact) {
                            psExact.pop_label(top_uid);
                        }
                        if constexpr (M == Mode::Sequence2) {
                            sqs.pop_label_seq2();
                        }
                        qs.vis[top_uid] = vis_cur + 1;
                        --s_pos;
                        continue;
                    }

                    if (enable_base_pruning() && top_u.t_out < V.t_out) {
                        if constexpr (M != Mode::AllLabel) {
                            qsSup.topology[top_uid] = false;
                        }
                        if constexpr (M == Mode::WholePath && UseExact) {
                            psExact.pop_label(top_uid);
                        }
                        if constexpr (M == Mode::Sequence2) {
                            sqs.pop_label_seq2();
                        }

                        qs.vis[top_uid] = vis_cur + 1;
                        --s_pos;
                        continue;
                    }
                    if constexpr (M == Mode::WholePath) {
                        if (enable_base_pruning() && top_u.t_in <= V.t_in &&
                            V.t_out <= top_u.t_out && label_accept_check(top_uid, bitsL))
                            return true;
                    }
                    if constexpr (M == Mode::Sequence1 || M == Mode::Sequence2) {
                        if (label_prune_sequences(top_u, sqs.need_pos, idsSeq)) {
                            qs.vis[top_uid] = vis_cur + 1;
                            if constexpr (M == Mode::Sequence2) {
                                sqs.pop_label_seq2();
                            }
                            --s_pos;
                            continue;
                        }
                    } else {
                        if (label_prune_not_sequence(top_uid, bitsL)) {
                            if constexpr (M == Mode::WholePath && UseExact) {
                                psExact.pop_label(top_uid);
                            }
                            qs.vis[top_uid] = vis_cur + 1;
                            --s_pos;
                            continue;
                        }
                    }
                    if (vertex_prune(top_u, V)) {
                        if constexpr (M != Mode::AllLabel) {
                            qsSup.topology[top_uid] = false;
                        }
                        if constexpr (M == Mode::WholePath && UseExact) {
                            psExact.pop_label(top_uid);
                        }
                        if constexpr (M == Mode::Sequence2) {
                            sqs.pop_label_seq2();
                        }
                        qs.vis[top_uid] = vis_cur + 1;
                        --s_pos;
                        continue;
                    }

                    qs.vis[top_uid] = vis_cur;
                    if constexpr (M == Mode::WholePath) {
                        if (top_u.group > 1) {
                            qs.gid[top_uid] = -1;
                            qs.next[top_uid] = source_offset[top_uid + 1];
                        } else {
                            qs.gid[top_uid] = 1;
                            qs.next[top_uid] = source_offset[top_uid];
                        }
                    } else {
                        if (top_u.group > 1) {
                            qs.gid[top_uid] = -1;
                            qs.next[top_uid] = source_offset[top_uid + 1];
                        } else {
                            bool flag = false;
                            if constexpr (M == Mode::AllLabel) {
                                flag = forward_label_prune_all_labels(0, top_u, bitsL);
                            } else if constexpr (M == Mode::Sequence1) {
                                flag = forward_label_prune_sequence1(0, top_u, idsSeq);
                            } else {
                                flag = forward_label_prune_sequence2(0, top_u, idsSeq);
                            }
                            if (flag) {
                                qs.vis[top_uid] = vis_cur + 1;
                                --s_pos;
                                continue;
                            }
                            qs.gid[top_uid] = 1;
                            qs.next[top_uid] = source_offset[top_uid];
                        }
                    }
                }
                bool finished = true;
                while (qs.gid[top_uid] < top_u.group) {
                    if (qs.next[top_uid] >= source_offset[top_uid + 1]) {
                        qs.gid[top_uid]++;
                        if (qs.gid[top_uid] == top_u.group) break;
                        if constexpr (M == Mode::AllLabel) {
                            if (forward_label_prune_all_labels(qs.gid[top_uid], top_u, bitsL)) continue;
                        } else if constexpr (M == Mode::Sequence1) {
                            if (forward_label_prune_sequence1(qs.gid[top_uid], top_u, idsSeq)) continue;
                        } else if constexpr (M == Mode::Sequence2) {
                            if (forward_label_prune_sequence2(qs.gid[top_uid], top_u, idsSeq)) continue;
                        } else {
                            if (group_label_prune(top_uid, bitsL, qs.gid[top_uid])) continue;
                        }
                        if (group_vertex_prune(top_u, V, qs.gid[top_uid])) {
                            continue;
                        }
                        qs.next[top_uid] = source_offset[top_uid] + qs.gid[top_uid];
                        finished = false;
                        break;
                    } else {
                        finished = false;
                        break;
                    }
                }
                while (qs.next[top_uid] < source_offset[top_uid + 1]) {
                    const int edge_idx = qs.next[top_uid];
                    ++query_edge_visits;
                    if constexpr (M != Mode::AllLabel) {
                        if (qs.label_edge[top_uid] != edge_idx) {
                            qs.label_edge[top_uid] = edge_idx;
                            qsSup.l_next[top_uid] = target_offset[edge_idx];
                        }
                    }
                    int tid = targets[edge_idx];
                    if constexpr (M == Mode::AllLabel) {
                        if (qs.vis[tid] < vis_cur) {
                            bool flag;
                            if constexpr (UseExact) {
                                int temp;
                                flag = matcher::feasible_exact(target_offset[edge_idx],
                                                               target_offset[edge_idx + 1],
                                                               idsL, temp, 0);
                            } else {
                                flag = matcher::feasible_bits(target_offset[edge_idx],
                                                              target_offset[edge_idx + 1], {}, bitsL);
                            }
                            if (flag) {
                                isAdd = true;
                                ensure_search_stack_capacity(s_pos + 2);
                                s[++s_pos] = tid;
                                finished = false;
                                break;
                            }
                        }
                        qs.next[top_uid] += top_u.group;
                    } else {
                        if (qs.vis[tid] < vis_cur) {
                            if constexpr (M == Mode::WholePath) {
                                qsSup.l_next[top_uid] = target_offset[edge_idx];
                                if constexpr (UseExact) {
                                    int tl = 0;
                                    memcpy(ps.pl(tid), ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                                    bool flag = matcher::feasible_exact(qsSup.l_next[top_uid],
                                                                        target_offset[edge_idx + 1],
                                                                        idsL, tl, s_pos);
                                    psExact.push_label(tl, flag, s_pos);
                                    if (flag) {
                                        const int p = lht.p[tl];
                                        ps.pl(tid)[p] |= lht.mask[tl];
                                    }
                                } else {
                                    memcpy(ps.pl(tid), ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                                    bool flag = matcher::feasible_bits(qsSup.l_next[top_uid],
                                                                       target_offset[edge_idx + 1],
                                                                       ps.pl(tid), bitsL);
                                }
                                isAdd = true;
                                ensure_search_stack_capacity(s_pos + 2);
                                s[++s_pos] = tid;
                                finished = false;
                                break;
                            } else if constexpr (M == Mode::Sequence1) {
                                int l1 = -1, l2 = -1;
                                const int r = feasible_sequence1(top_uid, idsSeq, l1, l2);
                                if (r == 3) {
                                    sqs.push_label_seq1(tid, l1, idsSeq);
                                    sqs.set_alternative(tid, l2);
                                } else if (r == 2) {
                                    sqs.push_label_seq1(tid, l2, idsSeq);
                                } else if (r == 1) {
                                    sqs.push_label_seq1(tid, l1, idsSeq);
                                }
                                if (r != 0) {
                                    isAdd = true;
                                    ensure_search_stack_capacity(s_pos + 2);
                                    s[++s_pos] = tid;
                                    finished = false;
                                    break;
                                }
                            } else {
                                if (feasible_sequence2(top_uid, idsSeq)) {
                                    isAdd = true;
                                    ensure_search_stack_capacity(s_pos + 2);
                                    s[++s_pos] = tid;
                                    sqs.push_label_seq2(tid, idsSeq[sqs.need_pos]);
                                    finished = false;
                                    break;
                                }
                            }
                        } else if (qs.vis[tid] > vis_cur && qsSup.topology[tid]) {
                            if constexpr (M == Mode::WholePath) {
                                if constexpr (UseExact) {
                                    int tl = 0;
                                    bool flag = matcher::feasible_exact(qsSup.l_next[top_uid],
                                                                        target_offset[edge_idx + 1],
                                                                        idsL, tl, s_pos);
                                    if (psExact.should_revisit(tid, tl, flag)) {
                                        isAdd = true;
                                        ensure_search_stack_capacity(s_pos + 2);
                                        s[++s_pos] = tid;
                                        finished = false;
                                        memcpy(ps.pl(tid), ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                                        psExact.push_label(tl, flag, s_pos);
                                        if (flag) {
                                            const int p = lht.p[tl];
                                            ps.pl(tid)[p] |= lht.mask[tl];
                                        }
                                        break;
                                    }
                                } else {
                                    uint32_t *tps = qs.scratch_bits.data();
                                    memcpy(tps, ps.pl(top_uid), sizeof(uint32_t) * label_hash_len);
                                    bool flag = matcher::feasible_bits(qsSup.l_next[top_uid],
                                                                       target_offset[edge_idx + 1],
                                                                       tps, bitsL);
                                    bool subset = true;
                                    const uint32_t *old_bits = ps.pl(tid);
                                    for (int bi = 0; bi < label_hash_len; ++bi) {
                                        if ((old_bits[bi] & tps[bi]) != tps[bi]) {
                                            subset = false;
                                            break;
                                        }
                                    }
                                    if (!subset) {
                                        memcpy(ps.pl(tid), tps, sizeof(uint32_t) * label_hash_len);
                                        isAdd = true;
                                        ensure_search_stack_capacity(s_pos + 2);
                                        s[++s_pos] = tid;
                                        finished = false;
                                        break;
                                    }
                                }
                            } else if constexpr (M == Mode::Sequence1) {
                                if (sqs.alternative[tid] != -1) {
                                    int l = sqs.take_alternative(tid);
                                    if (sqs.should_revisit(tid, l)) {
                                        sqs.push_label_seq1(tid, l, idsSeq);
                                        isAdd = true;
                                        ensure_search_stack_capacity(s_pos + 2);
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
                                            if (sqs.should_revisit(tid, l2)) sqs.set_alternative(tid, l2);
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
                                        ensure_search_stack_capacity(s_pos + 2);
                                        s[++s_pos] = tid;
                                        finished = false;
                                        break;
                                    }
                                }
                            } else {
                                if (feasible_sequence2(top_uid, idsSeq) && sqs.
                                    should_revisit(tid, idsSeq[sqs.need_pos])) {
                                    isAdd = true;
                                    ensure_search_stack_capacity(s_pos + 2);
                                    s[++s_pos] = tid;
                                    sqs.push_label_seq2(tid, idsSeq[sqs.need_pos]);
                                    finished = false;
                                    break;
                                }
                            }
                        }
                        qs.next[top_uid] += top_u.group;
                        if constexpr (M == Mode::WholePath) {
                            if (qs.next[top_uid] < source_offset[top_uid + 1])
                                qsSup.l_next[top_uid] = target_offset[qs.next[top_uid]];
                        }
                    }
                }

                if (finished) {
                    --s_pos; 
                    qs.vis[top_uid] = vis_cur + 1;
                    isAdd = false;
                    if constexpr (M == Mode::WholePath && UseExact) {
                        psExact.pop_label(top_uid);
                    }
                    if constexpr (M == Mode::Sequence2) {
                        sqs.pop_label_seq2();
                    }
                }
            }
            return false;
        }
    };

    class ReachableQuery {
        template<int Ptn>
        static bool searcher(const int uid, const int vid,
                             const std::vector<uint32_t> &bitsL,
                             const std::vector<int> &idsL,
                             const std::vector<int> &idsSeq) {
            if constexpr (Ptn == 1) {
                static ReachabilitySearcher<Mode::WholePath, LabelMatch::AND> searcher;
                return searcher.search(uid, vid, bitsL, {}, {});
            } else if constexpr (Ptn == 2) {
                static ReachabilitySearcher<Mode::WholePath, LabelMatch::OR> searcher;
                return searcher.search(uid, vid, bitsL, {}, {});
            } else if constexpr (Ptn == 3) {
                static ReachabilitySearcher<Mode::AllLabel, LabelMatch::NOT> searcher;
                return searcher.search(uid, vid, bitsL, {}, {});
            } else if constexpr (Ptn == 4) {
                static ReachabilitySearcher<Mode::AllLabel, LabelMatch::LCR> searcher;
                return searcher.search(uid, vid, bitsL, {}, {});
            } else if constexpr (Ptn == 5) {
                if (idsL.size() > 8) {
                    static ReachabilitySearcher<Mode::WholePath, LabelMatch::AND> wide_searcher;
                    return wide_searcher.search(uid, vid, bitsL, {}, {});
                }
                static ReachabilitySearcher<Mode::WholePath, LabelMatch::AND, true> searcher;
                return searcher.search(uid, vid, bitsL, idsL, {});
            } else if constexpr (Ptn == 6) {
                if (idsL.size() > 8) {
                    static ReachabilitySearcher<Mode::WholePath, LabelMatch::OR> wide_searcher;
                    return wide_searcher.search(uid, vid, bitsL, {}, {});
                }
                static ReachabilitySearcher<Mode::WholePath, LabelMatch::OR, true> searcher;
                return searcher.search(uid, vid, bitsL, idsL, {});
            } else if constexpr (Ptn == 7) {
                static ReachabilitySearcher<Mode::AllLabel, LabelMatch::NOT, true> searcher;
                return searcher.search(uid, vid, bitsL, idsL, {});
            } else if constexpr (Ptn == 8) {
                static ReachabilitySearcher<Mode::AllLabel, LabelMatch::LCR, true> searcher;
                return searcher.search(uid, vid, bitsL, idsL, {});
            } else if constexpr (Ptn == 9) {
                static ReachabilitySearcher<Mode::Sequence1, LabelMatch::AND> searcher;
                return searcher.search(uid, vid, {}, {}, idsSeq);
            } else if constexpr (Ptn == 10) {
                static ReachabilitySearcher<Mode::Sequence2, LabelMatch::AND> searcher;
                return searcher.search(uid, vid, {}, {}, idsSeq);
            } else {
                throw std::invalid_argument("Invalid ptn value");
            }
        }

    public:
        static bool search_dispatch(const int ptn, const int uid, const int vid,
                                    const std::vector<uint32_t> &bitsL,
                                    const std::vector<int> &idsL,
                                    const std::vector<int> &idsSeq) {
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
                default: throw std::invalid_argument("Invalid ptn value: " + std::to_string(ptn));
            }
        }
    };
    bool read_graph(const string &filename) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        ifstream file;
        file.open(filename, ios::in);
        string str;
        if (!file.is_open()) {
            cerr << "Failed to open graph file!" << endl;
            return false;
        }

        cout << "********* start to read graph! *********" << endl;
        file >> str >> num_V >> str >> num_TE >> str >> num_LE >> str >> num_L;
        if (num_V <= 0) {
            cerr << "File format error!" << endl;
            return false;
        }
        cout << "### Graph Statistic ###" << endl;
        cout << "Vertices: " << num_V << " " << "Topological edges:" << num_TE
                << " Labeled edges:" << num_TE << " Labels: " << num_L << endl;

        indices_base.resize(num_V);
        indices_horizontal.assign(num_V, VertexIndexH{});
        indices_vertical.assign(num_V, VertexIndexV{});
        label_hash_len = std::max(1, (num_L + UINT_bits - 1) / UINT_bits);
        for (auto &vtx: indices_base) {
            vtx.t_in = 0;
            vtx.t_out = 0;
            vtx.root = true;
        }
        bst.init(num_V);

        graph_label_count.assign(num_L, 0);
        lht.init(num_L, label_hash_len);
        label_universe_bits.assign(label_hash_len, 0);
        for (int l = 0; l < num_L; ++l) {
            label_universe_bits[lht.p[l]] |= lht.mask[l];
        }

        source_offset.assign(num_V + 1, 0);
        targets.resize(num_TE);
        target_offset.assign(num_TE + 1, 0);
        labels.resize(num_LE);

        int u, v, v_pos = 0, l_pos = 0;
        while (file >> u >> v >> str) {
            if (u < 0 || u >= num_V || v < 0 || v >= num_V) {
                cerr << "Graph vertex id out of range: " << u << " -> " << v << endl;
                return false;
            }
            if (v_pos >= num_TE) {
                cerr << "Graph contains more edges than declared in header" << endl;
                return false;
            }
            targets[v_pos] = v;
            indices_base[v].root = false;
            for_each_label(str, [&](const int l) {
                if (l < 0 || l >= num_L || l_pos >= num_LE) {
                    throw std::runtime_error("Graph label id/count is inconsistent with header");
                }
                labels[l_pos] = l;
                graph_label_count[l]++;
                l_pos++;
                target_offset[v_pos + 1] = l_pos;
            });
            v_pos++;
            source_offset[u + 1] = v_pos;
        }
        file.close();
        if (v_pos != num_TE || l_pos != num_LE) {
            cerr << "Graph data count mismatch: read " << v_pos << " edges and " << l_pos
                    << " labels, expected " << num_TE << " edges and " << num_LE << " labels" << endl;
            return false;
        }
        for (int i = 1; i < num_V + 1; ++i) {
            if (source_offset[i] == 0) {
                source_offset[i] = source_offset[i - 1];
            }
        }
        for (int i = 1; i < num_TE + 1; ++i) {
            if (target_offset[i] == 0) target_offset[i] = target_offset[i - 1];
        }

        for (int i = 0; i < num_V; ++i) {
            if (source_offset[i] > source_offset[i + 1]) {
                cerr << "Graph edges must be sorted by source vertex; source_offset is not monotonic"
                        << endl;
                return false;
            }
        }


        cout << "### Label Distribution ###" << endl;
        for (int i = 0; i < num_L; i++) {
            cout << "Label" << i << ": " << graph_label_count[i] << "/" << num_LE << " = "
                    << graph_label_count[i] / static_cast<double>(num_TE) << endl;
        }
        const auto end_time = std::chrono::high_resolution_clock::now();
        const double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        cout << "The time to read graph " << filename << " is: " << total_time << " ms" << endl;
        cout << endl;
        return true;
    }
    void DFS_out(const int &start) {
        s[s_pos] = start;
        bool isAdd = true;
        SCC_pos = 0;
        SCC[SCC_pos] = start;
        while (s_pos != -1) {
            int top_vid = s[s_pos];
            VertexIndex &top_v = indices_base[top_vid];
            if (isAdd) {
                isAdd = false;
                bst.vis[top_vid] = vis_cur;
                bst.next[top_vid] = source_offset[top_vid];
                top_v.t_in = ++cur;
                bst.low[top_vid] = cur;
            }
            bool finished = true;
            const int max_neighbor = source_offset[top_vid + 1];
            while (bst.next[top_vid] < max_neighbor) {
                const int vid = targets[bst.next[top_vid]];
                if (bst.vis[vid] < vis_cur) {
                    bst.parent[vid] = top_vid;
                    ++s_pos;
                    s[s_pos] = vid;
                    ++SCC_pos;
                    SCC[SCC_pos] = vid;
                    isAdd = true;
                    finished = false;
                    break;
                }
                if (bst.vis[vid] == vis_cur) {
                    if (bst.parent[vid] == top_vid) {
                        bst.low[top_vid] = min(bst.low[top_vid], bst.low[vid]);
                    } else {
                        bst.low[top_vid] = min(bst.low[top_vid], indices_base[vid].t_in);
                    }
                }
                bst.next[top_vid]++;
            }
            if (finished) {
                if (const int n = max_neighbor - source_offset[top_vid]; n == 0) {
                    bst.hash_value[top_vid] = vertex_hash(top_vid);
                    top_v.make_leaf(bst.hash_value[top_vid]);
                } else {
                    top_v.make_internal();
                    auto &top_vertices_all = top_v.vertices_all();
                    if (top_vertices_all.size() != static_cast<size_t>(vertex_hash_len))
                        top_vertices_all.resize(vertex_hash_len);
                    if (top_v.labels_all.size() != static_cast<size_t>(label_hash_len))
                        top_v.labels_all.resize(label_hash_len);
                    std::fill(top_vertices_all.begin(), top_vertices_all.end(), 0);
                    std::fill(top_v.labels_all.begin(), top_v.labels_all.end(), 0);
                    const int group = prepared_horizontal_group_count(top_vid, n);
                    if (group <= 0 || top_vid < 0 || top_vid >= static_cast<int>(indices_base.size())) {
                        throw std::runtime_error("Invalid DFS index layout");
                    }

                    VertexIndexH *top_h = build_group_label_index ? &indices_horizontal[top_vid] : nullptr;
                    VertexIndexV *top_vertical = build_forward_label_index ? &indices_vertical[top_vid] : nullptr;
                    if (top_h != nullptr) top_h->group = group;
                    if (build_forward_label_index) {
                        prepare_forward_layout(*top_vertical, group);
                        if (const auto forward_size = static_cast<size_t>(forward_total_words(*top_vertical));
                            top_vertical->labels_forward.size() != forward_size)
                            top_vertical->labels_forward.resize(forward_size);
                        std::fill(top_vertical->labels_forward.begin(), top_vertical->labels_forward.end(), 0);
                    }
                    if (top_h != nullptr && group > 1) {
                        const size_t vertices_group_size =
                                static_cast<size_t>(group) * static_cast<size_t>(vertex_hash_len);
                        if (top_h->vertices_group.size() != vertices_group_size)
                            top_h->vertices_group.resize(vertices_group_size);
                        std::fill(top_h->vertices_group.begin(), top_h->vertices_group.end(), 0);
                        const size_t labels_group_size =
                                static_cast<size_t>(group) * static_cast<size_t>(label_hash_len);
                        if (top_h->labels_group.size() != labels_group_size)
                            top_h->labels_group.resize(labels_group_size);
                        std::fill(top_h->labels_group.begin(), top_h->labels_group.end(), 0);
                    }
                    int order = 0;
                    for (int i = source_offset[top_vid]; i < max_neighbor; i++) {
                        const int vid = targets[i], index = order % group;
                        int hv;
                        order++;
                        VertexIndex &v = indices_base[vid];
                        if (bst.hash_value[vid] == 0) {
                            hv = vertex_hash(vid);
                            bst.hash_value[vid] = hv;
                        } else {
                            hv = static_cast<int>(bst.hash_value[vid]);
                        }
                        const int forward_base = top_vertical != nullptr
                                                     ? forward_group_offset(*top_vertical, index)
                                                     : 0;
                        const int group_base = index * label_hash_len;
                        for (int li = target_offset[i]; li < target_offset[i + 1]; ++li) {
                            const int label = labels[li];
                            const int b = lht.p[label];
                            const uint32_t edge_bits = lht.mask[label];
                            top_v.labels_all[b] |= edge_bits;
                            if (top_vertical != nullptr && forward_group_depth(*top_vertical, index) > 0)
                                top_vertical->labels_forward[forward_base + b] |= edge_bits;
                            if (top_h != nullptr && group > 1) top_h->labels_group[group_base + b] |= edge_bits;
                        }
                        const int p = vertex_hash_block(static_cast<uint32_t>(hv));
                        const int h = vertex_hash_bit(static_cast<uint32_t>(hv));
                        top_vertices_all[p] |= (1u << h);
                        if (top_h != nullptr && group > 1)
                            top_h->vertices_group[index * vertex_hash_len + p] |= (1u << h);
                        if (v.leaf || bst.vis[vid] != vis_cur + 1) continue;
                        if (source_offset[vid] != source_offset[vid + 1]) {
                            or_words(top_vertices_all.data(), v.vertices_all().data(), vertex_hash_len);
                            or_words(top_v.labels_all.data(), v.labels_all.data(), label_hash_len);
                        }
                    }
                }
                top_v.t_out = ++cur;

                if (top_v.t_in == bst.low[top_vid]) {
                    int component_begin = SCC_pos;
                    while (component_begin >= 0 && SCC[component_begin] != top_vid)
                        --component_begin;
                    if (component_begin < 0)
                        throw std::runtime_error("SCC root missing from stack");
                    if (!top_v.leaf) {
                        for (int si = component_begin + 1; si <= SCC_pos; ++si) {
                            const int member_id = SCC[si];
                            const VertexIndex &member = indices_base[member_id];
                            if (member.leaf) continue;
                            or_words(top_v.vertices_all().data(), member.vertices_all().data(), vertex_hash_len);
                            or_words(top_v.labels_all.data(), member.labels_all.data(), label_hash_len);
                        }
                    }
                    bool found_scc_root = false;
                    while (SCC_pos >= 0) {
                        const int tid = SCC[SCC_pos];
                        VertexIndex &t = indices_base[tid];
                        SCC_pos--;
                        bst.vis[tid] = vis_cur + 1;
                        if (tid == top_vid) {
                            found_scc_root = true;
                            break;
                        }
                        t.t_in = top_v.t_in;
                        t.t_out = top_v.t_out;
                        t.labels_all = top_v.labels_all;
                        t.make_internal();
                        t.vertices_all() = top_v.vertices_all();
                    }
                    if (!found_scc_root) {
                        throw std::runtime_error("SCC stack corrupted: top vertex not found");
                    }
                }
                s_pos--;
            }
        }
    }
    struct IndexBuildStats {
        double total_time = 0.0;
        long long index_size = 0;
        long long index_space_base = 0;
        long long index_space_horizontal = 0;
        long long index_space_vertical = 0;
        long long horizontal_uncompressed = 0;
    };
    void reset_index_state() {
        cur = 0;
        vis_cur += 2;
        s_pos = 0;
        SCC_pos = 0;
        reset_vertex_hash_state();
        bst.init(num_V);
        horizontal_group_count.clear();
        for (auto &v: indices_base) {
            v.t_in = 0;
            v.t_out = 0;
            v.make_internal();
            vector<uint32_t>().swap(v.vertices_all());
            vector<uint32_t>().swap(v.labels_all);
        }
        if (build_group_label_index) indices_horizontal.assign(num_V, VertexIndexH{});
        else vector<VertexIndexH>().swap(indices_horizontal);
        if (build_forward_label_index) indices_vertical.assign(num_V, VertexIndexV{});
        else vector<VertexIndexV>().swap(indices_vertical);
    }
    void prepare_baseline_traversal_state() {
        cur = 0;
        vis_cur += 2;
        s_pos = 0;
        SCC_pos = 0;
        reset_vertex_hash_state();
        bst.init(num_V);
        horizontal_group_count.clear();
        for (int uid = 0; uid < num_V; ++uid) {
            VertexIndex &v = indices_base[uid];
            v.t_in = 0;
            v.t_out = 0;
            if (source_offset[uid] == source_offset[uid + 1]) v.make_leaf(0);
            else {
                v.make_internal();
                vector<uint32_t>().swap(v.vertices_all());
            }
            vector<uint32_t>().swap(v.labels_all);
        }
        vector<VertexIndexH>().swap(indices_horizontal);
        vector<VertexIndexV>().swap(indices_vertical);
    }
    IndexBuildStats calculate_index_space() {
        IndexBuildStats stats;
        stats.horizontal_uncompressed = horizontal_dense_bytes;
        for (const auto &v: indices_base) {
            stats.index_space_base += sizeof(v.t_in) + sizeof(v.t_out);
            stats.index_space_base += static_cast<long long>(sizeof(uint32_t)) * v.labels_all.size();
            stats.index_space_base += v.leaf
                                          ? static_cast<long long>(sizeof(uint32_t))
                                          : static_cast<long long>(sizeof(uint32_t) * v.vertices_all().size());
        }
        for (const auto &v: indices_horizontal) {
            stats.index_space_horizontal += static_cast<long long>(sizeof(uint32_t) * v.vertices_group.size());
            stats.index_space_horizontal += static_cast<long long>(sizeof(uint32_t) * v.labels_group.size());
            stats.index_space_horizontal += sizeof(v.group);
            stats.index_space_horizontal += sizeof(int) * (v.vertices_descriptor.size() + v.labels_descriptor.size());
        }
        for (const auto &v: indices_vertical) {
            stats.index_space_vertical += static_cast<long long>(sizeof(uint32_t) * v.labels_forward.size());
            stats.index_space_vertical += static_cast<long long>(sizeof(uint8_t) * v.forward_depth.size());
            stats.index_space_vertical += static_cast<long long>(sizeof(int) * v.forward_offset.size());
            stats.index_space_vertical += static_cast<long long>(sizeof(int) * v.forward_bitmap_offset.size());
        }
        stats.index_size = stats.index_space_base + stats.index_space_horizontal + stats.index_space_vertical;
        return stats;
    }
    void print_index_build_stats(const string &profile_name, const IndexBuildStats &stats) {
        constexpr int one_hour = 1 * 3600 * 1000, one_minute = 1 * 60 * 1000;
        cout << "### Index of " << profile_name << " ###" << endl;
        cout << "indexTime: " << stats.total_time << " ms";
        if (stats.total_time >= one_hour)
            cout << " = " << stats.total_time / one_hour << " h" << endl;
        else if (stats.total_time >= one_minute)
            cout << " = " << stats.total_time / one_minute << " min" << endl;
        else
            cout << endl;
        cout << "groupIndex: " << (build_group_label_index ? "on" : "off") << endl;
        cout << "forwardIndex: " << (build_forward_label_index ? "on" : "off") << endl;
        cout << "indexPayloadBytes: base=" << stats.index_space_base
                << " horizontal=" << stats.index_space_horizontal
                << " vertical=" << stats.index_space_vertical << endl;
        cout << "horizontalBitmapBytes: dense=" << stats.horizontal_uncompressed
                << " packed=" << stats.index_space_horizontal << endl;
        cout << "indexSpace: " << std::fixed << std::setprecision(3)
                << static_cast<double>(stats.index_size) / (1024.0 * 1024.0)
                << " MB" << std::defaultfloat << std::setprecision(6) << endl;
    }
    IndexBuildStats index_construction(const string &profile_name) {
        reset_index_state();
        auto start_time = std::chrono::high_resolution_clock::now();
        prepare_horizontal_group_features(graph_label_count);
        for (int u = 0; u < num_V; ++u) {
            if (indices_base[u].root) {
                s_pos = 0;
                DFS_out(u);
            }
        }
        for (int u = 0; u < num_V; ++u) {
            if (bst.vis[u] != vis_cur + 1) {
                s_pos = 0;
                DFS_out(u);
            }
        }
        for (int uid = 0; uid < num_V; ++uid) {
            VertexIndex &u_base = indices_base[uid];
            if (u_base.leaf) continue;
            VertexIndexH *u_h = build_group_label_index ? &indices_horizontal[uid] : nullptr;
            VertexIndexV *u_v = build_forward_label_index ? &indices_vertical[uid] : nullptr;
            const int group = u_h != nullptr ? u_h->group : u_v->group;
            if (group > 1) {
                int order = 0;
                for (int i = source_offset[uid]; i < source_offset[uid + 1]; i++) {
                    const int vid = targets[i];
                    const int index = order % group;
                    order++;
                    VertexIndex &v_base = indices_base[vid];
                    if (source_offset[vid] != source_offset[vid + 1]) {
                        if (v_base.leaf || v_base.vertices_all().size() < static_cast<size_t>(vertex_hash_len) ||
                            v_base.labels_all.size() < static_cast<size_t>(label_hash_len)) {
                            throw std::runtime_error("Missing completed vertex summary");
                        }
                        if (u_h != nullptr) {
                            or_words(&u_h->vertices_group[index * vertex_hash_len],
                                     v_base.vertices_all().data(), vertex_hash_len);
                            or_words(&u_h->labels_group[index * label_hash_len],
                                     v_base.labels_all.data(), label_hash_len);
                        }
                        if (u_v != nullptr) {
                            const VertexIndexV &v_v = indices_vertical[vid];
                            const int u_depth = forward_group_depth(*u_v, index);
                            const int u_offset = forward_group_offset(*u_v, index);
                            for (int pi = 1; pi < u_depth; ++pi) {
                                const int index_u = u_offset + pi * label_hash_len;
                                for (int gi = 0; gi < v_v.group; ++gi) {
                                    if (forward_group_depth(v_v, gi) <= pi - 1) continue;
                                    const int index_v = forward_group_offset(v_v, gi) + (pi - 1) * label_hash_len;
                                    or_words(&u_v->labels_forward[index_u], &v_v.labels_forward[index_v],
                                             label_hash_len);
                                }
                            }
                        }
                    }
                }
            } else if (group == 1 && u_v != nullptr) {
                for (int i = source_offset[uid]; i < source_offset[uid + 1]; i++) {
                    int vid = targets[i];
                    if (source_offset[vid] != source_offset[vid + 1]) {
                        const VertexIndexV &v_v = indices_vertical[vid];
                        const int u_depth = forward_group_depth(*u_v, 0);
                        const int u_offset = forward_group_offset(*u_v, 0);
                        for (int pi = 1; pi < u_depth; ++pi) {
                            const int index_u = u_offset + pi * label_hash_len;
                            for (int gi = 0; gi < v_v.group; ++gi) {
                                if (forward_group_depth(v_v, gi) <= pi - 1) continue;
                                const int index_v = forward_group_offset(v_v, gi) + (pi - 1) * label_hash_len;
                                or_words(&u_v->labels_forward[index_u], &v_v.labels_forward[index_v], label_hash_len);
                            }
                        }
                    }
                }
            }
        }
        horizontal_dense_bytes = 0;
        for (auto &h: indices_horizontal) {
            horizontal_dense_bytes += sizeof(h.group) + sizeof(uint32_t) *
                    (h.vertices_group.size() + h.labels_group.size());
            compact_horizontal_words(h.vertices_group, h.vertices_descriptor, vertex_hash_len, true);
            compact_horizontal_words(h.labels_group, h.labels_descriptor, label_hash_len, false);
        }
        finalize_forward_density_flags();
        const auto end_time = std::chrono::high_resolution_clock::now();

        IndexBuildStats stats = calculate_index_space();
        stats.total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        print_index_build_stats(profile_name, stats);
        return stats;
    }
    bool read_queries(const string &filename) {
        auto start_time = std::chrono::high_resolution_clock::now();
        queries.clear();
        ifstream file;
        file.open(filename, ios::in);
        if (!file.is_open()) {
            cout << "Failed to read query file!" << endl;
            return false;
        }
        Query q{};
        string str;
        while (file >> q.origin >> q.destination >> q.pattern >> str) {
            q.bits_label.assign(label_hash_len, 0);
            q.ids_label.clear();
            q.ids_sequence.clear();
            for_each_label(str, [&](const int l) {
                if (q.pattern == 9) {
                    if (q.ids_sequence.empty() || l != q.ids_sequence.back()) q.ids_sequence.push_back(l);
                } else if (q.pattern == 10) {
                    q.ids_sequence.push_back(l);
                } else {
                    if (!contains_label(q.ids_label, l)) q.ids_label.push_back(l);
                }
                const int p = lht.p[l];
                q.bits_label[p] |= lht.mask[l];
            });
            q.outcome = false;
            queries.push_back(q);
        }
        file.close();
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        auto found = filename.rfind('/');
        string name = found != std::string::npos
                          ? filename.substr(found + 1)
                          : filename;
        cout << "The time to read " << name << " is: " << total_time << " ms" << endl;
        return true;
    }
    struct QueryRunStats {
        ProfileRuntimeStats runtime;
        vector<bool> outcomes;
        vector<long long> edge_visits;
    };

    const char *query_profile_name(const QueryProfile profile) {
        switch (profile) {
            case QueryProfile::Baseline: return "baseline";
            case QueryProfile::OnlyH: return "only-H";
            case QueryProfile::OnlyV: return "only-V";
            case QueryProfile::Full: return "TDR";
        }
        return "unknown";
    }
    QueryRunStats run_query_profile(const QueryProfile profile) {
        active_profile = profile;
        QueryRunStats stats;
        stats.outcomes.resize(queries.size(), false);
        stats.edge_visits.resize(queries.size(), 0);

        for (size_t qi = 0; qi < queries.size(); ++qi) {
            Query &q = queries[qi];
            int L = static_cast<int>(q.ids_label.size());
            if (q.pattern > 8) L = static_cast<int>(q.ids_sequence.size());
            query_edge_visits = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            bool outcome;
            {
                reset_query_marks_if_needed();
                s_pos = 0;
                vis_cur += 2;
                outcome = ReachableQuery::search_dispatch(q.pattern, q.origin, q.destination,
                                                          q.bits_label, q.ids_label, q.ids_sequence);
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            const double qtime = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            stats.outcomes[qi] = outcome;
            stats.edge_visits[qi] = query_edge_visits;
            stats.runtime.add(q.pattern, L, outcome, qtime);
            if (profile == QueryProfile::Full) q.outcome = outcome;
        }
        return stats;
    }
    void print_profile_runtime(const char *name, const QueryRunStats &stats) {
        cout << "### Runtime of " << name << " ###" << endl;
        for (auto &p: stats.runtime.total_count) {
            cout << "pattern: " << p.first << " total number: " << p.second << " total time: "
                    << stats.runtime.total_time.at(p.first) << " ms" << endl;
        }
    }

    void print_runtime_summary(const QueryRunStats &tdr_stats,
                               const QueryRunStats *baseline_stats = nullptr,
                               const QueryRunStats *only_h_stats = nullptr,
                               const QueryRunStats *only_v_stats = nullptr) {
        if (baseline_stats && only_h_stats && only_v_stats) {
            print_profile_runtime("Baseline", *baseline_stats);
            print_profile_runtime("Only-H", *only_h_stats);
            print_profile_runtime("Only-V", *only_v_stats);
            print_profile_runtime("TDR", tdr_stats);

            VisitRatioStats only_h_ratio, only_v_ratio, tdr_ratio;
            for (size_t i = 0; i < queries.size(); ++i) {
                only_h_ratio.add(baseline_stats->edge_visits[i], only_h_stats->edge_visits[i],
                                 baseline_stats->outcomes[i] == only_h_stats->outcomes[i]);
                only_v_ratio.add(baseline_stats->edge_visits[i], only_v_stats->edge_visits[i],
                                 baseline_stats->outcomes[i] == only_v_stats->outcomes[i]);
                tdr_ratio.add(baseline_stats->edge_visits[i], tdr_stats.edge_visits[i],
                              baseline_stats->outcomes[i] == tdr_stats.outcomes[i]);
            }
            cout << "### Visited Edge Ratio vs Baseline ###" << endl;
            cout << "only-H: " << only_h_ratio.profile_edges << "/" << only_h_ratio.baseline_edges
                    << "=" << only_h_ratio.ratio() << endl;
            cout << "only-V: " << only_v_ratio.profile_edges << "/" << only_v_ratio.baseline_edges
                    << "=" << only_v_ratio.ratio() << endl;
            cout << "TDR: " << tdr_ratio.profile_edges << "/" << tdr_ratio.baseline_edges
                    << "=" << tdr_ratio.ratio() << endl;
        } else {
            print_profile_runtime("TDR", tdr_stats);
        }
        cout << "********* Finish! *********" << endl;
    }

    double runtime_mean(const std::map<int, std::vector<std::pair<double, int> > > &table,
                        const int pattern, const int length) {
        const auto it = table.find(pattern);
        if (it == table.end() || length <= 0 || length > static_cast<int>(it->second.size())) return 0.0;
        const auto &entry = it->second[length - 1];
        return entry.second == 0 ? 0.0 : entry.first / static_cast<double>(entry.second);
    }

    int runtime_count(const std::map<int, std::vector<std::pair<double, int> > > &table,
                      const int pattern, const int length) {
        const auto it = table.find(pattern);
        if (it == table.end() || length <= 0 || length > static_cast<int>(it->second.size())) return 0;
        return it->second[length - 1].second;
    }

    int map_count_or_zero(const std::map<int, int> &values, const int key) {
        const auto it = values.find(key);
        return it == values.end() ? 0 : it->second;
    }

    double map_double_or_zero(const std::map<int, double> &values, const int key) {
        const auto it = values.find(key);
        return it == values.end() ? 0.0 : it->second;
    }

    std::vector<int> query_patterns() {
        std::vector<int> patterns;
        patterns.reserve(queries.size());
        for (const Query &q: queries) patterns.push_back(q.pattern);
        std::sort(patterns.begin(), patterns.end());
        patterns.erase(std::unique(patterns.begin(), patterns.end()), patterns.end());
        return patterns;
    }
    VisitRatioStats calculate_visit_ratio(const QueryRunStats &baseline,
                                          const QueryRunStats &profile) {
        VisitRatioStats ratio;
        const size_t n = std::min(baseline.outcomes.size(), profile.outcomes.size());
        for (size_t i = 0; i < n; ++i) {
            ratio.add(baseline.edge_visits[i], profile.edge_visits[i],
                      baseline.outcomes[i] == profile.outcomes[i]);
        }
        return ratio;
    }
    void write_static_file(const string &filename, const string &graph_name,
                           const IndexBuildStats &only_h_index_stats,
                           const IndexBuildStats &only_v_index_stats,
                           const IndexBuildStats &tdr_index_stats,
                           const QueryRunStats &tdr_stats,
                           const QueryRunStats *baseline_stats = nullptr,
                           const QueryRunStats *only_h_stats = nullptr,
                           const QueryRunStats *only_v_stats = nullptr) {
        const bool compare = baseline_stats && only_h_stats && only_v_stats;
        std::ofstream out(filename, std::ios::app);
        if (!out.is_open()) {
            std::cerr << "Failed to create static output file: " << filename << std::endl;
            return;
        }
        out.setf(std::ios::fixed, std::ios::floatfield);
        out << std::setprecision(6);
        const std::vector<int> patterns = query_patterns();
        if (compare) {
            out << std::left << std::setw(32) << graph_name
                << std::right << std::setw(16) << only_h_index_stats.total_time
                << std::setw(16) << static_cast<double>(only_h_index_stats.index_size) / (1024.0 * 1024.0)
                << std::setw(16) << only_v_index_stats.total_time
                << std::setw(16) << static_cast<double>(only_v_index_stats.index_size) / (1024.0 * 1024.0)
                << std::setw(16) << tdr_index_stats.total_time
                << std::setw(16) << static_cast<double>(tdr_index_stats.index_size) / (1024.0 * 1024.0);
        }else {
            out << std::left << std::setw(32) << graph_name
                << std::right << std::setw(16) << tdr_index_stats.total_time
                << std::setw(16) << static_cast<double>(tdr_index_stats.index_size) / (1024.0 * 1024.0);
        }


        auto write_means = [&](const std::map<int, std::vector<std::pair<double, int> > > &table) {
            for (const int pattern: patterns) {
                for (int length = 1; length <= output_max_label_length; ++length) {
                    out << std::setw(16) << runtime_mean(table, pattern, length);
                }
            }
        };
        write_means(tdr_stats.runtime.true_qtime);
        write_means(tdr_stats.runtime.false_qtime);

        auto write_summary = [&](const QueryRunStats &stats) {
            for (const int pattern: patterns) {
                const auto count_it = stats.runtime.total_count.find(pattern);
                const int count = count_it == stats.runtime.total_count.end() ? 0 : count_it->second;
                const auto time_it = stats.runtime.total_time.find(pattern);
                const double total_time = time_it == stats.runtime.total_time.end() ? 0.0 : time_it->second;
                const auto true_count_it = stats.runtime.true_count.find(pattern);
                const int true_count = true_count_it == stats.runtime.true_count.end() ? 0 : true_count_it->second;
                const auto true_time_it = stats.runtime.true_time.find(pattern);
                const double true_time = true_time_it == stats.runtime.true_time.end() ? 0.0 : true_time_it->second;
                out << std::setw(12) << count << std::setw(16) << total_time
                        << std::setw(12) << true_count << std::setw(16) << true_time;
            }
        };
        write_summary(tdr_stats);
        if (compare) {
            write_summary(*baseline_stats);
            write_summary(*only_h_stats);
            write_summary(*only_v_stats);
            const VisitRatioStats only_h = calculate_visit_ratio(*baseline_stats, *only_h_stats);
            const VisitRatioStats only_v = calculate_visit_ratio(*baseline_stats, *only_v_stats);
            const VisitRatioStats tdr = calculate_visit_ratio(*baseline_stats, tdr_stats);
            out << std::setw(16) << only_h.ratio() << std::setw(16) << only_v.ratio()
                    << std::setw(16) << tdr.ratio()
                    << std::setw(12) << only_h.mismatches << std::setw(12) << only_v.mismatches
                    << std::setw(12) << tdr.mismatches;
        }
        out << '\n';
    }
    void write_runtime_file(const string &filename, const bool compare,
                            const QueryRunStats &tdr_stats,
                            const IndexBuildStats &only_h_index_stats,
                            const IndexBuildStats &only_v_index_stats,
                            const IndexBuildStats &tdr_index_stats,
                            const QueryRunStats *baseline_stats = nullptr,
                            const QueryRunStats *only_h_stats = nullptr,
                            const QueryRunStats *only_v_stats = nullptr) {
        std::ofstream out(filename, std::ios::out);
        if (!out.is_open()) {
            std::cerr << "Failed to create runtime output file: " << filename << std::endl;
            return;
        }
        out.setf(std::ios::fixed, std::ios::floatfield);
        out << std::setprecision(6);
        const std::vector<int> patterns = query_patterns();
        out << "index_build_time_ms only-H=" << only_h_index_stats.total_time
                << " only-V=" << only_v_index_stats.total_time
                << " TDR=" << tdr_index_stats.total_time << '\n';
        out << "index_space_mb only-H="
                << static_cast<double>(only_h_index_stats.index_size) / (1024.0 * 1024.0)
                << " only-V="
                << static_cast<double>(only_v_index_stats.index_size) / (1024.0 * 1024.0)
                << " TDR="
                << static_cast<double>(tdr_index_stats.index_size) / (1024.0 * 1024.0) << "\n\n";
        auto write_profile = [&](const char *name, const QueryRunStats &stats) {
            out << "[" << name << "]\n";
            for (const int pattern: patterns) {
                const auto it = stats.runtime.total_count.find(pattern);
                if (it == stats.runtime.total_count.end()) continue;
                out << "pattern=" << pattern
                        << " total_count=" << it->second
                        << " total_time_ms=" << map_double_or_zero(stats.runtime.total_time, pattern)
                        << " true_count=" << map_count_or_zero(stats.runtime.true_count, pattern)
                        << " true_time_ms=" << map_double_or_zero(stats.runtime.true_time, pattern) << '\n';
                out << "  true_mean_ms:";
                for (int length = 1; length <= output_max_label_length; ++length)
                    out << ' ' << runtime_mean(stats.runtime.true_qtime, pattern, length);
                out << "\n  false_mean_ms:";
                for (int length = 1; length <= output_max_label_length; ++length)
                    out << ' ' << runtime_mean(stats.runtime.false_qtime, pattern, length);
                out << '\n';
            }
            out << '\n';
        };
        if (compare && baseline_stats && only_h_stats && only_v_stats) {
            write_profile("baseline", *baseline_stats);
            write_profile("only-H", *only_h_stats);
            write_profile("only-V", *only_v_stats);
        }
        write_profile("TDR", tdr_stats);
        if (compare && baseline_stats && only_h_stats && only_v_stats) {
            const VisitRatioStats only_h = calculate_visit_ratio(*baseline_stats, *only_h_stats);
            const VisitRatioStats only_v = calculate_visit_ratio(*baseline_stats, *only_v_stats);
            const VisitRatioStats tdr = calculate_visit_ratio(*baseline_stats, tdr_stats);
            out << "[comparison]\n"
                    << "only-H edge_ratio=" << only_h.ratio() << " mismatches=" << only_h.mismatches << '\n'
                    << "only-V edge_ratio=" << only_v.ratio() << " mismatches=" << only_v.mismatches << '\n'
                    << "TDR edge_ratio=" << tdr.ratio() << " mismatches=" << tdr.mismatches << '\n';
        }
    }
    void write_result_file(const string &filename, const string &graph_name,
                           const IndexBuildStats &index_stats,
                           const QueryRunStats &tdr_stats,
                           const QueryRunStats *baseline_stats = nullptr,
                           const QueryRunStats *only_h_stats = nullptr,
                           const QueryRunStats *only_v_stats = nullptr) {
        const bool compare = baseline_stats && only_h_stats && only_v_stats;
        std::ofstream out(filename, std::ios::out);
        if (!out.is_open()) {
            std::cerr << "Failed to create result output file: " << filename << std::endl;
            return;
        }
        out.setf(std::ios::fixed, std::ios::floatfield);
        out << std::setprecision(6);
        const std::vector<int> patterns = query_patterns();
        out << "graph: " << graph_name << '\n'
                << "indexTime: " << index_stats.total_time << " ms\n"
                << "indexSpace: " << static_cast<double>(index_stats.index_size) / (1024.0 * 1024.0) << " MB\n\n";
        auto write_result_summary = [&](const char *name, const QueryRunStats &stats) {
            out << '[' << name << "]\n";
            for (const int pattern: patterns) {
                const auto it = stats.runtime.total_count.find(pattern);
                if (it == stats.runtime.total_count.end()) continue;
                out << "pattern=" << pattern << " total=" << it->second
                        << " time_ms=" << map_double_or_zero(stats.runtime.total_time, pattern)
                        << " reachable=" << map_count_or_zero(stats.runtime.true_count, pattern) << '\n';
            }
            out << '\n';
        };
        if (compare) {
            write_result_summary("baseline", *baseline_stats);
            write_result_summary("only-H", *only_h_stats);
            write_result_summary("only-V", *only_v_stats);
        }
        write_result_summary("TDR", tdr_stats);
        if (compare) {
            const VisitRatioStats only_h = calculate_visit_ratio(*baseline_stats, *only_h_stats);
            const VisitRatioStats only_v = calculate_visit_ratio(*baseline_stats, *only_v_stats);
            const VisitRatioStats tdr = calculate_visit_ratio(*baseline_stats, tdr_stats);
            out << "[comparison]\n"
                    << "only-H edge_ratio=" << only_h.ratio() << " mismatches=" << only_h.mismatches << endl
                    << "only-V edge_ratio=" << only_v.ratio() << " mismatches=" << only_v.mismatches << endl
                    << "TDR edge_ratio=" << tdr.ratio() << " mismatches=" << tdr.mismatches << endl << endl;
            out << "[TDR mismatches]\n";
            for (size_t i = 0; i < queries.size() && i < baseline_stats->outcomes.size(); ++i) {
                if (baseline_stats->outcomes[i] != tdr_stats.outcomes[i]) {
                    out << "query_index=" << i << " origin=" << queries[i].origin
                            << " destination=" << queries[i].destination
                            << " pattern=" << queries[i].pattern
                            << " baseline=" << baseline_stats->outcomes[i]
                            << " TDR=" << tdr_stats.outcomes[i] << '\n';
                }
            }
        }
    }
}
int main(const int argc, char *argv[]) {
    using namespace bs;

    if (argc < 3) {
        cerr << "command: ./TDR graphFile queryFile [--compare-pruning]" << endl;
        return 1;
    }
    bool compare_pruning = false;
    for (int i = 3; i < argc; ++i) {
        if (string arg = argv[i]; arg == "--compare-pruning" || arg == "--compare") {
            compare_pruning = true;
        } else {
            cerr << "Unknown option: " << arg << endl;
            cerr << "command: ./TDR graphFile queryFile [--compare-pruning]" << endl;
            return 1;
        }
    }

    const string input = argv[1];
    const auto found = input.rfind('/');
    const string graph_name = found != std::string::npos
                                  ? input.substr(found + 1)
                                  : input;
    cout << graph_name << ":" << endl;

    if (!read_graph(input))
        return 2;

    s = new int[num_V];
    s_capacity = num_V;
    SCC = new int[num_V];

    const string query_name = argv[2];
    const string reach_mode = query_name.substr(query_name.length() - 3);

    if (!read_queries(query_name))
        return 3;
    qs.init(num_V);
    if (reach_mode != "lcr") qsSup.init(num_V);
    bool has_whole_path_pattern = false;
    for (const Query &q: queries) {
        if (q.pattern == 1 || q.pattern == 2 || q.pattern == 5 || q.pattern == 6) {
            has_whole_path_pattern = true;
            break;
        }
    }
    if (has_whole_path_pattern) ps.init(num_V, label_hash_len);

    cout << "********* start to build index and answer queries! *********" << endl;
    if (compare_pruning) {
        configure_index_components(reach_mode, IndexProfile::OnlyH);
        const IndexBuildStats only_h_index_stats = index_construction("Only-H");
        const QueryRunStats only_h_stats = run_query_profile(QueryProfile::OnlyH);

        configure_index_components(reach_mode, IndexProfile::OnlyV);
        const IndexBuildStats only_v_index_stats = index_construction("Only-V");
        const QueryRunStats only_v_stats = run_query_profile(QueryProfile::OnlyV);

        configure_index_components(reach_mode, IndexProfile::TDR);
        const IndexBuildStats tdr_index_stats = index_construction("TDR");
        const QueryRunStats tdr_stats = run_query_profile(QueryProfile::Full);

        prepare_baseline_traversal_state();
        const QueryRunStats baseline_stats = run_query_profile(QueryProfile::Baseline);

        cout << endl;
        print_runtime_summary(tdr_stats, &baseline_stats, &only_h_stats, &only_v_stats);
    } else {
        configure_index_components(reach_mode, IndexProfile::TDR);
        const IndexBuildStats tdr_index_stats = index_construction("TDR");
        const QueryRunStats tdr_stats = run_query_profile(QueryProfile::Full);
        print_runtime_summary(tdr_stats);
    }

    delete[] s;
    delete[] SCC;

    return 0;
}
