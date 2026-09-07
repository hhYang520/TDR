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
#include <map>
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
    int s_pos;
    int *SCC;
    int SCC_pos;
    const int vertex_hash_len = 32;
    int label_hash_len = 1;
    const int forward_default_k = 2;
    const int UINT_bits = sizeof(uint32_t) * 8;

    bool build_group_label_index = true;
    bool build_forward_label_index = true;

    inline int vertex_hash_block(const uint32_t hv) {
        return (int) (hv & (vertex_hash_len - 1));
    }

    inline int vertex_hash_bit(const uint32_t hv) {
        return (int) ((hv >> 5) & (UINT_bits - 1));
    }

    enum class QueryProfile { Baseline, Full, OnlyV, OnlyH };

    QueryProfile active_profile = QueryProfile::Full;
    long long query_edge_visits = 0;

    inline bool enable_interval_pruning() {
        return active_profile != QueryProfile::Baseline;
    }

    inline bool enable_all_pruning() {
        return active_profile != QueryProfile::Baseline;
    }

    inline bool enable_group_pruning() {
        return active_profile == QueryProfile::Full || active_profile == QueryProfile::OnlyH;
    }

    inline bool enable_forward_pruning() {
        return build_forward_label_index && (active_profile == QueryProfile::Full || active_profile ==
                                             QueryProfile::OnlyV);
    }

    struct VisitRatioStats {
        long long baseline_edges = 0;
        long long profile_edges = 0;
        long long queries = 0;

        void add(long long baseline, long long profile, bool same_outcome) {
            baseline_edges += baseline;
            profile_edges += profile;
            queries++;
        }

        void reset() {
            baseline_edges = 0;
            profile_edges = 0;
            queries = 0;
        }

        [[nodiscard]] double ratio() const {
            return baseline_edges == 0 ? 0.0 : (double) profile_edges / (double) baseline_edges;
        }
    };

    VisitRatioStats ratio_only_h, ratio_only_v, ratio_tdr;

    struct VertexIndex {
        bool root{};
        int t_in{}, t_out{};
        vector<uint32_t> vertices_all;
        vector<uint32_t> labels_all;
        int group{};
        vector<uint32_t> labels_group;
        vector<uint32_t> vertices_group;
        vector<uint32_t> labels_forward;
        vector<uint8_t> forward_depth;
        vector<int> forward_offset;
        vector<uint8_t> forward_layer_dense;
    };

    vector<VertexIndex> vertices;
    vector<uint32_t> label_universe_bits;

    inline int forward_group_depth(const VertexIndex &U, const int gid) {
        if (gid < 0 || gid >= U.group || U.forward_depth.empty()) return 0;
        return (int) U.forward_depth[gid];
    }

    inline int forward_group_offset(const VertexIndex &U, const int gid) {
        if (gid < 0 || gid >= U.group || U.forward_offset.empty()) return 0;
        return (int) U.forward_offset[gid] * label_hash_len;
    }

    inline int forward_total_words(const VertexIndex &U) {
        return U.forward_offset.empty() ? 0 : (int) U.forward_offset.back() * label_hash_len;
    }

    inline int forward_layer_slot(const VertexIndex &U, const int gid, const int layer) {
        if (gid < 0 || gid >= U.group || layer < 0 || U.forward_offset.empty()) return -1;
        return (int) U.forward_offset[gid] + layer;
    }

    inline bool forward_layer_is_dense(const VertexIndex &U, const int gid, const int layer) {
        const int slot = forward_layer_slot(U, gid, layer);
        return slot >= 0 && slot < (int) U.forward_layer_dense.size() && U.forward_layer_dense[slot] != 0;
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
        const int group = std::max(1, vertices[uid].group);
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

        void add(int pattern, int L, bool outcome, double qtime) {
            map<int, vector<pair<double, int> > > &bucket = outcome ? true_qtime : false_qtime;
            if (bucket.find(pattern) == bucket.end()) {
                bucket[pattern] = vector<pair<double, int> >(8, {0.0, 0});
            }
            vector<pair<double, int> > &items = bucket[pattern];
            if ((int) items.size() < L) items.resize(L);
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

    inline int vertex_hash(const int &parent) {
        if (parent == -1) {
            static int leaf = 0;
            int x = ++leaf;
            x ^= x >> 16;
            x *= 0x7feb352d;
            x ^= x >> 15;
            x *= 0x846ca68b;
            x ^= x >> 16;
            return x;
        }
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

    template<class Fn>
    inline void for_each_label(const string &s, Fn &&fn) {
        int value = 0;
        bool has_digit = false;
        bool negative = false;
        for (char c: s) {
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
        for (int l: idsL) {
            if (l == label) return true;
        }
        return false;
    }

    inline void configure_index_components(const string & /*reach_mode*/) {
        build_group_label_index = true;
        build_forward_label_index = true;
    }

    std::vector<int> horizontal_group_count;

    const int group_merge_max_loss_percent = 40;
    const int group_initial_max_groups = 8;
    long long group_merge_vertices = 0;
    long long group_merge_before = 0;
    long long group_merge_after = 0;

    inline int degree_group_cap(const int out_degree) {
        if (out_degree <= 8) return 1;
        if (out_degree <= 32) return 2;
        if (out_degree <= 128) return 4;
        return std::min(group_initial_max_groups, out_degree);
    }

    inline int prepared_horizontal_group_count(const int uid, const int out_degree) {
        if (out_degree <= 0) return 0;
        const int cap = degree_group_cap(out_degree);
        if (uid >= 0 && uid < (int) horizontal_group_count.size() && horizontal_group_count[uid] > 0) {
            return std::min(cap, horizontal_group_count[uid]);
        }
        return cap;
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

        void init(int numL, int /*L*/) {
            p.resize(numL);
            h.resize(numL);
            mask.resize(numL);
            for (int l = 0; l < numL; ++l) {
                const int blk = l >> 5;
                p[l] = blk;
                h[l] = l & (UINT_bits - 1);
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

    void prepare_horizontal_group_features(const std::vector<int> & /*label_count*/) {
        horizontal_group_count.assign(num_V, 1);
        for (int uid = 0; uid < num_V; ++uid) {
            horizontal_group_count[uid] = degree_group_cap(source_offset[uid + 1] - source_offset[uid]);
        }
    }

    inline void rebuild_forward_offsets(VertexIndex &U) {
        U.forward_offset.assign(U.forward_depth.size() + 1, 0);
        int prefix = 0;
        for (int gid = 0; gid < (int) U.forward_depth.size(); ++gid) {
            U.forward_offset[gid] = prefix;
            prefix += (int) U.forward_depth[gid];
        }
        U.forward_offset[U.forward_depth.size()] = prefix;
    }

    inline void prepare_forward_layout(VertexIndex &U, const int group) {
        U.forward_depth.clear();
        U.forward_offset.clear();
        U.forward_layer_dense.clear();
        if (!build_forward_label_index || group <= 0) return;
        U.forward_depth.assign(group, (uint8_t) forward_default_k);
        rebuild_forward_offsets(U);
    }

    void refresh_forward_density_flags(VertexIndex &U) {
        if (!build_forward_label_index || U.group <= 0 || U.labels_forward.empty()) {
            U.forward_layer_dense.clear();
            return;
        }
        const int layers = forward_total_words(U) / label_hash_len;
        U.forward_layer_dense.assign(layers, 0);
        for (int slot = 0; slot < layers; ++slot) {
            const int base = slot * label_hash_len;
            if (label_bitset_full(&U.labels_forward[base])) {
                U.forward_layer_dense[slot] = 1;
            }
        }
    }

    void compact_forward_layers(VertexIndex &U) {
        if (!build_forward_label_index || U.group <= 0 || U.labels_forward.empty() || U.forward_depth.empty()) {
            U.forward_layer_dense.clear();
            return;
        }

        std::vector<uint8_t> new_depth(U.group, 0);
        for (int gid = 0; gid < U.group; ++gid) {
            const int old_depth = forward_group_depth(U, gid);
            if (old_depth <= 0) continue;

            const int base = forward_group_offset(U, gid);
            const uint32_t *layer0 = &U.labels_forward[base];
            const bool first_useful = !label_bitset_empty(layer0) && !label_bitset_full(layer0);
            bool second_useful = false;
            if (old_depth >= 2) {
                const uint32_t *layer1 = layer0 + label_hash_len;
                second_useful = !label_bitset_empty(layer1) && !label_bitset_full(layer1);
            }

            if (second_useful) {
                new_depth[gid] = std::min<int>(old_depth, forward_default_k);
            } else if (first_useful) {
                new_depth[gid] = 1;
            } else {
                new_depth[gid] = 0;
            }
        }

        std::vector<int> new_offset(U.group + 1, 0);
        int prefix = 0;
        for (int gid = 0; gid < U.group; ++gid) {
            new_offset[gid] = prefix;
            prefix += (int) new_depth[gid];
        }
        new_offset[U.group] = prefix;

        std::vector<uint32_t> new_forward((size_t) prefix * label_hash_len, 0);
        for (int gid = 0; gid < U.group; ++gid) {
            const int copy_depth = std::min<int>(forward_group_depth(U, gid), new_depth[gid]);
            if (copy_depth <= 0) continue;
            const int old_base = forward_group_offset(U, gid);
            const int new_base = (int) new_offset[gid] * label_hash_len;
            for (int layer = 0; layer < copy_depth; ++layer) {
                or_words(&new_forward[new_base + layer * label_hash_len],
                         &U.labels_forward[old_base + layer * label_hash_len],
                         label_hash_len);
            }
        }

        U.labels_forward.swap(new_forward);
        U.forward_depth.swap(new_depth);
        U.forward_offset.swap(new_offset);
        U.forward_layer_dense.clear();
    }

    void finalize_forward_density_flags() {
        if (!build_forward_label_index) return;
        for (auto &v: vertices) compact_forward_layers(v);
        for (auto &v: vertices) refresh_forward_density_flags(v);
    }

    inline long long count_group_bits(const VertexIndex &U, const int gid) {
        long long total = 0;
        const int vertex_base = gid * vertex_hash_len;
        for (int i = 0; i < vertex_hash_len; ++i) {
            total += popcount32(U.vertices_group[vertex_base + i]);
        }
        if (build_group_label_index && !U.labels_group.empty()) {
            const int label_base = gid * label_hash_len;
            for (int i = 0; i < label_hash_len; ++i) {
                total += popcount32(U.labels_group[label_base + i]);
            }
        }
        return total;
    }

    long long merge_extra_bits(const VertexIndex &U, const int new_group) {
        if (new_group <= 0 || new_group >= U.group) return 0;

        std::vector<uint32_t> merged_vertices((size_t) new_group * vertex_hash_len, 0);
        std::vector<uint32_t> merged_labels;
        if (build_group_label_index && !U.labels_group.empty()) {
            merged_labels.assign((size_t) new_group * label_hash_len, 0);
        }

        for (int gid = 0; gid < U.group; ++gid) {
            const int ng = gid % new_group;
            or_words(&merged_vertices[ng * vertex_hash_len], &U.vertices_group[gid * vertex_hash_len], vertex_hash_len);
            if (!merged_labels.empty()) {
                or_words(&merged_labels[ng * label_hash_len], &U.labels_group[gid * label_hash_len], label_hash_len);
            }
        }

        long long extra = 0;
        for (int gid = 0; gid < U.group; ++gid) {
            const int ng = gid % new_group;
            const int old_vertex_base = gid * vertex_hash_len;
            const int new_vertex_base = ng * vertex_hash_len;
            for (int i = 0; i < vertex_hash_len; ++i) {
                extra += popcount32(merged_vertices[new_vertex_base + i] & ~U.vertices_group[old_vertex_base + i]);
            }
            if (!merged_labels.empty()) {
                const int old_label_base = gid * label_hash_len;
                const int new_label_base = ng * label_hash_len;
                for (int i = 0; i < label_hash_len; ++i) {
                    extra += popcount32(merged_labels[new_label_base + i] & ~U.labels_group[old_label_base + i]);
                }
            }
        }

        return extra;
    }

    int choose_merged_group_count(const VertexIndex &U) {
        if (U.group <= 1 || U.vertices_group.empty()) return U.group;

        long long old_bits = 0;
        for (int gid = 0; gid < U.group; ++gid) old_bits += count_group_bits(U, gid);
        if (old_bits <= 0) return 1;

        for (int candidate = 1; candidate < U.group; ++candidate) {
            if (U.group % candidate != 0) continue;
            const long long extra = merge_extra_bits(U, candidate);
            if (extra * 100 <= old_bits * group_merge_max_loss_percent) {
                return candidate;
            }
        }
        return U.group;
    }

    void merge_vertex_groups(VertexIndex &U, const int new_group) {
        const int old_group = U.group;
        if (new_group <= 0 || new_group >= old_group || old_group % new_group != 0) return;

        std::vector<uint32_t> merged_vertices;
        std::vector<uint32_t> merged_labels;
        if (new_group > 1) {
            merged_vertices.assign((size_t) new_group * vertex_hash_len, 0);
            for (int gid = 0; gid < old_group; ++gid) {
                const int ng = gid % new_group;
                or_words(&merged_vertices[ng * vertex_hash_len], &U.vertices_group[gid * vertex_hash_len],
                         vertex_hash_len);
            }

            if (build_group_label_index && !U.labels_group.empty()) {
                merged_labels.assign((size_t) new_group * label_hash_len, 0);
                for (int gid = 0; gid < old_group; ++gid) {
                    const int ng = gid % new_group;
                    or_words(&merged_labels[ng * label_hash_len], &U.labels_group[gid * label_hash_len],
                             label_hash_len);
                }
            }
        }

        if (build_forward_label_index && !U.labels_forward.empty() && !U.forward_depth.empty()) {
            std::vector<uint8_t> new_depth(new_group, (uint8_t) forward_default_k);
            for (int gid = 0; gid < old_group; ++gid) {
                const int ng = gid % new_group;
                const uint8_t old_depth = U.forward_depth[gid];
                new_depth[ng] = std::min(new_depth[ng], old_depth);
            }

            std::vector<int> new_offset(new_group + 1, 0);
            int extra_depth_prefix = 0;
            for (int gid = 0; gid < new_group; ++gid) {
                new_offset[gid] = extra_depth_prefix;
                extra_depth_prefix += (int) new_depth[gid];
            }
            new_offset[new_group] = extra_depth_prefix;

            const size_t new_words = (size_t) extra_depth_prefix * label_hash_len;
            std::vector<uint32_t> new_forward(new_words, 0);
            for (int gid = 0; gid < old_group; ++gid) {
                const int ng = gid % new_group;
                const int old_depth = forward_group_depth(U, gid);
                const int old_base = forward_group_offset(U, gid);
                const int new_base = (int) new_offset[ng] * label_hash_len;
                const int copy_depth = std::min(old_depth, (int) new_depth[ng]);
                for (int layer = 0; layer < copy_depth; ++layer) {
                    or_words(&new_forward[new_base + layer * label_hash_len],
                             &U.labels_forward[old_base + layer * label_hash_len],
                             label_hash_len);
                }
            }

            U.labels_forward.swap(new_forward);
            U.forward_depth.swap(new_depth);
            U.forward_offset.swap(new_offset);
            U.forward_layer_dense.clear();
        }

        U.group = new_group;
        if (new_group == 1) {
            vector<uint32_t>().swap(U.vertices_group);
            vector<uint32_t>().swap(U.labels_group);
        } else {
            U.vertices_group.swap(merged_vertices);
            U.labels_group.swap(merged_labels);
        }
    }

    void merge_groups_after_construction() {
        group_merge_vertices = 0;
        group_merge_before = 0;
        group_merge_after = 0;

        for (auto &u: vertices) {
            if (u.group <= 0) continue;
            group_merge_before += u.group;
            if (u.group > 1) {
                const int old_group = u.group;
                const int new_group = choose_merged_group_count(u);
                if (new_group < old_group) {
                    merge_vertex_groups(u, new_group);
                    ++group_merge_vertices;
                }
            }
            group_merge_after += u.group;
        }
    }

    struct BuildState {
        std::vector<int> vis;
        std::vector<int> next;
        std::vector<int> low;
        std::vector<uint32_t> hash_value;

        void init(int n) {
            vis.assign(n, 0);
            next.assign(n, 0);
            low.assign(n, 0);
            hash_value.assign(n, 0);
        }
    } bst;

    struct QueryState {
        std::vector<int> vis;
        std::vector<int> next;
        std::vector<int> gid;
        std::vector<uint32_t> scratch_bits;

        void init(const int &n) {
            vis.assign(n, 0);
            next.resize(n);
            gid.resize(n);
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

        [[nodiscard]] bool vertex_prune(const VertexIndex &U, const VertexIndex &V) const {
            if (!enable_all_pruning()) return false;
            bool pruned = false;
            if (V.group < 0) {
                uint32_t vh = (uint32_t) (-V.group);
                int blk = vertex_hash_block(vh);
                int bit = vertex_hash_bit(vh);
                pruned = (U.vertices_all[blk] & (1u << bit)) == 0;
            } else {
                for (int i = 0; i < vertex_hash_len; ++i) {
                    if ((U.vertices_all[i] & V.vertices_all[i]) != V.vertices_all[i]) {
                        pruned = true;
                        break;
                    }
                }
            }
            return pruned;
        }

        [[nodiscard]] bool label_prune_not_sequence(const int &uid, const std::vector<uint32_t> &bitsL) const {
            if (!enable_all_pruning()) return false;
            const VertexIndex &U = vertices[uid];
            for (int i = 0; i < label_hash_len; ++i) {
                uint32_t path_bits = 0u;
                if constexpr (M != Mode::AllLabel) path_bits = ps.pl(uid)[i];
                if (!matcher::prune_by_bits(U.labels_all[i], path_bits, bitsL[i])) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] bool label_prune_sequences(const VertexIndex &U,
                                                 const int &needed_pos,
                                                 const std::vector<int> &idsSeq) const {
            if (!enable_all_pruning()) return false;
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

        [[nodiscard]] bool group_vertex_prune(const VertexIndex &U,
                                              const VertexIndex &V,
                                              const int &gid) const {
            if (!enable_group_pruning()) return false;
            bool pruned = false;
            if (V.group < 0) {
                uint32_t vh = (uint32_t) (-V.group);
                int blk = vertex_hash_block(vh);
                int bit = vertex_hash_bit(vh);
                uint32_t temp = U.vertices_group[gid * vertex_hash_len + blk];
                pruned = (temp & (1u << bit)) == 0;
            } else {
                for (int i = 0; i < vertex_hash_len; ++i) {
                    uint32_t temp = U.vertices_group[gid * vertex_hash_len + i];
                    if ((temp & V.vertices_all[i]) != V.vertices_all[i]) {
                        pruned = true;
                        break;
                    }
                }
            }
            return pruned;
        }

        [[nodiscard]] bool group_label_prune(const int &uid,
                                             const std::vector<uint32_t> &bitsL,
                                             const int &gid) const {
            if (!enable_group_pruning()) return false;
            const VertexIndex &U = vertices[uid];
            int index = gid * label_hash_len;
            for (int i = 0; i < label_hash_len; ++i) {
                uint32_t vertex_bits = U.labels_group[index + i];
                if (!matcher::prune_by_bits(vertex_bits, ps.pl(uid)[i], bitsL[i])) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] bool forward_label_prune_all_labels(const int &gid,
                                                          const VertexIndex &U,
                                                          const std::vector<uint32_t> &bitsL) const {
            if (!enable_forward_pruning()) return false;
            const int depth = forward_group_depth(U, gid);
            if (depth <= 0 || forward_layer_is_dense(U, gid, 0)) return false;
            const int index = forward_group_offset(U, gid);

            if constexpr (LM == LabelMatch::AND) {
                bool checked_forward_layer = false;
                for (int i = 0; i < label_hash_len; ++i) {
                    const uint32_t required_bits = bitsL[i];
                    if (required_bits == 0) continue;
                    if (!checked_forward_layer) {
                        checked_forward_layer = true;
                    }
                    uint32_t vertex_bits = U.labels_forward[index + i];
                    if ((vertex_bits & required_bits) != required_bits) {
                        return true;
                    }
                }
                return false;
            } else {
                for (int i = 0; i < label_hash_len; ++i) {
                    uint32_t vertex_bits = U.labels_forward[index + i];
                    if (!matcher::prune_by_bits(vertex_bits, 0, bitsL[i])) {
                        return false;
                    }
                }
                return true;
            }
        }

        [[nodiscard]] bool forward_label_prune_sequence1(const int &gid, const VertexIndex &U,
                                                         const std::vector<int> &idsSeq) const {
            if (!enable_forward_pruning()) return false;
            const int depth = forward_group_depth(U, gid);
            if (depth <= 0 || forward_layer_is_dense(U, gid, 0)) return false;
            int index = forward_group_offset(U, gid);
            int needed_pos = sqs.need_pos;
            const int needed_label = idsSeq[needed_pos];
            uint32_t mask = !!needed_pos;
            const int current_label = idsSeq[needed_pos - mask];
            bool checked_forward_layer = false;

            auto missing_label = [&](const int label) {
                if (!checked_forward_layer) {
                    checked_forward_layer = true;
                }
                const int p = lht.p[label];
                return (U.labels_forward[index + p] & lht.mask[label]) == 0;
            };

            if (missing_label(needed_label) ||
                (current_label != needed_label && missing_label(current_label))) {
                return true;
            }
            return false;
        }

        [[nodiscard]] bool forward_label_prune_sequence2(const int &gid, const VertexIndex &U,
                                                         const std::vector<int> &idsSeq) const {
            if (!enable_forward_pruning()) return false;
            const int depth = forward_group_depth(U, gid);
            if (depth <= 0) return false;
            int needed_pos = sqs.need_pos;
            const int steps = std::min(depth, sqs.num - needed_pos);

            if (steps <= 0) {
                return false;
            }

            const int base = forward_group_offset(U, gid);
            bool checked_forward_layer = false;
            for (int i = 0; i < steps; ++i) {
                int tl = idsSeq[needed_pos + i];
                if (forward_layer_is_dense(U, gid, i)) continue;
                if (!checked_forward_layer) {
                    checked_forward_layer = true;
                }
                int temp = base + i * label_hash_len;
                const int p = lht.p[tl];
                if ((U.labels_forward[temp + p] & lht.mask[tl]) == 0) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool label_accept_check(const int &uid, const std::vector<uint32_t> &bitsL) const {
            if constexpr (UseExact) {
                if (matcher::accept_exact(psExact.current_bits, psExact.target_bits))
                    return true;
            } else {
                for (int i = 0; i < (int) bitsL.size(); ++i) {
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
            const VertexIndex &V = vertices[vid];
            if (V.root) return false;

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
                VertexIndex &top_u = vertices[top_uid];
                if constexpr (M == Mode::Sequence1) {
                    sqs.pop_label_seq1(top_uid, idsSeq);
                }

                if (isAdd) {
                    isAdd = false;
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

                    if (enable_interval_pruning() && top_u.t_out < V.t_out) {
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
                        if (enable_interval_pruning() && top_u.t_in <= V.t_in && label_accept_check(top_uid, bitsL))
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
                        const int current_gid = qs.gid[top_uid];
                        if constexpr (M == Mode::AllLabel) {
                            if (forward_label_prune_all_labels(current_gid, top_u, bitsL)) {
                                continue;
                            }
                        } else if constexpr (M == Mode::Sequence1) {
                            if (forward_label_prune_sequence1(current_gid, top_u, idsSeq)) {
                                continue;
                            }
                        } else if constexpr (M == Mode::Sequence2) {
                            if (forward_label_prune_sequence2(current_gid, top_u, idsSeq)) {
                                continue;
                            }
                        } else {
                            if (group_label_prune(top_uid, bitsL, current_gid)) {
                                continue;
                            }
                        }

                        if (group_vertex_prune(top_u, V, current_gid)) {
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
                    query_edge_visits++;
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
                                    matcher::feasible_bits(qsSup.l_next[top_uid],
                                                           target_offset[edge_idx + 1],
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
                                    sqs.set_alternative(tid, l2);
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
                            if constexpr (M == Mode::WholePath) {
                                if constexpr (UseExact) {
                                    int tl = 0;
                                    bool flag = matcher::feasible_exact(qsSup.l_next[top_uid],
                                                                        target_offset[edge_idx + 1],
                                                                        idsL, tl, s_pos);
                                    if (psExact.should_revisit(tid, tl, flag)) {
                                        isAdd = true;
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
                                    matcher::feasible_bits(qsSup.l_next[top_uid],
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
                                        s[++s_pos] = tid;
                                        finished = false;
                                        break;
                                    }
                                }
                            } else {
                                if (feasible_sequence2(top_uid, idsSeq) && sqs.
                                    should_revisit(tid, idsSeq[sqs.need_pos])) {
                                    isAdd = true;
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
    private:
        template<int Ptn>
        bool searcher(int uid, int vid,
                      const std::vector<uint32_t> &bitsL,
                      const std::vector<int> &idsL,
                      const std::vector<int> &idsSeq) {
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
                throw std::invalid_argument("Invalid ptn value");
            }
        }

    public:
        bool search_dispatch(int ptn, int uid, int vid,
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
        clock_t start_time, end_time;
        start_time = clock();
        ifstream file;
        file.open(filename, ios::in);
        string str;
        if (!file.is_open()) {
            cout << "Failed to read graph file!" << endl;
            return false;
        }

        cout << "********* start to read graph! *********" << endl;
        file >> str >> num_V >> str >> num_TE >> str >> num_LE >> str >> num_L;
        vertices.resize(num_V);
        label_hash_len = std::max(1, (num_L + UINT_bits - 1) / UINT_bits);
        for (auto &vtx: vertices) {
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
            targets[v_pos] = v;
            vertices[v].root = false;
            for_each_label(str, [&](const int l) {
                labels[l_pos] = l;
                graph_label_count[l]++;
                l_pos++;
                target_offset[v_pos + 1] = l_pos;
            });
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


        cout << "### Label Distribution ###" << endl;
        for (int i = 0; i < num_L; i++) {
            cout << "Label" << i << ": " << graph_label_count[i] << "/" << num_LE << " = "
                    << graph_label_count[i] / (double) num_TE << endl;
        }
        end_time = clock();
        double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
        printf("read time(graph): %.3fs\n", total_time);
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
                if (bst.vis[vid] < vis_cur) {
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

            if (finished) {
                int n = max_neighbor - source_offset[top_vid];
                if (n == 0) {
                    bst.hash_value[top_vid] = vertex_hash(-1);
                    top_v.group = (int) -bst.hash_value[top_vid];
                } else {
                    if (top_v.vertices_all.size() != (size_t) vertex_hash_len)
                        top_v.vertices_all.resize(
                            vertex_hash_len);
                    if (top_v.labels_all.size() != (size_t) label_hash_len) top_v.labels_all.resize(label_hash_len);
                    std::fill(top_v.vertices_all.begin(), top_v.vertices_all.end(), 0);
                    std::fill(top_v.labels_all.begin(), top_v.labels_all.end(), 0);
                    int group = prepared_horizontal_group_count(top_vid, n);
                    top_v.group = group;
                    if (build_forward_label_index) {
                        prepare_forward_layout(top_v, group);
                        const size_t forward_size = (size_t) forward_total_words(top_v);
                        if (top_v.labels_forward.size() != forward_size) top_v.labels_forward.resize(forward_size);
                        std::fill(top_v.labels_forward.begin(), top_v.labels_forward.end(), 0);
                    } else {
                        vector<uint32_t>().swap(top_v.labels_forward);
                        vector<uint8_t>().swap(top_v.forward_depth);
                        vector<int>().swap(top_v.forward_offset);
                        vector<uint8_t>().swap(top_v.forward_layer_dense);
                    }
                    if (group > 1) {
                        const size_t vertices_group_size = (size_t) group * (size_t) vertex_hash_len;
                        if (top_v.vertices_group.size() != vertices_group_size)
                            top_v.vertices_group.resize(
                                vertices_group_size);
                        std::fill(top_v.vertices_group.begin(), top_v.vertices_group.end(), 0);
                        if (build_group_label_index) {
                            const size_t labels_group_size = (size_t) group * (size_t) label_hash_len;
                            if (top_v.labels_group.size() != labels_group_size)
                                top_v.labels_group.resize(
                                    labels_group_size);
                            std::fill(top_v.labels_group.begin(), top_v.labels_group.end(), 0);
                        } else {
                            vector<uint32_t>().swap(top_v.labels_group);
                        }
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
                        const int forward_base = forward_group_offset(top_v, index);
                        const int group_base = index * label_hash_len;
                        for (int li = target_offset[i]; li < target_offset[i + 1]; ++li) {
                            const int label = labels[li];
                            const int b = lht.p[label];
                            const uint32_t edge_bits = lht.mask[label];
                            top_v.labels_all[b] |= edge_bits;
                            if (build_forward_label_index && forward_group_depth(top_v, index) > 0)
                                top_v.labels_forward[forward_base + b] |= edge_bits;
                            if (group > 1 && build_group_label_index) top_v.labels_group[group_base + b] |= edge_bits;
                        }
                        int p = vertex_hash_block((uint32_t) hv), h = vertex_hash_bit((uint32_t) hv);
                        top_v.vertices_all[p] |= (1u << h);
                        if (group > 1) top_v.vertices_group[index * vertex_hash_len + p] |= (1u << h);
                        if (v.vertices_all.empty()) continue;
                        if (source_offset[vid] != source_offset[vid + 1]) {
                            or_words(top_v.vertices_all.data(), v.vertices_all.data(), vertex_hash_len);
                            or_words(top_v.labels_all.data(), v.labels_all.data(), label_hash_len);
                        }
                    }
                }
                top_v.t_out = ++cur;

                if (top_v.t_in == bst.low[top_vid]) {
                    while (true) {
                        int tid = SCC[SCC_pos];
                        VertexIndex &t = vertices[tid];
                        SCC_pos--;
                        bst.vis[tid] = vis_cur + 1;
                        if (tid == top_vid) break;
                        t.t_in = top_v.t_in;
                        t.t_out = top_v.t_out;
                        t.labels_all = top_v.labels_all;
                        t.vertices_all = top_v.vertices_all;
                    }
                }
                s_pos--;
            }
        }
    }

    void index_construction() {
        cout << "********* start to build index! *********" << endl;
        clock_t start_time, end_time;
        start_time = clock();
        prepare_horizontal_group_features(graph_label_count);
        for (int u = 0; u < num_V; ++u) {
            if (vertices[u].root) {
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
            VertexIndex &u = vertices[uid];
            if (u.group > 1) {
                int order = 0;
                for (int i = source_offset[uid]; i < source_offset[uid + 1]; i++) {
                    int vid = targets[i], index = order % u.group;
                    order++;
                    VertexIndex &v = vertices[vid];
                    if (source_offset[vid] != source_offset[vid + 1]) {
                        or_words(&u.vertices_group[index * vertex_hash_len], v.vertices_all.data(), vertex_hash_len);
                        if (build_group_label_index) {
                            or_words(&u.labels_group[index * label_hash_len], v.labels_all.data(), label_hash_len);
                        }
                        if (build_forward_label_index) {
                            const int u_depth = forward_group_depth(u, index);
                            const int u_base = forward_group_offset(u, index);
                            for (int pi = 1; pi < u_depth; ++pi) {
                                int index_u = u_base + pi * label_hash_len;
                                for (int gi = 0; gi < v.group; ++gi) {
                                    if (forward_group_depth(v, gi) <= pi - 1) continue;
                                    int index_v = forward_group_offset(v, gi) + (pi - 1) * label_hash_len;
                                    or_words(&u.labels_forward[index_u], &v.labels_forward[index_v], label_hash_len);
                                }
                            }
                        }
                    }
                }
            } else if (u.group == 1 && build_forward_label_index) {
                for (int i = source_offset[uid]; i < source_offset[uid + 1]; i++) {
                    int vid = targets[i];
                    VertexIndex &v = vertices[vid];
                    if (source_offset[vid] != source_offset[vid + 1]) {
                        const int u_depth = forward_group_depth(u, 0);
                        const int u_base = forward_group_offset(u, 0);
                        for (int pi = 1; pi < u_depth; ++pi) {
                            int index_u = u_base + pi * label_hash_len;
                            for (int gi = 0; gi < v.group; ++gi) {
                                if (forward_group_depth(v, gi) <= pi - 1) continue;
                                int index_v = forward_group_offset(v, gi) + (pi - 1) * label_hash_len;
                                or_words(&u.labels_forward[index_u], &v.labels_forward[index_v], label_hash_len);
                            }
                        }
                    }
                }
            }
        }
        double group_merge_time = 0.0;
        {
            clock_t merge_start = clock();
            merge_groups_after_construction();
            clock_t merge_end = clock();
            group_merge_time = (double) (merge_end - merge_start) / CLOCKS_PER_SEC * 1000;
        }
        finalize_forward_density_flags();
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
        cout << "groupIndex: " << (build_group_label_index ? "on" : "off") << endl;
        cout << "forwardIndex: " << (build_forward_label_index ? "on" : "off") << endl;

        long long index_size = 0;
        long long index_space_base = 0;
        long long index_space_horizontal = 0;
        long long index_space_vertical = 0;
        for (auto &v: vertices) {
            index_space_base += (sizeof(v.t_in) + sizeof(v.t_out));
            index_space_base += (long long) sizeof(uint32_t) * label_hash_len;
            index_space_base += (long long) sizeof(uint32_t) * vertex_hash_len;
            if (v.group < 0)
                continue;
            if (build_forward_label_index) {
                index_space_vertical += (long long) sizeof(uint32_t) * (long long) v.labels_forward.size();
                index_space_vertical += (long long) sizeof(uint8_t) * (long long) v.forward_depth.size();
                index_space_vertical += (long long) sizeof(int) * (long long) v.forward_offset.size();
                index_space_vertical += (long long) sizeof(uint8_t) * (long long) v.forward_layer_dense.size();
            }
            if (v.group == 1) continue;
            index_space_horizontal += (long long) sizeof(uint32_t) * v.group * vertex_hash_len;
            if (build_group_label_index)
                index_space_horizontal += (long long) sizeof(uint32_t) * v.group * label_hash_len;
        }
        index_size = index_space_base + index_space_horizontal + index_space_vertical;
        printf("indexSpace: %.3fMB\n", double(index_size) / (1024 * 1024));

        printf("indexSpaceBreakdown(H/V): %.3fMB %.3fMB\n",
               double(index_space_base + index_space_horizontal) / (1024 * 1024),
               double(index_space_base + index_space_vertical) / (1024 * 1024));
        cout << endl;
    }

    bool read_queries(const string &filename) {
        clock_t start_time, end_time;
        start_time = clock();

        ifstream file;
        file.open(filename, ios::in);
        if (!file.is_open()) {
            cout << "Failed to read query file!" << endl;
            return false;
        }
        cout << "********* start to answer queries! *********" << endl;
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
        end_time = clock();
        double total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;

        auto found = filename.rfind('/');
        string name;
        if (found != std::string::npos) {
            string temp(filename.begin() + found + 1, filename.end());
            name = temp;
        } else {
            name = filename;
        }
        cout << "The time to read " << name << " is: " << total_time << endl;
        return true;
    }

    void run_queries(const bool compare_pruning) {
        clock_t start_time, end_time;
        ratio_only_h.reset();
        ratio_only_v.reset();
        ratio_tdr.reset();

        ProfileRuntimeStats runtime_baseline, runtime_only_h, runtime_only_v, runtime_tdr;
        ReachableQuery reach;
        auto execute_query = [&](Query &q, const QueryProfile profile, long long &edge_visits, double &qtime) {
            active_profile = profile;
            query_edge_visits = 0;
            start_time = clock();
            if (q.origin == q.destination) {
                end_time = clock();
                qtime = (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
                edge_visits = 0;
                return true;
            }
            reset_query_marks_if_needed();
            s_pos = 0;
            vis_cur += 2;
            bool outcome = reach.search_dispatch(q.pattern, q.origin, q.destination,
                                                 q.bits_label, q.ids_label, q.ids_sequence);
            end_time = clock();
            qtime = (double) (end_time - start_time) / CLOCKS_PER_SEC * 1000;
            edge_visits = query_edge_visits;
            return outcome;
        };
        for (auto &q: queries) {
            int L = (int) q.ids_label.size();
            if (q.pattern > 8) L = (int) q.ids_sequence.size();

            long long baseline_edges = 0;
            long long only_h_edges = 0;
            long long only_v_edges = 0;
            long long tdr_edges = 0;
            double baseline_time = 0.0;
            double only_h_time = 0.0;
            double only_v_time = 0.0;
            double tdr_time = 0.0;
            bool baseline_outcome = false;
            bool only_h_outcome = false;
            bool only_v_outcome = false;

            if (compare_pruning) {
                baseline_outcome = execute_query(q, QueryProfile::Baseline, baseline_edges, baseline_time);
                only_h_outcome = execute_query(q, QueryProfile::OnlyH, only_h_edges, only_h_time);
                only_v_outcome = execute_query(q, QueryProfile::OnlyV, only_v_edges, only_v_time);
                runtime_baseline.add(q.pattern, L, baseline_outcome, baseline_time);
                runtime_only_h.add(q.pattern, L, only_h_outcome, only_h_time);
                runtime_only_v.add(q.pattern, L, only_v_outcome, only_v_time);
            }

            q.outcome = execute_query(q, QueryProfile::Full, tdr_edges, tdr_time);
            runtime_tdr.add(q.pattern, L, q.outcome, tdr_time);
            if (compare_pruning) {
                ratio_only_h.add(baseline_edges, only_h_edges, baseline_outcome == only_h_outcome);
                ratio_only_v.add(baseline_edges, only_v_edges, baseline_outcome == only_v_outcome);
                ratio_tdr.add(baseline_edges, tdr_edges, baseline_outcome == q.outcome);
            }
        }

        cout << "### Runtime of TDR ###" << endl;
        for (auto &p: runtime_tdr.total_count) {
            cout << "pattern: " << p.first << " total number: " << p.second << " total time: "
                    << runtime_tdr.total_time[p.first] << " ms" << endl;
        }
        if (compare_pruning) {
            cout << "### Visited Edge Ratio vs Baseline ###" << endl;
            cout << "only-H: " << ratio_only_h.profile_edges << "/" << ratio_only_h.baseline_edges
                    << "=" << ratio_only_h.ratio() << endl;
            cout << "only-V: " << ratio_only_v.profile_edges << "/" << ratio_only_v.baseline_edges
                    << "=" << ratio_only_v.ratio() << endl;
            cout << "TDR: " << ratio_tdr.profile_edges << "/" << ratio_tdr.baseline_edges
                    << "=" << ratio_tdr.ratio() << endl;
        }
        cout << "********* Finish! *********" << endl;
    }
}

int main(int argc, char *argv[]) {
    using namespace bs;

    if (argc < 3) {
        cerr << "command: ./TDR graphFile queryFile [--compare-pruning]" << endl;
        return 1;
    }
    bool compare_pruning = false;
    for (int i = 3; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--compare-pruning" || arg == "--compare") {
            compare_pruning = true;
        } else {
            cerr << "Unknown option: " << arg << endl;
            cerr << "command: ./TDR graphFile queryFile [--compare-pruning]" << endl;
            return 1;
        }
    }

    string input = argv[1];
    auto found = input.rfind('/');
    string graph_name;
    if (found != std::string::npos) {
        string temp(input.begin() + found + 1, input.end());
        graph_name = temp;
    } else {
        graph_name = input;
    }
    cout << graph_name << ":" << endl;

    if (!read_graph(input))
        return 2;

    s = new int[num_V];
    SCC = new int[num_V];

    string query_name = argv[2];
    string reach_mode = query_name.substr(query_name.length() - 3);
    configure_index_components(reach_mode);

    index_construction();


    if (!read_queries(query_name))
        return 3;
    qs.init(num_V);
    if (reach_mode != "lcr") qsSup.init(num_V);
    if (reach_mode == "pcr") ps.init(num_V, label_hash_len);
    run_queries(compare_pruning);

    delete[] s;
    delete[] SCC;

    return 0;
}
