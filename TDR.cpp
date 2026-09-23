#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <climits>
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
    constexpr int vertex_hash_len = 32;
    int label_hash_len = 1;
    constexpr int forward_default_k = 2;
    constexpr int UINT_bits = sizeof(uint32_t) * 8;

    bool build_group_label_index = true;
    bool build_forward_label_index = true;

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

    inline bool enable_base_pruning() {
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

        void add(const long long baseline, const long long profile) {
            baseline_edges += baseline;
            profile_edges += profile;
        }

        [[nodiscard]] double ratio() const {
            return baseline_edges == 0 ? 0.0 : static_cast<double>(profile_edges) / static_cast<double>(baseline_edges);
        }
    };

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
        return U.forward_depth[gid];
    }

    inline int forward_group_offset(const VertexIndex &U, const int gid) {
        if (gid < 0 || gid >= U.group || U.forward_offset.empty()) return 0;
        return static_cast<int>(U.forward_offset[gid]) * label_hash_len;
    }

    inline int forward_total_words(const VertexIndex &U) {
        return U.forward_offset.empty() ? 0 : static_cast<int>(U.forward_offset.back()) * label_hash_len;
    }

    inline int forward_layer_slot(const VertexIndex &U, const int gid, const int layer) {
        if (gid < 0 || gid >= U.group || layer < 0 || U.forward_offset.empty()) return -1;
        return static_cast<int>(U.forward_offset[gid]) + layer;
    }

    inline bool forward_layer_is_dense(const VertexIndex &U, const int gid, const int layer) {
        const int slot = forward_layer_slot(U, gid, layer);
        return slot >= 0 && slot < static_cast<int>(U.forward_layer_dense.size()) && U.forward_layer_dense[slot] != 0;
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

        void add(const int pattern, const int L, const bool outcome, const double qtime) {
            map<int, vector<pair<double, int> > > &bucket = outcome ? true_qtime : false_qtime;
            if (bucket.find(pattern) == bucket.end()) {
                bucket[pattern] = vector<pair<double, int> >(8, {0.0, 0});
            }
            vector<pair<double, int> > &items = bucket[pattern];
            if (static_cast<int>(items.size()) < L) items.resize(L);
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
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    }

    inline void reset_vertex_hash_state() {
        vertex_hash_leaf = 0;
        vertex_hash_parent = -1;
        vertex_hash_value = 0;
    }

    inline int vertex_hash(const int &parent) {
        if (parent == -1) {
            const int x = ++vertex_hash_leaf;
            return mix_hash(x);
        }
        if (vertex_hash_parent != parent) {
            vertex_hash_parent = parent;
            vertex_hash_value = parent;
        }
        vertex_hash_value++;
        return mix_hash(vertex_hash_value);
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

    inline void configure_index_components(const string & /*reach_mode*/,
                                           const IndexProfile profile = IndexProfile::TDR) {
        active_index_profile = profile;
        build_group_label_index = profile != IndexProfile::OnlyV;
        build_forward_label_index = profile != IndexProfile::OnlyH;
    }

    std::vector<int> horizontal_group_count;

    constexpr int group_merge_max_loss_percent = 40;
    constexpr int group_initial_max_groups = 8;
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
        if (active_index_profile == IndexProfile::OnlyV) return 1;
        const int cap = degree_group_cap(out_degree);
        if (uid >= 0 && uid < static_cast<int>(horizontal_group_count.size()) && horizontal_group_count[uid] > 0) {
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

        void init(const int numL, int /*L*/) {
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

    void prepare_horizontal_group_features(const std::vector<int> & /*label_count*/) {
        horizontal_group_count.assign(num_V, 1);
        for (int uid = 0; uid < num_V; ++uid) {
            horizontal_group_count[uid] = degree_group_cap(source_offset[uid + 1] - source_offset[uid]);
        }
    }

    inline void rebuild_forward_offsets(VertexIndex &U) {
        U.forward_offset.assign(U.forward_depth.size() + 1, 0);
        int prefix = 0;
        for (int gid = 0; gid < static_cast<int>(U.forward_depth.size()); ++gid) {
            U.forward_offset[gid] = prefix;
            prefix += static_cast<int>(U.forward_depth[gid]);
        }
        U.forward_offset[U.forward_depth.size()] = prefix;
    }

    inline void prepare_forward_layout(VertexIndex &U, const int group) {
        U.forward_depth.clear();
        U.forward_offset.clear();
        U.forward_layer_dense.clear();
        if (!build_forward_label_index || group <= 0) return;
        U.forward_depth.assign(group, forward_default_k);
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
            if (const int base = slot * label_hash_len; label_bitset_full(&U.labels_forward[base])) {
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

        std::vector new_offset(U.group + 1, 0);
        int prefix = 0;
        for (int gid = 0; gid < U.group; ++gid) {
            new_offset[gid] = prefix;
            prefix += static_cast<int>(new_depth[gid]);
        }
        new_offset[U.group] = prefix;

        std::vector<uint32_t> new_forward(static_cast<size_t>(prefix) * label_hash_len, 0);
        for (int gid = 0; gid < U.group; ++gid) {
            const int copy_depth = std::min<int>(forward_group_depth(U, gid), new_depth[gid]);
            if (copy_depth <= 0) continue;
            const int old_base = forward_group_offset(U, gid);
            const int new_base = new_offset[gid] * label_hash_len;
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

        std::vector<uint32_t> merged_vertices(static_cast<size_t>(new_group) * vertex_hash_len, 0);
        std::vector<uint32_t> merged_labels;
        if (build_group_label_index && !U.labels_group.empty()) {
            merged_labels.assign(static_cast<size_t>(new_group) * label_hash_len, 0);
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
            if (const long long extra = merge_extra_bits(U, candidate);
                extra * 100 <= old_bits * group_merge_max_loss_percent) {
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
            merged_vertices.assign(static_cast<size_t>(new_group) * vertex_hash_len, 0);
            for (int gid = 0; gid < old_group; ++gid) {
                const int ng = gid % new_group;
                or_words(&merged_vertices[ng * vertex_hash_len], &U.vertices_group[gid * vertex_hash_len],
                         vertex_hash_len);
            }

            if (build_group_label_index && !U.labels_group.empty()) {
                merged_labels.assign(static_cast<size_t>(new_group) * label_hash_len, 0);
                for (int gid = 0; gid < old_group; ++gid) {
                    const int ng = gid % new_group;
                    or_words(&merged_labels[ng * label_hash_len], &U.labels_group[gid * label_hash_len],
                             label_hash_len);
                }
            }
        }

        if (build_forward_label_index && !U.labels_forward.empty() && !U.forward_depth.empty()) {
            std::vector<uint8_t> new_depth(new_group, forward_default_k);
            for (int gid = 0; gid < old_group; ++gid) {
                const int ng = gid % new_group;
                const uint8_t old_depth = U.forward_depth[gid];
                new_depth[ng] = std::min(new_depth[ng], old_depth);
            }

            std::vector new_offset(new_group + 1, 0);
            int extra_depth_prefix = 0;
            for (int gid = 0; gid < new_group; ++gid) {
                new_offset[gid] = extra_depth_prefix;
                extra_depth_prefix += static_cast<int>(new_depth[gid]);
            }
            new_offset[new_group] = extra_depth_prefix;

            const size_t new_words = static_cast<size_t>(extra_depth_prefix) * label_hash_len;
            std::vector<uint32_t> new_forward(new_words, 0);
            for (int gid = 0; gid < old_group; ++gid) {
                const int ng = gid % new_group;
                const int old_depth = forward_group_depth(U, gid);
                const int old_base = forward_group_offset(U, gid);
                const int new_base = new_offset[ng] * label_hash_len;
                const int copy_depth = std::min(old_depth, static_cast<int>(new_depth[ng]));
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
                if (const int new_group = choose_merged_group_count(u); new_group < old_group) {
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
            return &path_bits[static_cast<size_t>(v) * static_cast<size_t>(label_hash_len)];
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

        void init(const std::vector<int> &idsL) {
            const int n = static_cast<int>(idsL.size());
            if (label_pos.size() != static_cast<size_t>(num_L)) {
                label_pos.assign(num_L, 255);
            } else {
                for (const int l: touched_labels) label_pos[l] = 255;
            }
            touched_labels.clear();
            first_matched_pos.clear();
            first_matched_pos.resize(n);
            target_bits = 0;
            uint8_t index = 0;
            for (auto &l: idsL) {
                label_pos[l] = index;
                touched_labels.push_back(l);
                first_matched_pos[index] = INT_MAX;
                target_bits |= 1u << index;
                index++;
            }
            current_bits = 0;
            path_label_seq.clear();
            if (visited_vertex.size() != static_cast<size_t>(num_V)) {
                visited_vertex.clear();
                visited_vertex.resize(num_V);
            } else {
                for (const int uid: touched_vertices) visited_vertex[uid].clear();
            }
            touched_vertices.clear();
        }

        [[nodiscard]] bool should_push_label(const int &l, const int &p) const {
            const uint8_t index = label_pos[l];
            if (index == 255) return false;
            return first_matched_pos[index] > p;
        }

        void push_label(const int &l, const bool &flag, const int &p) {
            if (flag) {
                const uint8_t index = label_pos[l];
                first_matched_pos[index] = p;
                path_label_seq.push_back(l);
                current_bits |= 1u << index;
            } else {
                path_label_seq.push_back(-1);
            }
        }

        bool should_revisit(const int &uid, const int &l, const bool &flag) {
            uint8_t new_bits = current_bits;
            if (flag) {
                const uint8_t index = label_pos[l];
                if (index == 255) return false;
                new_bits |= 1u << index;
            }
            for (const auto &saved_bits: visited_vertex[uid]) {
                if ((saved_bits & new_bits) == new_bits) {
                    return false;
                }
            }
            if (visited_vertex[uid].empty()) touched_vertices.push_back(uid);
            visited_vertex[uid].push_back(new_bits);
            return true;
        }

        void pop_label(const int &uid) {
            if (visited_vertex[uid].empty()) touched_vertices.push_back(uid);
            visited_vertex[uid].push_back(current_bits);
            if (path_label_seq.empty()) return;
            const int l = path_label_seq.back();
            path_label_seq.pop_back();
            if (l >= 0) {
                const uint8_t index = label_pos[l];
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
            num = static_cast<int>(idsSeq.size());
            need_pos = 0;
            if (matched_label.size() != static_cast<size_t>(num_V)) {
                matched_label.clear();
                matched_label.resize(num_V);
            } else {
                for (const int uid: touched_vertices) matched_label[uid].clear();
            }
            touched_vertices.clear();
            if (alternative.size() != static_cast<size_t>(num_V)) {
                alternative.assign(num_V, -1);
            } else {
                for (const int uid: touched_alternative) alternative[uid] = -1;
            }
            touched_alternative.clear();
        }

        void push_label_seq1(const int &uid, const int &l, const std::vector<int> &idsSeq) {
            if (matched_label[uid].empty()) touched_vertices.push_back(uid);
            matched_label[uid].push_back(l);
            if (l == idsSeq[need_pos]) {
                need_pos = need_pos < num - 1 ? need_pos + 1 : need_pos;
            }
        }

        void push_label_seq2(const int &uid, const int &l) {
            if (matched_label[uid].empty()) touched_vertices.push_back(uid);
            matched_label[uid].push_back(l);
            need_pos = (need_pos + 1) % num;
        }

        void pop_label_seq1(const int &uid, const std::vector<int> &idsSeq) {
            if (matched_label[uid].empty()) return;
            const int current_label = matched_label[uid].back();
            if (const int check_pos = need_pos > 0 ? need_pos - 1 : need_pos;
                current_label != idsSeq[check_pos] && current_label != idsSeq[num - 1]) {
                need_pos = check_pos;
            }
        }

        void pop_label_seq2() {
            need_pos = (need_pos - 1 + num) % num;
        }

        bool should_revisit(const int &uid, const int &l) {
            return std::none_of(matched_label[uid].begin(), matched_label[uid].end(),
                                [&l](const int &old_match) { return old_match == l; });
        }

        [[nodiscard]] bool accept(const int &uid, const std::vector<int> &idsSeq) const {
            return !matched_label[uid].empty() && matched_label[uid].back() == idsSeq[num - 1];
        }

        void set_alternative(const int &uid, const int &l) {
            if (alternative[uid] == -1) touched_alternative.push_back(uid);
            alternative[uid] = l;
        }

        int take_alternative(const int &uid) {
            const int l = alternative[uid];
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
            } else {
                return false;
            }
        }

        static bool feasible_bits(int &l_next, const int &end_pos,
                                  uint32_t *v_ps,
                                  const vector<uint32_t> &bitsL) {
            for (int i = l_next; i < end_pos; ++i) {
                const int tl = labels[i];
                const int p = lht.p[tl];
                const uint32_t mask = lht.mask[tl];
                const bool label_in_query = (bitsL[p] & mask) != 0;
                if constexpr (LM == LabelMatch::NOT) {
                    if (!label_in_query) {
                        return true;
                    }
                } else if constexpr (LM == LabelMatch::LCR) {
                    if (label_in_query) {
                        return true;
                    }
                } else {
                    if (const bool label_in_path = (v_ps[p] & mask) != 0; label_in_query && !label_in_path) {
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
                const bool label_in_query = contains_label(idsL, tl);
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
                        if (psExact.should_push_label(tl, pos)) {
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
        bool search(const int uid, const int vid,
                    const std::vector<uint32_t> &bitsL,
                    const std::vector<int> &idsL,
                    const std::vector<int> &idsSeq) {
            return search_impl(uid, vid, bitsL, idsL, idsSeq);
        }

    private:
        using matcher = LabelMatcher<LM>;

        [[nodiscard]] static bool vertex_prune(const VertexIndex &U, const VertexIndex &V) {
            if (!enable_base_pruning()) return false;
            bool pruned = false;
            if (V.group < 0) {
                const auto vh = static_cast<uint32_t>(-V.group);
                const int blk = vertex_hash_block(vh);
                const int bit = vertex_hash_bit(vh);
                pruned = (U.vertices_all[blk] & 1u << bit) == 0;
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

        [[nodiscard]] static bool label_prune_not_sequence(const int &uid, const std::vector<uint32_t> &bitsL) {
            if (!enable_base_pruning()) return false;

            if constexpr (M == Mode::WholePath && UseExact && LM == LabelMatch::OR) return false;

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

        [[nodiscard]] static bool label_prune_sequences(const VertexIndex &U,
                                                        const int &needed_pos,
                                                        const std::vector<int> &idsSeq) {
            if (!enable_base_pruning()) return false;
            const int n = static_cast<int>(idsSeq.size());
            for (int i = needed_pos; i < n; ++i) {
                const int l = idsSeq[i];
                if (const int p = lht.p[l]; (U.labels_all[p] & lht.mask[l]) == 0) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] static bool group_vertex_prune(const VertexIndex &U,
                                                     const VertexIndex &V,
                                                     const int &gid) {
            if (!enable_group_pruning()) return false;

            if constexpr (M == Mode::WholePath && UseExact && LM == LabelMatch::OR) return false;

            bool pruned = false;
            if (V.group < 0) {
                const auto vh = static_cast<uint32_t>(-V.group);
                const int blk = vertex_hash_block(vh);
                const int bit = vertex_hash_bit(vh);
                const uint32_t temp = U.vertices_group[gid * vertex_hash_len + blk];
                pruned = (temp & 1u << bit) == 0;
            } else {
                for (int i = 0; i < vertex_hash_len; ++i) {
                    if (const uint32_t temp = U.vertices_group[gid * vertex_hash_len + i];
                        (temp & V.vertices_all[i]) != V.vertices_all[i]) {
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

            if constexpr (M == Mode::WholePath && UseExact && LM == LabelMatch::OR) return false;

            const VertexIndex &U = vertices[uid];
            const int index = gid * label_hash_len;
            for (int i = 0; i < label_hash_len; ++i) {
                if (uint32_t vertex_bits = U.labels_group[index + i]; !matcher::prune_by_bits(
                    vertex_bits, ps.pl(uid)[i], bitsL[i])) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] bool forward_label_prune_all_labels(const int &gid,
                                                          const VertexIndex &U,
                                                          const std::vector<uint32_t> &bitsL) const {
            if (!enable_forward_pruning()) return false;
            if (const int depth = forward_group_depth(U, gid); depth <= 0 || forward_layer_is_dense(U, gid, 0))
                return false;
            const int index = forward_group_offset(U, gid);

            if constexpr (LM == LabelMatch::AND) {
                bool checked_forward_layer = false;
                for (int i = 0; i < label_hash_len; ++i) {
                    const uint32_t required_bits = bitsL[i];
                    if (required_bits == 0) continue;
                    if (!checked_forward_layer) {
                        checked_forward_layer = true;
                    }
                    if (const uint32_t vertex_bits = U.labels_forward[index + i];
                        (vertex_bits & required_bits) != required_bits) {
                        return true;
                    }
                }
                return false;
            } else {
                for (int i = 0; i < label_hash_len; ++i) {
                    if (uint32_t vertex_bits = U.labels_forward[index + i]; !matcher::prune_by_bits(
                        vertex_bits, 0, bitsL[i])) {
                        return false;
                    }
                }
                return true;
            }
        }

        [[nodiscard]] bool forward_label_prune_sequence1(const int &gid, const VertexIndex &U,
                                                         const std::vector<int> &idsSeq) const {
            if (!enable_forward_pruning()) return false;
            if (const int depth = forward_group_depth(U, gid); depth <= 0 || forward_layer_is_dense(U, gid, 0))
                return false;
            const int index = forward_group_offset(U, gid);
            const int needed_pos = sqs.need_pos;
            const int needed_label = idsSeq[needed_pos];
            const uint32_t mask = !!needed_pos;
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

        [[nodiscard]] static bool forward_label_prune_sequence2(const int &gid, const VertexIndex &U,
                                                                const std::vector<int> &idsSeq) {
            if (!enable_forward_pruning()) return false;
            const int depth = forward_group_depth(U, gid);
            if (depth <= 0) return false;
            const int needed_pos = sqs.need_pos;
            const int steps = std::min(depth, sqs.num - needed_pos);

            if (steps <= 0) {
                return false;
            }

            const int base = forward_group_offset(U, gid);
            bool checked_forward_layer = false;
            for (int i = 0; i < steps; ++i) {
                const int tl = idsSeq[needed_pos + i];
                if (forward_layer_is_dense(U, gid, i)) continue;
                if (!checked_forward_layer) {
                    checked_forward_layer = true;
                }
                const int temp = base + i * label_hash_len;
                if (const int p = lht.p[tl]; (U.labels_forward[temp + p] & lht.mask[tl]) == 0) {
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
                for (int i = 0; i < static_cast<int>(bitsL.size()); ++i) {
                    if (!matcher::accept_bits(ps.pl(uid)[i], bitsL[i])) {
                        return false;
                    }
                }
                return true;
            }
            return true;
        }

        static int feasible_sequence1(const int &uid, const std::vector<int> &idsSeq, int &l1, int &l2) {
            bool flag1 = false, flag2 = false;
            const int check_pos = sqs.need_pos > 0 ? sqs.need_pos - 1 : sqs.need_pos;
            int endLabel = -1;
            if (!sqs.matched_label[uid].empty()) endLabel = sqs.matched_label[uid].back();
            const bool endPos = endLabel == idsSeq[sqs.num - 1];
            for (int i = target_offset[qs.next[uid]]; i < target_offset[qs.next[uid] + 1]; ++i) {
                const int tl = labels[i];
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

        static bool feasible_sequence2(const int &uid, const std::vector<int> &idsSeq) {
            for (int i = target_offset[qs.next[uid]]; i < target_offset[qs.next[uid] + 1]; ++i) {
                if (const int tl = labels[i]; tl == idsSeq[sqs.need_pos]) {
                    return true;
                }
            }
            return false;
        }

        bool search_impl(const int uid, const int vid,
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
                        if (enable_base_pruning() && top_u.t_out < V.t_out && top_u.t_in <= V.t_in &&
                            label_accept_check(top_uid, bitsL))
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
                    }
                    finished = false;
                    break;
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
                                    if (int l = sqs.take_alternative(tid); sqs.should_revisit(tid, l)) {
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
            VertexIndex &top_v = vertices[top_vid];
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
                        bst.low[top_vid] = min(bst.low[top_vid], vertices[vid].t_in);
                    }
                }
                bst.next[top_vid]++;
            }

            if (finished) {
                if (const int n = max_neighbor - source_offset[top_vid]; n == 0) {
                    bst.hash_value[top_vid] = vertex_hash(-1);
                    top_v.group = static_cast<int>(-bst.hash_value[top_vid]);
                } else {
                    if (top_v.vertices_all.size() != static_cast<size_t>(vertex_hash_len))
                        top_v.vertices_all.resize(
                            vertex_hash_len);
                    if (top_v.labels_all.size() != static_cast<size_t>(label_hash_len))
                        top_v.labels_all.resize(
                            label_hash_len);
                    std::fill(top_v.vertices_all.begin(), top_v.vertices_all.end(), 0);
                    std::fill(top_v.labels_all.begin(), top_v.labels_all.end(), 0);
                    const int group = prepared_horizontal_group_count(top_vid, n);
                    top_v.group = group;
                    if (build_forward_label_index) {
                        prepare_forward_layout(top_v, group);
                        if (const auto forward_size = static_cast<size_t>(forward_total_words(top_v));
                            top_v.labels_forward.size() != forward_size)
                            top_v.labels_forward.resize(forward_size);
                        std::fill(top_v.labels_forward.begin(), top_v.labels_forward.end(), 0);
                    } else {
                        vector<uint32_t>().swap(top_v.labels_forward);
                        vector<uint8_t>().swap(top_v.forward_depth);
                        vector<int>().swap(top_v.forward_offset);
                        vector<uint8_t>().swap(top_v.forward_layer_dense);
                    }
                    if (group > 1) {
                        const size_t vertices_group_size =
                                static_cast<size_t>(group) * static_cast<size_t>(vertex_hash_len);
                        if (top_v.vertices_group.size() != vertices_group_size)
                            top_v.vertices_group.resize(
                                vertices_group_size);
                        std::fill(top_v.vertices_group.begin(), top_v.vertices_group.end(), 0);
                        if (build_group_label_index) {
                            const size_t labels_group_size =
                                    static_cast<size_t>(group) * static_cast<size_t>(label_hash_len);
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
                        const int vid = targets[i], index = order % group;
                        int hv;
                        order++;
                        VertexIndex &v = vertices[vid];
                        if (bst.hash_value[vid] == 0) {
                            hv = vertex_hash(top_vid);
                            bst.hash_value[vid] = hv;
                        } else {
                            hv = static_cast<int>(bst.hash_value[vid]);
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
                        const int p = vertex_hash_block(static_cast<uint32_t>(hv));
                        const int h = vertex_hash_bit(static_cast<uint32_t>(hv));
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
                        const int tid = SCC[SCC_pos];
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

    struct IndexBuildStats {
        double total_time = 0.0;
        long long index_size = 0;
        long long index_space_base = 0;
        long long index_space_horizontal = 0;
        long long index_space_vertical = 0;
    };

    void reset_index_state() {
        cur = 0;
        vis_cur += 2;
        s_pos = 0;
        SCC_pos = 0;
        reset_vertex_hash_state();
        bst.init(num_V);
        horizontal_group_count.clear();
        for (auto &v: vertices) {
            v.t_in = 0;
            v.t_out = 0;
            v.group = 0;
            vector<uint32_t>().swap(v.vertices_all);
            vector<uint32_t>().swap(v.labels_all);
            vector<uint32_t>().swap(v.labels_group);
            vector<uint32_t>().swap(v.vertices_group);
            vector<uint32_t>().swap(v.labels_forward);
            vector<uint8_t>().swap(v.forward_depth);
            vector<int>().swap(v.forward_offset);
            vector<uint8_t>().swap(v.forward_layer_dense);
        }
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
            VertexIndex &v = vertices[uid];
            v.t_in = 0;
            v.t_out = 0;
            v.group = source_offset[uid] == source_offset[uid + 1] ? -1 : 1;
            vector<uint32_t>().swap(v.vertices_all);
            vector<uint32_t>().swap(v.labels_all);
            vector<uint32_t>().swap(v.labels_group);
            vector<uint32_t>().swap(v.vertices_group);
            vector<uint32_t>().swap(v.labels_forward);
            vector<uint8_t>().swap(v.forward_depth);
            vector<int>().swap(v.forward_offset);
            vector<uint8_t>().swap(v.forward_layer_dense);
        }
    }

    IndexBuildStats calculate_index_space() {
        IndexBuildStats stats;
        for (auto &v: vertices) {
            stats.index_space_base += sizeof(v.t_in) + sizeof(v.t_out);
            stats.index_space_base += static_cast<long long>(sizeof(uint32_t)) * label_hash_len;
            stats.index_space_base += static_cast<long long>(sizeof(uint32_t)) * vertex_hash_len;
            if (v.group < 0)
                continue;
            if (build_forward_label_index) {
                stats.index_space_vertical += static_cast<long long>(sizeof(uint32_t) * v.labels_forward.size());
                stats.index_space_vertical += static_cast<long long>(sizeof(uint8_t) * v.forward_depth.size());
                stats.index_space_vertical += static_cast<long long>(sizeof(int) * v.forward_offset.size());
                stats.index_space_vertical += static_cast<long long>(sizeof(uint8_t) * v.forward_layer_dense.size());
            }
            if (v.group == 1) continue;
            stats.index_space_horizontal += static_cast<long long>(sizeof(uint32_t) * v.group * vertex_hash_len);
            if (build_group_label_index)
                stats.index_space_horizontal += static_cast<long long>(sizeof(uint32_t) * v.group * label_hash_len);
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
        printf("indexSpace: %.3fMB\n", static_cast<double>(stats.index_size) / (1024 * 1024));
    }

    IndexBuildStats index_construction(const string &profile_name) {
        reset_index_state();
        auto start_time = std::chrono::high_resolution_clock::now();
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
                    const int vid = targets[i];
                    const int index = order % u.group;
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
                                const int index_u = u_base + pi * label_hash_len;
                                for (int gi = 0; gi < v.group; ++gi) {
                                    if (forward_group_depth(v, gi) <= pi - 1) continue;
                                    const int index_v = forward_group_offset(v, gi) + (pi - 1) * label_hash_len;
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
                            const int index_u = u_base + pi * label_hash_len;
                            for (int gi = 0; gi < v.group; ++gi) {
                                if (forward_group_depth(v, gi) <= pi - 1) continue;
                                const int index_v = forward_group_offset(v, gi) + (pi - 1) * label_hash_len;
                                or_words(&u.labels_forward[index_u], &v.labels_forward[index_v], label_hash_len);
                            }
                        }
                    }
                }
            }
        }
        merge_groups_after_construction();
        finalize_forward_density_flags();
        const auto end_time = std::chrono::high_resolution_clock::now();

        IndexBuildStats stats = calculate_index_space();
        stats.total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        print_index_build_stats(profile_name, stats);
        return stats;
    }

    bool read_queries(const string &filename) {
        auto start_time = std::chrono::high_resolution_clock::now();

        ifstream file;
        file.open(filename, ios::in);
        if (!file.is_open()) {
            cerr << "Failed to read query file!" << endl;
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
            if (q.origin == q.destination) {
                outcome = true;
            } else {
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
                only_h_ratio.add(baseline_stats->edge_visits[i], only_h_stats->edge_visits[i]);
                only_v_ratio.add(baseline_stats->edge_visits[i], only_v_stats->edge_visits[i]);
                tdr_ratio.add(baseline_stats->edge_visits[i], tdr_stats.edge_visits[i]);
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
    SCC = new int[num_V];

    const string query_name = argv[2];
    const string reach_mode = query_name.substr(query_name.length() - 3);

    if (!read_queries(query_name))
        return 3;
    qs.init(num_V);
    if (reach_mode != "lcr") qsSup.init(num_V);
    if (reach_mode == "pcr") ps.init(num_V, label_hash_len);

    cout << "********* start to build index and answer queries! *********" << endl;
    if (compare_pruning) {
        configure_index_components(reach_mode, IndexProfile::OnlyH);
        index_construction("Only-H");
        const QueryRunStats only_h_stats = run_query_profile(QueryProfile::OnlyH);

        configure_index_components(reach_mode, IndexProfile::OnlyV);
        index_construction("Only-V");
        const QueryRunStats only_v_stats = run_query_profile(QueryProfile::OnlyV);

        configure_index_components(reach_mode, IndexProfile::TDR);
        index_construction("TDR");
        const QueryRunStats tdr_stats = run_query_profile(QueryProfile::Full);

        prepare_baseline_traversal_state();
        const QueryRunStats baseline_stats = run_query_profile(QueryProfile::Baseline);

        cout << endl;
        print_runtime_summary(tdr_stats, &baseline_stats, &only_h_stats, &only_v_stats);
    } else {
        configure_index_components(reach_mode, IndexProfile::TDR);
        index_construction("TDR");
        const QueryRunStats tdr_stats = run_query_profile(QueryProfile::Full);
        print_runtime_summary(tdr_stats);
    }

    delete[] s;
    delete[] SCC;

    return 0;
}
