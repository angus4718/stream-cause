#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>

namespace sc {

// Edge endpoint marks in a Partial Ancestral Graph (PAG).
// An edge between i and j has one mark at each end.
// TAIL (-): i has no arrowhead on its side
// ARROW (>): arrowhead at this endpoint
// CIRCLE (o): uncertain (ambiguously oriented)
//
// Common edge types (mark_i is the mark at node i's endpoint):
// i -> j: mark_i=TAIL, mark_j=ARROW
// i <-> j: mark_i=ARROW, mark_j=ARROW (bidirected: latent confounder)
// i o-> j: mark_i=CIRCLE, mark_j=ARROW
// i o-o j:mark_i=CIRCLE, mark_j=CIRCLE
enum class Mark : uint8_t { TAIL = 0, ARROW = 1, CIRCLE = 2 };

struct Edge {
    int i, j;
    Mark mark_i; // mark at node i's endpoint
    Mark mark_j; // mark at node j's endpoint
    double weight; // avg alpha_hat_{ij} magnitude (used in graph edit distance
};

// Partial Ancestral Graph snapshot at time tau_s.
// Stored as adjacency list for O(degree) edge lookup.
class PAG {
public:
    explicit PAG(int n_nodes = 0, int64_t timestamp_ns = 0);

    void add_edge(int i, int j, Mark mark_i, Mark mark_j, double weight = 1.0);
    void remove_edge(int i, int j);
    bool has_edge(int i, int j) const;
    // Returns nullptr if no edge exists.
    const Edge* get_edge(int i, int j) const;
    void set_mark(int i, int j, Mark mark_i, Mark mark_j);

    // All edges in the PAG.
    const std::vector<Edge>& edges() const { return edges_; }

    // Weighted graph edit distance to another PAG snapshot:
    // g(tau_s) = sum_{(i,j)} 1[edge_type changes] * w_{ij}
    // where w_{ij} proportional to average |alpha_hat_{ij}| magnitude of the pair.
    //
    double graph_edit_distance(const PAG& other) const;

    int n_nodes() const { return n_; }
    int64_t timestamp_ns() const { return timestamp_ns_; }
    int n_edges() const { return static_cast<int>(edges_.size()); }

    // Graph density: |E| / (N*(N-1)).
    double density() const;

    // In-degree of node i (number of arrows pointing into i).
    int in_degree(int i) const;

    // Serialize/deserialize for GraphStore (JSON format).
    nlohmann::json to_json() const;
    static PAG from_json(const nlohmann::json& j);

private:
    // Edge lookup: key = min(i,j)*N + max(i,j) -- symmetric, order-independent.
    using EdgeKey = int;
    EdgeKey make_key(int i, int j) const {
        return (i < j ? i : j) * n_ + (i < j ? j : i);
    }

    int n_;
    int64_t timestamp_ns_;
    std::vector<Edge> edges_;
    std::unordered_map<EdgeKey, int> edge_index_; // key -> index in edges_
};

} // namespace sc
