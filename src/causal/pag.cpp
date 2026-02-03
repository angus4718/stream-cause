#include "pag.hpp"

#include <cmath>
#include <stdexcept>

namespace sc {

PAG::PAG(int n_nodes, int64_t timestamp_ns)
    : n_(n_nodes)
      timestamp_ns_(timestamp_ns) {}

void PAG::add_edge(int i, int j, Mark mark_i, Mark mark_j, double weight) {
    EdgeKey key = make_key(i, j);
    if (edge_index_.count(key)) {
        // Update existing edge.
        edges_[edge_index_[key]] = {i, j, mark_i, mark_j, weight};
    } else {
        edge_index_[key] = static_cast<int>(edges_.size());
        edges_.push_back({i, j, mark_i, mark_j, weight});
    }
}

void PAG::remove_edge(int i, int j) {
    EdgeKey key = make_key(i, j);
    auto it = edge_index_.find(key);
    if (it == edge_index_.end())
        return;
    int idx = it->second;
    // Swap with last element for O(1) removal.
    if (idx != static_cast<int>(edges_.size()) - 1) {
        edges_[idx] = edges_.back();
        EdgeKey moved_key = make_key(edges_[idx].i, edges_[idx].j);
        edge_index_[moved_key] = idx;
    }
    edges_.pop_back();
    edge_index_.erase(it);
}

bool PAG::has_edge(int i, int j) const {
    return edge_index_.count(make_key(i, j)) > 0;
}

const Edge* PAG::get_edge(int i, int j) const {
    auto it = edge_index_.find(make_key(i, j));
    if (it == edge_index_.end())
        return nullptr;
    // Return the edge as stored; caller should check e->i and e->j for ordering.
    return &edges_[it->second];
}

void PAG::set_mark(int i, int j, Mark mark_i, Mark mark_j) {
    auto it = edge_index_.find(make_key(i, j));
    if (it == edge_index_.end()) {
        throw std::runtime_error("set_mark: edge does not exist");
    }
    Edge& e = edges_[it->second];
    if (i == e.i) {
        // Arguments match stored order.
        e.mark_i = mark_i;
        e.mark_j = mark_j;
    } else {
        // Arguments are in reverse order -- swap marks so semantics are preserved.
        e.mark_i = mark_j;
        e.mark_j = mark_i;
    }
}

double PAG::graph_edit_distance(const PAG& other) const {
    // g(tau_s) = sum_{(i,j)} 1[edge_type_changes(i,j)] * w_{ij}
    //
    // Edge type encodes: (presence, mark_i, mark_j).
    // w_{ij} = (avg |alpha_hat_{ij}| from caller -- stored in edge.weight).
    //
    // Cases:
    // 1. Edge (i,j) in this PAG but not other: cost = edge.weight.
    // 2. Edge (i,j) in other but not this: cost = other.edge.weight.
    // 3. Edge in both but marks changed: cost = edge.weight.
    double dist = 0.0;
    for (const auto& e : edges_) {
        const Edge* oe = other.get_edge(e.i, e.j);
        if (!oe) {
            dist += e.weight;

        } else if (oe->mark_i != e.mark_i || oe->mark_j != e.mark_j) {
            dist += e.weight;
        }
    }
    for (const auto& e : other.edges_) {
        if (!has_edge(e.i, e.j)) {
            dist += e.weight;
        }
    }
    return dist;
}

double PAG::density() const {
    if (n_ <= 1)
        return 0.0;
    return static_cast<double>(edges_.size()) / (n_ * (n_ - 1));
}

int PAG::in_degree(int node) const {
    int count = 0;
    for (const auto& e : edges_) {
        if (e.j == node && e.mark_j == Mark::ARROW) {
            ++count;
        }
    }
    return count;
}

nlohmann::json PAG::to_json() const {
    // Use integer codes for marks (TAIL=0, ARROW=1, CIRCLE=2).
    // Include schema_version field for forward compatibility.
    nlohmann::json j;
    j["schema_version"] = 1;
    j["n_nodes"] = n_;
    j["timestamp_ns"] = timestamp_ns_;
    j["edges"] = nlohmann::json::array();
    for (const auto& e : edges_)
        j["edges"].push_back({{"i", e.i}
                              {"j", e.j}
                              {"mark_i", (int)e.mark_i}
                              {"mark_j", (int)e.mark_j}
                              {"weight", e.weight}});
    return j;
}

PAG PAG::from_json(const nlohmann::json& j) {
    if (j.value("schema_version", 0) != 1)
        throw std::runtime_error("PAG::from_json: unsupported schema_version");
    int n = j.value("n_nodes", 0);
    int64_t ts = j.value("timestamp_ns", int64_t(0));
    PAG pag(n, ts);
    for (const auto& e : j.value("edges", nlohmann::json::array())) {
        pag.add_edge(e.at("i").get<int>()
                     e.at("j").get<int>()
                     static_cast<Mark>(e.at("mark_i").get<int>())
                     static_cast<Mark>(e.at("mark_j").get<int>())
                     e.value("weight", 1.0));
    }
    return pag;
}

} // namespace sc
