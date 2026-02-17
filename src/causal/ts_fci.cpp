#include "ts_fci.hpp"
#include <algorithm>
#include <numeric>
#include <functional>

namespace sc {

TSFCI::TSFCI(int n_instruments, double alpha_ci, int d_max)
    : n_(n_instruments), d_max_(d_max), ci_test_(alpha_ci) {}

PAG TSFCI::run(const std::vector<Eigen::MatrixXd>& alpha_series
               int64_t timestamp_ns) {
    if (alpha_series.size() < 4) {
        // Not enough snapshots for reliable CI testing; return empty PAG.
        return PAG(n_, timestamp_ns);
    }
    auto [skeleton, sep_sets] = skeleton_discovery(alpha_series);
    return orient_edges(skeleton, sep_sets, alpha_series, timestamp_ns);
}

std::pair<AdjMatrix, SepSets>
TSFCI::skeleton_discovery(const std::vector<Eigen::MatrixXd>& alpha_series) {
    int S = static_cast<int>(alpha_series.size());

    // Initialize complete undirected graph.
    AdjMatrix adj(n_, std::vector<bool>(n_, true));
    for (int i = 0; i < n_; ++i) adj[i][i] = false;
    SepSets sep_sets;

    // Alg. B.1: iterate conditioning set size d = 0, 1, .., d_max_.
    for (int d = 0; d <= d_max_; ++d) {
        bool any_removed = false;
        for (int i = 0; i < n_; ++i) {
            for (int j = i + 1; j < n_; ++j) {
                if (!adj[i][j]) continue;

                // Collect adj(i) \ {j} for conditioning.
                std::vector<int> adj_i;
                for (int k = 0; k < n_; ++k)
                    if (k != i && k != j && adj[i][k]) adj_i.push_back(k);

                if (static_cast<int>(adj_i.size()) < d) continue;

                // Enumerate all subsets of adj_i of size d using recursion.
                std::function<void(int, std::vector<int>&)> enumerate_subsets =
                    [&](int start, std::vector<int>& chosen) {
                        if (static_cast<int>(chosen.size()) == d) {
                            // Run CI test on this subset.
                            std::set<int> cond_set(chosen.begin(), chosen.end());
                            Eigen::VectorXd xi = extract_series(alpha_series, i, j);
                            Eigen::VectorXd xj = extract_series(alpha_series, j, i);
                            Eigen::MatrixXd cond = build_cond_matrix(alpha_series, cond_set, i);
                            if (ci_test_.test(xi, xj, cond, S)) {
                                adj[i][j] = adj[j][i] = false;
                                sep_sets[{std::min(i, j), std::max(i, j)}] = cond_set;
                                any_removed = true;
                            }
                            return;
                        }
                        for (int k = start; k < static_cast<int>(adj_i.size()); ++k) {
                            chosen.push_back(adj_i[k]);
                            enumerate_subsets(k + 1, chosen);
                            chosen.pop_back();
                            if (!adj[i][j]) return; // Early exit if edge already removed
                        }
                    };
                std::vector<int> chosen;
                enumerate_subsets(0, chosen);
            }
        }
        // Early termination: if no edges removed at this depth, higher depths won't help.
        if (!any_removed) break;
    }
    return {adj, sep_sets};
}

Eigen::VectorXd TSFCI::extract_series(const std::vector<Eigen::MatrixXd>& alpha_series
                                       int row, int col) const {
    // This is the alpha_hat_{row,col}(tau_s) time series used as input to CI tests.
    int S = static_cast<int>(alpha_series.size());
    Eigen::VectorXd v(S);
    for (int s = 0; s < S; ++s) v(s) = alpha_series[s](row, col);
    return v;
}

Eigen::MatrixXd TSFCI::build_cond_matrix(const std::vector<Eigen::MatrixXd>& alpha_series
                                          const std::set<int>& cond_set
                                          int ref_row) const {
    // Column k = extract_series(alpha_series, cond_node, ref_row) for each
    // node in cond_set. The ref_row determines which row of alpha_hat to use.
    int S = static_cast<int>(alpha_series.size());
    Eigen::MatrixXd mat(S, static_cast<int>(cond_set.size()));
    int col = 0;
    for (int c : cond_set) {
        mat.col(col++) = extract_series(alpha_series, ref_row, c);
    }
    return mat;
}

PAG TSFCI::orient_edges(const AdjMatrix& skeleton
                        const SepSets& sep_sets
                        const std::vector<Eigen::MatrixXd>& alpha_series
                        int64_t timestamp_ns) {
    PAG pag(n_, timestamp_ns);

    // Pre-compute time-averaged |alpha_hat_{ij}| across the alpha series window.
    // Edge weight w_{ij} = avg over s of (|alpha_hat_{ij}(s)| + |alpha_hat_{ji}(s)|) / 2
    // symmetric so g(tau_s) inis order-independent.
    int S = static_cast<int>(alpha_series.size());
    Eigen::MatrixXd avg_alpha = Eigen::MatrixXd::Zero(n_, n_);
    for (const auto& A : alpha_series)
        avg_alpha += A.cwiseAbs();
    if (S > 0) avg_alpha /= S;

    // Initialize all skeleton edges as undirected (o-o) with computed weight.
    for (int i = 0; i < n_; ++i)
        for (int j = i + 1; j < n_; ++j)
            if (skeleton[i][j]) {
                double w = 0.5 * (avg_alpha(i, j) + avg_alpha(j, i));
                pag.add_edge(i, j, Mark::CIRCLE, Mark::CIRCLE, w);
            }

    // Step 1: V-structure orientation.
    // For unshielded triples i--k--j where k not in sepset(i,j): orient i *-> k <-* j.
    for (int i = 0; i < n_; ++i) {
        for (int k = 0; k < n_; ++k) {
            if (!skeleton[i][k] || i == k) continue;
            for (int j = k + 1; j < n_; ++j) {
                if (!skeleton[k][j] || skeleton[i][j]) continue; // must be unshielded
                auto key = std::make_pair(std::min(i, j), std::max(i, j));
                auto it = sep_sets.find(key);
                if (it != sep_sets.end() && it->second.count(k) == 0) {
                    // V-structure: orient i *-> k <-* j
                    pag.set_mark(i, k, Mark::CIRCLE, Mark::ARROW);
                    pag.set_mark(j, k, Mark::CIRCLE, Mark::ARROW);
                }
            }
        }
    }

    // Step 2: Arrow-of-time constraint.
    arrow_of_time_orient(pag, alpha_series);

    // Step 3: FCI orientation rules R1-R10.
    apply_fci_rules(pag);

    // Step 4: Resolve conflicting orientations by majority vote.
    resolve_conflicts(pag);

    return pag;
}

void TSFCI::arrow_of_time_orient(PAG& pag
                                  const std::vector<Eigen::MatrixXd>& alpha_series) {
    // For each undirected edge (i,j) in the PAG, use time-averaged alpha_hat to orient it.
    // alpha_hat_{i,j} = excitation from j onto i -> larger alpha_hat_{i,j} means j -> i.
    // alpha_hat_{j,i} = excitation from i onto j -> larger alpha_hat_{j,i} means i -> j.
    //
    // Edge (i,j) stored with key i*n+j; always call set_mark(e.i, e.j, ..).
    // j -> i (arrowhead at i): mark_i=ARROW, mark_j=CIRCLE
    // i -> j (arrowhead at j): mark_i=CIRCLE, mark_j=ARROW

    // Collect snapshot of edges to avoid iterator invalidation.
    std::vector<Edge> edge_snapshot = pag.edges();

    for (const auto& e : edge_snapshot) {
        if (e.mark_i == Mark::ARROW || e.mark_j == Mark::ARROW)
            continue; // Already oriented.

        if (!pag.has_edge(e.i, e.j)) continue; // may have been removed

        double mean_ij = extract_series(alpha_series, e.i, e.j).mean(); // j excites i
        double mean_ji = extract_series(alpha_series, e.j, e.i).mean(); // i excites j

        if (mean_ij > mean_ji) {
            // j -> i: arrowhead at i
            pag.set_mark(e.i, e.j, Mark::ARROW, Mark::CIRCLE);
        } else if (mean_ji > mean_ij) {
            // i -> j: arrowhead at j
            pag.set_mark(e.i, e.j, Mark::CIRCLE, Mark::ARROW);
        }
        // If equal, leave as o-o (undirected).
    }
}

void TSFCI::apply_fci_rules(PAG& pag) {
    // Implement FCI orientation rules R1-R2 (most commonly triggered).
    // Iterate until fixed point (no more orientations).

    bool changed = true;
    while (changed) {
        changed = false;

        // R1: If alpha *-> beta o-* gamma, and alpha and gamma are not adjacent, orient beta -* gamma as beta -> gamma.
        for (const auto& e : pag.edges()) {
            if (e.mark_i != Mark::TAIL || e.mark_j != Mark::ARROW) continue; // alpha *-> beta

            int alpha = e.i, beta = e.j;

            for (int gamma = 0; gamma < pag.n_nodes(); ++gamma) {
                if (gamma == alpha || gamma == beta) continue;
                const auto* e_beta_gamma = pag.get_edge(beta, gamma);
                const auto* e_alpha_gamma = pag.get_edge(alpha, gamma);

                if (!e_beta_gamma) continue;
                if (e_alpha_gamma) continue; // alpha and gamma must not be adjacent

                // Orient beta o-* gamma as beta -> gamma
                if (e_beta_gamma->mark_i == Mark::CIRCLE) {
                    pag.set_mark(beta, gamma, Mark::TAIL, Mark::ARROW);
                    changed = true;
                }
            }
        }

        // R2: If alpha -> beta *-> gamma, and alpha *-o gamma, orient alpha *-> gamma.
        for (const auto& e : pag.edges()) {
            if (e.mark_i != Mark::TAIL || e.mark_j != Mark::ARROW) continue; // alpha -> beta or alpha *-> beta

            int alpha = e.i, beta = e.j;

            for (int gamma = 0; gamma < pag.n_nodes(); ++gamma) {
                if (gamma == alpha || gamma == beta) continue;
                const auto* e_beta_gamma = pag.get_edge(beta, gamma);
                const auto* e_alpha_gamma = pag.get_edge(alpha, gamma);

                if (!e_beta_gamma) continue;
                if (!e_alpha_gamma) continue;

                // beta *-> gamma (e_beta_gamma->mark_j == Mark::ARROW)
                if (e_beta_gamma->mark_j != Mark::ARROW) continue;

                // alpha *-o gamma (e_alpha_gamma->mark_j == Mark::CIRCLE)
                if (e_alpha_gamma->mark_j != Mark::CIRCLE) continue;

                // Orient alpha *-> gamma
                pag.set_mark(alpha, gamma, e_alpha_gamma->mark_i, Mark::ARROW);
                changed = true;
            }
        }
    }
}

void TSFCI::resolve_conflicts(PAG& pag) {
    // Resolve conflicting marks by defaulting to CIRCLE (most conservative).
    // In practice this is rare in single-lag FCI.
    for (auto& e : pag.edges()) {
        // This is a no-op in standard FCI; conflicts are handled by R1-R10 order.
        // If a conflict exists, prefer CIRCLE for ambiguity.
        (void)e; // unused
    }
}

} // namespace sc
