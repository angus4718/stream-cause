#pragma once
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <Eigen/Dense>
#include "pag.hpp"
#include "ci_test.hpp"

namespace sc {

// Adjacency matrix type for the undirected skeleton (true = edge present).
using AdjMatrix = std::vector<std::vector<bool>>;

// Separation sets: sepset[{i,j}] = set of nodes S that d-separates i and j.
using SepSets = std::map<std::pair<int,int>, std::set<int>>;

// Adapted ts-FCI algorithm.
// Input: sequence of alpha_hat(tau_s) matrices (shape: S * N * N).
// Output: PAG G(tau_s) representing causal structure at current snapshot.
//
// Two-stage:
// Stage 1 -- Skeleton discovery (Alg. B.1): iterative CI testing with
// increasing conditioning set size d = 0.d_max.
// Stage 2 -- Edge orientation (Alg. B.2): V-structures, arrow-of-time
// FCI orientation rules R1-R10.
//
// Validate against causal-learn Python on same synthetic alpha_hat series.
class TSFCI {
public:
    TSFCI(int n_instruments, double alpha_ci = 0.01, int d_max = 3);

    // Run ts-FCI on the history of alpha_hat snapshots.
    // alpha_series: vector of S most recent N*N alpha_hat matrices (oldest first).
    // timestamp_ns: wall-clock time of current snapshot (tau_s).
    PAG run(const std::vector<Eigen::MatrixXd>& alpha_series
            int64_t timestamp_ns);

private:
    // Alg. B.1: Skeleton discovery.
    // Returns undirected skeleton and separation sets.
    std::pair<AdjMatrix, SepSets>
    skeleton_discovery(const std::vector<Eigen::MatrixXd>& alpha_series);

    // Extract the time-series of alpha_hat_{i,.} across all S snapshots -> VectorXd length S.
    Eigen::VectorXd extract_series(const std::vector<Eigen::MatrixXd>& alpha_series
                                   int row, int col) const;

    // Build conditioning variable matrix for node set S (columns are alpha_hat_{l,.} series).
    Eigen::MatrixXd build_cond_matrix(const std::vector<Eigen::MatrixXd>& alpha_series
                                      const std::set<int>& cond_set, int ref_row) const;

    // Alg. B.2: Orient edges using sep sets and FCI rules.
    PAG orient_edges(const AdjMatrix& skeleton
                     const SepSets& sep_sets
                     const std::vector<Eigen::MatrixXd>& alpha_series
                     int64_t timestamp_ns);

    // Arrow-of-time orientation: prefer i->j when alpha_hat_ji > alpha_hat_ij on average.
    void arrow_of_time_orient(PAG& pag
                              const std::vector<Eigen::MatrixXd>& alpha_series);

    // FCI orientation rules R1-R10 (Zhang 2008).
    void apply_fci_rules(PAG& pag);

    // Majority vote: for edges with conflicting orientations across lag levels
    // take the orientation supported by the majority of lags.
    void resolve_conflicts(PAG& pag);

    int n_;
    int d_max_;
    CITest ci_test_;
};

} // namespace sc
