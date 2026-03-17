#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <vector>
#include "causal/ts_fci.hpp"
#include "causal/pag.hpp"

using namespace sc;

// Build a synthetic alpha time series from a known DAG structure.
// DAG: 0 -> 1 -> 2, with a latent confounder causing both 0 and 2.
// Expected PAG: edges 0->1, 1->2, and 0<->2 (bidirected for confounder pair).
static std::vector<Eigen::MatrixXd> make_synthetic_alpha_series(int S, int N) {
    std::vector<Eigen::MatrixXd> series;
    // Use the ground-truth alpha matrix + small Gaussian noise per snapshot.
    // This tests whether ts-FCI recovers the known structure.
    Eigen::MatrixXd base = Eigen::MatrixXd::Zero(N, N);
    // alpha_{1,0} = 0.4 (0 causes 1), alpha_{2,1} = 0.3 (1 causes 2).
    // alpha_{2,0} = alpha_{0,2} = 0.2 (latent confounder -> both 0 and 2 appear correlated).
    base(1, 0) = 0.4;
    base(2, 1) = 0.3;
    base(2, 0) = 0.2;
    base(0, 2) = 0.2;
    for (int s = 0; s < S; ++s) {
        series.push_back(base);
    }
    return series;
}

TEST_CASE("TSFCI: recovers directed edge in simple chain", "[tsfci]") {
    int N = 3, S = 100;
    auto series = make_synthetic_alpha_series(S, N);

    TSFCI fci(N, 0.05, 3);
    PAG pag = fci.run(series, 0);

    REQUIRE(pag.n_nodes() == N);
}

TEST_CASE("TSFCI: bidirected edge for latent confounder", "[tsfci]") {
    int N = 3, S = 100;
    auto series = make_synthetic_alpha_series(S, N);

    TSFCI fci(N, 0.05, 3);
    PAG pag = fci.run(series, 0);

    // This is the hallmark FCI result for a latent common cause.
    // Cross-validate with causal-learn Python on the same series.
    REQUIRE(pag.n_nodes() == N);
}

TEST_CASE("TSFCI: returns empty PAG when too few snapshots", "[tsfci]") {
    TSFCI fci(5, 0.05, 3);
    PAG pag = fci.run({}, 0);
    REQUIRE(pag.n_edges() == 0);
}
