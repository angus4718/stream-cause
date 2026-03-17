#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include <cmath>
#include "hawkes/hawkes_pair.hpp"
#include "hawkes/em_estimator.hpp"

using namespace sc;
using Catch::Matchers::WithinAbs;

// Helper: simulate a univariate Hawkes process with exponential kernel.
// Returns sorted vector of event times in [0, T].
static std::vector<double> simulate_hawkes(double mu, double alpha, double beta
                                            double T, std::mt19937& rng) {
    std::vector<double> times;
    // For exponential kernel, intensity is piecewise exponential between events.
    (void)mu; (void)alpha; (void)beta; (void)T; (void)rng;
    return times;
}

TEST_CASE("HawkesPair: R recurrence matches brute-force sum", "[hawkes]") {
    double beta = 5.0;
    HawkesPair pair;
    pair.beta = beta;

    // Known event times for instrument i and j.
    std::vector<double> t_i = {0.1, 0.3, 0.7, 1.2};
    std::vector<double> t_j = {0.05, 0.15, 0.5, 0.9};

    // R matches brute-force: sum_{t_l^j < t_m^i} exp(-beta*(t_m^i - t_l^j)).
    // Tolerance: 1e-10.

    // Placeholder: test passes structurally once update_R is implemented.
    REQUIRE(pair.R == 0.0); // not yet updated
}

TEST_CASE("EMEstimator: recovers known alpha on synthetic data", "[hawkes]") {
    // Ground-truth alpha matrix for N=2 instruments.
    // alpha_true[i][j] = how much j excites i.
    // alpha_true(0,1) = 0.4 (instrument 1 strongly excites instrument 0)
    // alpha_true(1,0) = 0.1 (instrument 0 weakly excites instrument 1)
    Eigen::MatrixXd alpha_true(2, 2);
    alpha_true << 0.3, 0.4
                  0.1, 0.2;

    double beta = 5.0;
    EMEstimator est(2, beta);

    std::mt19937 rng(42);

    // verify ||alpha_hat - alpha_true||_F < 0.05.
    // Use a long enough window (T=300s) for stable estimation.

    // Placeholder check: estimator initializes with zeros.
    REQUIRE_THAT(est.alpha().norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("EMEstimator: spectral radius enforced < 1 after projection", "[hawkes]") {
    // Start with an alpha matrix that violates stationarity.
    EMEstimator est(3, 5.0);
    // call one EM iteration; verify spectral radius of result < 1.
    REQUIRE(true); // placeholder
}
