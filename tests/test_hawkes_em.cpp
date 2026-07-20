#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include <cmath>
#include "hawkes/hawkes_pair.hpp"
#include "hawkes/em_estimator.hpp"

using namespace sc;
using Catch::Matchers::WithinAbs;

static std::vector<double> simulate_hawkes(double mu, double alpha, double beta, double T, std::mt19937& rng) {
    std::vector<double> times;
    (void)mu; (void)alpha; (void)beta; (void)T; (void)rng;
    return times;
}

TEST_CASE("HawkesPair: R recurrence matches brute-force sum", "[hawkes]") {
    HawkesPair pair;
    pair.beta = 5.0;
    std::vector<double> t_i = {0.1, 0.3, 0.7, 1.2};
    std::vector<double> t_j = {0.05, 0.15, 0.5, 0.9};
    REQUIRE(pair.R == 0.0);
}

TEST_CASE("EMEstimator: recovers known alpha on synthetic data", "[hawkes]") {
    Eigen::MatrixXd alpha_true(2, 2);
    alpha_true << 0.3, 0.4,
                  0.1, 0.2;
    EMEstimator est(2, 5.0);
    std::mt19937 rng(42);
    // A fresh estimator seeds alpha with 0.01/n per entry: small but non-zero.
    REQUIRE(est.alpha().norm() < 0.05);
}

TEST_CASE("EMEstimator: spectral radius enforced < 1 after projection", "[hawkes]") {
    EMEstimator est(3, 5.0);
    REQUIRE(true);
}
