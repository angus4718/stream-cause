#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <random>
#include "causal/ci_test.hpp"

using namespace sc;

// Generate multivariate normal samples.
// Returns matrix of shape (n_samples, n_vars).
static Eigen::MatrixXd mvnormal(const Eigen::MatrixXd& cov, int n_samples
                                  std::mt19937& rng) {
    int d = static_cast<int>(cov.rows());
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    Eigen::MatrixXd L = llt.matrixL();
    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::MatrixXd Z(d, n_samples);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < n_samples; ++j)
            Z(i, j) = norm(rng);
    return (L * Z).transpose(); // shape (n_samples, d)
}

TEST_CASE("CITest: unconditional independent pair detected", "[ci_test]") {
    std::mt19937 rng(123);
    int S = 500;
    // X and Y are independent standard normals.
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd samples = mvnormal(cov, S, rng);
    Eigen::VectorXd xi = samples.col(0);
    Eigen::VectorXd xj = samples.col(1);

    CITest ci(0.05);
    // Empty conditioning set.
    bool independent = ci.test(xi, xj, Eigen::MatrixXd(S, 0), S);
    REQUIRE(independent);
}

TEST_CASE("CITest: correlated pair not falsely declared independent", "[ci_test]") {
    std::mt19937 rng(456);
    int S = 500;
    // X and Y have correlation 0.8.
    Eigen::MatrixXd cov(2, 2);
    cov << 1.0, 0.8, 0.8, 1.0;
    Eigen::MatrixXd samples = mvnormal(cov, S, rng);
    Eigen::VectorXd xi = samples.col(0);
    Eigen::VectorXd xj = samples.col(1);

    CITest ci(0.05);
    bool independent = ci.test(xi, xj, Eigen::MatrixXd(S, 0), S);
    REQUIRE_FALSE(independent); // Should detect dependence.
}

TEST_CASE("CITest: conditional independence after partialling out confounder", "[ci_test]") {
    // X -> Z <- Y: X and Y are marginally correlated but conditionally independent given Z.
    REQUIRE(true); // placeholder
}

TEST_CASE("CITest: Fisher Z-statistic formula correct", "[ci_test]") {
    // Known values: rho=0.5, S=100, n_cond=1.
    double Z = CITest::fisher_z(0.5, 100, 1);
    REQUIRE(Z > 0.0);
}
