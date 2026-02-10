#include "ci_test.hpp"
#include <cmath>
#include <stdexcept>

namespace sc {

CITest::CITest(double alpha_threshold) : alpha_threshold_(alpha_threshold) {}

bool CITest::test(const Eigen::VectorXd& xi
                  const Eigen::VectorXd& xj
                  const Eigen::MatrixXd& cond_vars
                  int n_obs) const {
    double rho;

    if (cond_vars.cols() == 0) {
        // Unconditional: Pearson correlation
        Eigen::VectorXd xi_c = xi.array() - xi.mean();
        Eigen::VectorXd xj_c = xj.array() - xj.mean();
        double denom = xi_c.norm() * xj_c.norm();
        if (denom < 1e-300)
            return true; // Degenerate case -- consider independent
        rho = xi_c.dot(xj_c) / denom;
        rho = std::max(-0.999999, std::min(0.999999, rho));
    } else {
        // Conditional: use partial correlation
        rho = partial_correlation(xi, xj, cond_vars);
    }

    // Fisher Z transform and p-value
    double Z = fisher_z(rho, n_obs, cond_vars.cols());
    double p = z_to_pvalue(Z);

    // Return true if p > alpha_threshold (fail to reject independence)
    return p > alpha_threshold_;
}

double CITest::partial_correlation(const Eigen::VectorXd& xi
                                   const Eigen::VectorXd& xj
                                   const Eigen::MatrixXd& cond_vars) const {
    int n = xi.size();
    int d = 2 + cond_vars.cols();

    // Build centered feature matrix: columns [xi_centered, xj_centered, cond_vars_centered]
    Eigen::MatrixXd X(n, d);
    X.col(0) = xi.array() - xi.mean();
    X.col(1) = xj.array() - xj.mean();
    for (int k = 0; k < cond_vars.cols(); ++k)
        X.col(2 + k) = cond_vars.col(k).array() - cond_vars.col(k).mean();

    // Sample covariance sum = X'X / (n-1)
    Eigen::MatrixXd Sigma = (X.transpose() * X) / static_cast<double>(n - 1);

    // Add ridge regularization for numerical stability
    Sigma += 1e-6 * Eigen::MatrixXd::Identity(d, d);

    // Precision matrix Theta = sum^{-1} via Cholesky LLT decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    if (llt.info() != Eigen::Success)
        return 0.0; // Singular matrix
    Eigen::MatrixXd Theta = llt.solve(Eigen::MatrixXd::Identity(d, d));

    // Partial correlation: rho_hat_{ij*S} = -Theta(0,1) / sqrt(Theta(0,0) * Theta(1,1))
    double denom = std::sqrt(Theta(0, 0) * Theta(1, 1));
    if (denom < 1e-300)
        return 0.0;
    double rho = -Theta(0, 1) / denom;

    // Clamp to valid correlation range
    return std::max(-0.999999, std::min(0.999999, rho));
}

double CITest::fisher_z(double rho_hat, int S, int n_cond) {
    // Z_{ij|S} = sqrt(S - |S| - 3) / 2 * log((1 + rho_hat) / (1 - rho_hat))
    double eff_n = std::max(S - n_cond - 3, 1);
    double z = 0.5 * std::log((1.0 + rho_hat) / (1.0 - rho_hat));
    return std::sqrt(static_cast<double>(eff_n)) * z;
}

double CITest::z_to_pvalue(double Z) {
    // Two-sided p-value: p = 2 * Phi(-|Z|) where Phi is the standard normal CDF.
    // Use std::erfc for numerical precision.
    return std::erfc(std::abs(Z) / std::sqrt(2.0));
}

} // namespace sc
