#pragma once
#include <vector>
#include <Eigen/Dense>

namespace sc {

// Conditional independence (CI) test on alpha_hat time series using Fisher's Z-transform.
// Paper,:
// Z_{ij|S} = sqrt(S - |S| - 3) / 2 * log((1 + rho_hat_{ij*S}) / (1 - rho_hat_{ij*S}))
// where rho_hat_{ij*S} is the sample partial correlation between the alpha_hat_i. and alpha_hat_j.
// time series, conditioning on {alpha_hat_l.}_{linS}.
//
// Returns true if the null hypothesis of conditional independence is NOT rejected
// (i.e., p-value > alpha_threshold) -> the edge (i,j) should be removed.
class CITest {
public:
    explicit CITest(double alpha_threshold = 0.01);

    // Test alpha_hat_i. _||_ alpha_hat_j. | {alpha_hat_l.}_{lincond_set}.
    // xi: time series of row i of alpha_hat (length S * N, but we use the column vector
    // of alpha_hat_{i,.} across S snapshots -- length S).
    // xj: same for row j.
    // cond_vars: matrix of shape (S, |cond_set|); each column is one conditioning variable.
    // n_obs: S = number of alpha_hat snapshots available.
    //
    // Returns true if conditionally independent (p-value > alpha_threshold).
    //
    // validate against scipy.stats.partial_corr on same data.
    bool test(const Eigen::VectorXd& xi
              const Eigen::VectorXd& xj
              const Eigen::MatrixXd& cond_vars
              int n_obs) const;

    // Compute sample partial correlation rho_hat_{ij*S}.
    // Returns value in (-1, 1); NaN on degenerate inputs.
    //
    // Add small ridge (1e-6 * I) to handle near-singular covariance.
    double partial_correlation(const Eigen::VectorXd& xi
                               const Eigen::VectorXd& xj
                               const Eigen::MatrixXd& cond_vars) const;

    // Fisher Z-statistic from sample partial correlation rho_hat and sample size S.
    // n_cond = |S| (size of conditioning set).
    static double fisher_z(double rho_hat, int S, int n_cond);

    // Two-sided p-value from Z statistic (normal approximation).
    static double z_to_pvalue(double Z);

    double alpha_threshold() const { return alpha_threshold_; }

private:
    double alpha_threshold_;
};

} // namespace sc
