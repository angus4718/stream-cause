#pragma once
#include <vector>
#include <Eigen/Dense>
#include "hawkes_pair.hpp"
#include "./ingestion/market_event.hpp"

namespace sc {

struct InstrumentEvents {
    int instrument_id;
    std::vector<double> times;  // seconds since window start
};

using WindowEvents = std::vector<InstrumentEvents>;

// EM for the multivariate Hawkes process with a shared exponential kernel:
// baseline mu in R^N, branching ratios alpha in R^{NxN}, fixed decay beta.
class EMEstimator {
public:
    EMEstimator(int n_instruments, double beta, double alpha_reg = 0.0);

    // Warm-started EM; returns alpha_hat. Stops at ||dalpha||_F < epsilon or max_iter.
    Eigen::MatrixXd run(const WindowEvents& events, int max_iter = 50, double epsilon = 1e-4);

    double log_likelihood(const WindowEvents& events) const;
    void warm_start(const Eigen::MatrixXd& prev_alpha);

    const Eigen::MatrixXd& alpha() const { return alpha_; }
    const Eigen::VectorXd& mu() const { return mu_; }

private:
    void precompute_R(const WindowEvents& events);
    void e_step(const WindowEvents& events);
    void m_step(const WindowEvents& events, double T);
    // Scale alpha so spectral_radius(alpha) < 1 (stationarity).
    void project_alpha();
    void reset_pairs();

    int n_;
    double beta_;
    double alpha_reg_;
    Eigen::MatrixXd alpha_;
    Eigen::VectorXd mu_;
    std::vector<std::vector<HawkesPair>> pairs_;
};

}  // namespace sc
