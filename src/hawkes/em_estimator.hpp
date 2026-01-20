#pragma once
#include <vector>
#include <Eigen/Dense>
#include "hawkes_pair.hpp"
#include "./ingestion/market_event.hpp"

namespace sc {

// Events from a single instrument over a rolling window.
struct InstrumentEvents {
    int instrument_id;
    std::vector<double> times; // event times in seconds since window start
};

// Events across all instruments for one rolling window.
using WindowEvents = std::vector<InstrumentEvents>;

// Runs EM for the multivariate Hawkes process with exponential kernel.
// Parameters: mu in R^N (baseline intensities), alpha in R^{N*N} (branching ratios)
// beta in R (shared exponential decay rate, fixed across pairs).
//
// Paper references:
// Model:Log-likelihood:
// E-step:M-step:
// Sufficient statistic R_ij:
class EMEstimator {
public:
    EMEstimator(int n_instruments, double beta, double alpha_reg = 0.0);

    // Run warm-started EM on window_events.
    // Returns alpha_hat (N*N branching ratio matrix).
    // Convergence: stop when ||alpha^{k+1} - alpha^k||_F < epsilon or k >= max_iter.
    //
    // 1. Call reset_pairs() to clear per-window state.
    // 2. Pre-compute R_ij for all events using HawkesPair::update_R().
    // 3. Iterate E-step then M-step up to max_iter times.
    // 4. Check Frobenius norm convergence.
    // 5. Project alpha to enforce spectral_radius(alpha) < 1 (see project_alpha()).
    Eigen::MatrixXd run(const WindowEvents& events
                        int max_iter = 50
                        double epsilon = 1e-4);

    // Compute log-likelihood on window_events with current params.
    // Used for convergence monitoring (not for stopping criterion -- use Frobenius).
    double log_likelihood(const WindowEvents& events) const;

    // Warm-start: initialize alpha from previous window's estimate.
    void warm_start(const Eigen::MatrixXd& prev_alpha);

    const Eigen::MatrixXd& alpha() const { return alpha_; }
    const Eigen::VectorXd& mu() const { return mu_; }

private:
    // Pre-compute R_ij for all events using O(1) recurrence.
    // Must be called once per window before E-step.
    void precompute_R(const WindowEvents& events);

    // E-step: compute posterior responsibilities for each event.
    // Updates e_step_numerator_accum in each HawkesPair.
    void e_step(const WindowEvents& events);

    // M-step: closed-form parameter updates.
    void m_step(const WindowEvents& events, double T);

    // Project alpha so spectral_radius(alpha) < 1 (stationarity condition.
    // Simple approach: if rho(alpha) >= 1, scale alpha *= 0.99 / rho(alpha).
    void project_alpha();

    void reset_pairs();

    int n_;
    double beta_;
    double alpha_reg_; // L2 ridge floor added to off-diagonal after each M-step
    Eigen::MatrixXd alpha_; // N*N, warm-started
    Eigen::VectorXd mu_; // N
    // pairs_[i][j] holds state for the (i,j) pair.
    std::vector<std::vector<HawkesPair>> pairs_;
};

} // namespace sc
