#pragma once
#include <vector>
#include <Eigen/Dense>
#include "../ingestion/market_event.hpp"

namespace sc {

// Fast-layer: per-event O(N) Hawkes intensity tracker.
// Uses alpha_hat and mu_hat frozen from the slow-layer EM tick (refreshed every 30s).
// Maintains live R_ij via the same O(1) recurrence as EMEstimator but
// updates on every incoming event rather than in batch.
//
// lambda_i(t) = mu_i + sum_j alpha_ij * beta * R_ij(t)
class IntensityTracker {
public:
    IntensityTracker(int n_instruments, double beta);

    // Refresh parameters from the slow layer after each EM tick.
    void set_params(const Eigen::MatrixXd& alpha, const Eigen::VectorXd& mu);

    // Update R for the instrument that just fired and recompute its lambda.
    // O(N) per call. Returns reference to the full lambda vector (all instruments).
    const Eigen::VectorXd& on_event(const MarketEvent& ev);

    const Eigen::VectorXd& lambda() const { return lambda_; }

private:
    int    n_;
    double beta_;

    Eigen::MatrixXd alpha_;  // frozen from last slow tick
    Eigen::VectorXd mu_;     // frozen from last slow tick
    Eigen::VectorXd lambda_; // current per-instrument intensity

    // live_R_[i][j] = current R_ij value (updated at each i-event via recurrence)
    std::vector<std::vector<double>> live_R_;
    // last_t_[i] = seconds timestamp of most recent event in instrument i (-1 = none)
    std::vector<double> last_t_;
    // pending_[i][j] = # j-events that have arrived since the last i-event
    std::vector<std::vector<int>> pending_;
};

}  // namespace sc
