#pragma once
#include <vector>
#include <Eigen/Dense>
#include "../ingestion/market_event.hpp"

namespace sc {

// Fast-layer per-event Hawkes intensity tracker. alpha/mu are frozen from the
// slow EM tick; R_ij is updated live via the same O(1) recurrence as EMEstimator.
//   lambda_i(t) = mu_i + sum_j alpha_ij * beta * R_ij(t)
class IntensityTracker {
public:
    IntensityTracker(int n_instruments, double beta);

    void set_params(const Eigen::MatrixXd& alpha, const Eigen::VectorXd& mu);
    // Update R for the firing instrument and recompute its lambda; O(N) per call.
    const Eigen::VectorXd& on_event(const MarketEvent& ev);
    const Eigen::VectorXd& lambda() const { return lambda_; }

private:
    int n_;
    double beta_;
    Eigen::MatrixXd alpha_;
    Eigen::VectorXd mu_;
    Eigen::VectorXd lambda_;
    std::vector<std::vector<double>> live_R_;
    std::vector<double> last_t_;
    std::vector<std::vector<int>> pending_;  // j-events seen since instrument i last fired
};

}  // namespace sc
