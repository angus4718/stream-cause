#pragma once
#include <vector>
#include <deque>
#include <chrono>
#include <cstdint>
#include <Eigen/Dense>
#include "em_estimator.hpp"

namespace sc {

class EventRouter;

// Drives the update cycle: every update_interval seconds, run warm-started EM on
// the last window_seconds of events and return alpha_hat.
class RollingWindow {
public:
    RollingWindow(int n_instruments, double beta, int window_seconds, int update_interval_seconds, EventRouter& router, double alpha_reg = 0.0);

    Eigen::MatrixXd tick(int64_t now_ns, int max_iter = 50, double epsilon = 1e-4);

    const Eigen::MatrixXd& last_alpha() const { return last_alpha_; }
    const Eigen::VectorXd& last_mu() const { return estimator_.mu(); }
    int snapshot_count() const { return snapshot_count_; }

private:
    WindowEvents collect_and_convert(int64_t now_ns);

    int n_;
    int64_t window_ns_;
    int64_t update_interval_ns_;
    EventRouter& router_;
    EMEstimator estimator_;
    Eigen::MatrixXd last_alpha_;
    int snapshot_count_ = 0;
    std::deque<MarketEvent> event_store_;
};

}  // namespace sc
