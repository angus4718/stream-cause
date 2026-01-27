#pragma once
#include <vector>
#include <deque>
#include <chrono>
#include <cstdint>
#include <Eigen/Dense>
#include "em_estimator.hpp"

namespace sc {

class EventRouter;

// Orchestrates the 30-second update cycle.
// Drains ring buffers every Delta=30s, runs warm-started EM on the last w=10min
// of events, and returns the updated alpha_hat(tau_s) matrix.
class RollingWindow {
public:
    // window_seconds: w = 600 (10 min), update_interval_seconds: Delta = 30.
    RollingWindow(int n_instruments, double beta
                  int window_seconds, int update_interval_seconds
                  EventRouter& router, double alpha_reg = 0.0);

    // Called on each 30-second tick (tau_s = s * Delta).
    // Returns alpha_hat(tau_s) as an N*N matrix.
    //
    // 1. Call router_.drain_window(now_ns - window_ns, now_ns) to collect events.
    // 2. Convert nanosecond timestamps to seconds, zero-referenced to window start.
    // 3. Call estimator_.run(events, max_iter, epsilon).
    // 4. Warm-start estimator_ from returned alpha for next tick.
    // 5. Log alpha_hat with timestamp tau_s via spdlog.
    // 6. Return alpha_hat.
    Eigen::MatrixXd tick(int64_t now_ns, int max_iter = 50, double epsilon = 1e-4);

    // Access the most recently estimated alpha_hat and mu_hat.
    const Eigen::MatrixXd& last_alpha() const { return last_alpha_; }
    const Eigen::VectorXd& last_mu() const { return estimator_.mu(); }

    // Number of alpha_hat snapshots produced so far.
    int snapshot_count() const { return snapshot_count_; }

private:
    // before the old window_start should be excluded from the new window).
    WindowEvents collect_and_convert(int64_t now_ns);

    int n_;
    int64_t window_ns_; // w in nanoseconds
    int64_t update_interval_ns_; // Delta in nanoseconds
    EventRouter& router_;
    EMEstimator estimator_;
    Eigen::MatrixXd last_alpha_;
    int snapshot_count_ = 0;
    std::deque<MarketEvent> event_store_; // rolling buffer of last window_ns_ seconds
};

} // namespace sc
