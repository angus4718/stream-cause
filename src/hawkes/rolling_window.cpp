#include "rolling_window.hpp"
#include "./ingestion/event_router.hpp"
#include <algorithm>

namespace sc {

static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;

RollingWindow::RollingWindow(int n_instruments, double beta
                             int window_seconds, int update_interval_seconds
                             EventRouter& router, double alpha_reg)
    : n_(n_instruments)
      window_ns_(static_cast<int64_t>(window_seconds) * NS_PER_SEC)
      update_interval_ns_(static_cast<int64_t>(update_interval_seconds) * NS_PER_SEC)
      router_(router)
      estimator_(n_instruments, beta, alpha_reg)
      last_alpha_(Eigen::MatrixXd::Zero(n_instruments, n_instruments)) {}

Eigen::MatrixXd RollingWindow::tick(int64_t now_ns, int max_iter, double epsilon) {
    WindowEvents events = collect_and_convert(now_ns);
    estimator_.warm_start(last_alpha_);
    last_alpha_ = estimator_.run(events, max_iter, epsilon);
    ++snapshot_count_;
    // spdlog::info("snapshot {} ||alpha||_F={:.4f}", snapshot_count_, last_alpha_.norm());
    return last_alpha_;
}

WindowEvents RollingWindow::collect_and_convert(int64_t now_ns) {
    int64_t start_ns = now_ns - window_ns_;

    // Drain all newly arrived events from the ring buffers into event_store_.
    std::vector<MarketEvent> fresh = router_.drain_window(0, now_ns);
    for (auto& ev : fresh)
        event_store_.push_back(ev);

    // Evict events that have fallen outside the rolling window.
    while (!event_store_.empty() && event_store_.front().ts_event < start_ns)
        event_store_.pop_front();

    // Build WindowEvents from everything remaining in event_store_.
    WindowEvents events(n_);
    for (int i = 0; i < n_; ++i)
        events[i].instrument_id = i;

    for (const auto& ev : event_store_) {
        double t_sec = static_cast<double>(ev.ts_event - start_ns) / 1e9;
        events[ev.instrument_id].times.push_back(t_sec);
    }

    for (auto& ie : events)
        std::sort(ie.times.begin(), ie.times.end());

    for (auto& ie : events)
        ie.times.erase(std::unique(ie.times.begin(), ie.times.end()), ie.times.end());

    return events;
}

} // namespace sc
