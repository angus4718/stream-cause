#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <vector>
#include <Eigen/Dense>

namespace sc {

// Central configuration struct (loaded from config/default.json).
struct Config {
    int n_instruments = 59;
    int window_seconds = 600;
    int update_interval_seconds = 30;
    double beta = 10.0;
    int em_max_iter = 50;
    double em_epsilon = 1e-4;
    double fci_alpha_ci = 0.01;
    int fci_d_max = 3;
    double cusum_threshold = 5.0;
    double cusum_allowance_k = 0.5;
    double alpha_reg = 0.0;
    int ring_buffer_capacity = 1 << 20;
    int n_threads = 32;
    int alpha_series_history = 200; // max S for CI tests
    std::string output_dir = "output/snapshots";
    std::string macro_calendar = "data/calendars/macro_events.parquet";
    std::string log_level = "info";

    // Load from JSON file.
    static Config from_json_file(const std::string& path);
};

// Main pipeline orchestrating all five stages:
// 1. Databento ingestion -> EventRouter -> per-instrument ring buffers
// 2. RollingWindow tick -> alpha_hat(tau_s)
// 3. TSFCI -> PAG G(tau_s)
// 4. CUSUM per edge + BOCPD on g(tau_s)
// 5. GraphStore::append(pag, alpha)
class Pipeline {
public:
    explicit Pipeline(Config config);
    ~Pipeline();

    // Start live streaming pipeline. Blocks until shutdown().
    void run_live();

    // Run historical replay for a date range.
    void run_replay(const std::string& start_date, const std::string& end_date
                    double speed_multiplier = 1.0);

    // Signal graceful shutdown from signal handler or another thread.
    void shutdown();

    // Returns current per-instrument Hawkes intensity lambda (updated per event).
    Eigen::VectorXd query_intensity() const;

    // Performance counters (readable from monitoring thread).
    struct Metrics {
        std::atomic<int64_t> events_ingested{0};
        std::atomic<int64_t> em_updates{0};
        std::atomic<int64_t> graph_snapshots{0};
        std::atomic<int64_t> structural_breaks{0};
        std::atomic<double> last_em_latency_ms{0.0};
        std::atomic<double> last_fci_latency_ms{0.0};
    };
    const Metrics& metrics() const { return *metrics_; }

private:
    // Called every update_interval_seconds; runs one EM+FCI cycle.
    // Must not block the ingestion thread (uses double-buffered event snapshot).
    // 1. Snapshot alpha_hat = rolling_window_->tick(now_ns).
    // 2. Push alpha_hat to alpha_series_ (ring, max alpha_series_history_ entries).
    // 3. pag = tsfci_->run(alpha_series_, now_ns).
    // 4. g = pag.graph_edit_distance(prev_pag_).
    // 5. p_break = bocpd_->update(g, now_ns).
    // 6. For each edge in pag: cusum_[i][j].update(alpha_hat[i][j], now_ns).
    // 7. graph_store_->append(pag, alpha_hat).
    // 8. Update metrics_.
    void on_tick(int64_t now_ns);

    Config cfg_;
    std::atomic<bool> running_{false};

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<Metrics> metrics_;
};

} // namespace sc
