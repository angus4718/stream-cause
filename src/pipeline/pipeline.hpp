#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <vector>
#include <Eigen/Dense>

namespace sc {

struct Config {
    int n_instruments = 59;
    int window_seconds = 600;
    int update_interval_seconds = 30;
    double beta = 10.0;
    int em_max_iter = 50;
    double em_epsilon = 1e-4;
    double alpha_reg = 0.0;
    int ring_buffer_capacity = 1 << 20;
    int n_threads = 32;
    std::string output_dir = "output/snapshots";
    std::string log_level = "info";

    static Config from_json_file(const std::string& path);
};

// Ingest -> RollingWindow EM (alpha_hat) -> IntensityTracker (lambda), persisting
// alpha_hat and lambda per tick for offline analysis and backtests.
class Pipeline {
public:
    explicit Pipeline(Config config);
    ~Pipeline();

    void run_live();
    void run_replay(const std::string& start_date, const std::string& end_date, double speed_multiplier = 1.0);
    void shutdown();
    Eigen::VectorXd query_intensity() const;

    struct Metrics {
        std::atomic<int64_t> events_ingested{0};
        std::atomic<int64_t> em_updates{0};
        std::atomic<int64_t> snapshots{0};
        std::atomic<double> last_em_latency_ms{0.0};
    };
    const Metrics& metrics() const { return *metrics_; }

private:
    void on_tick(int64_t now_ns);

    Config cfg_;
    std::atomic<bool> running_{false};
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<Metrics> metrics_;
};

}  // namespace sc
