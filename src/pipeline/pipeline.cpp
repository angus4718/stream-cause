#include "pipeline.hpp"
#include "./ingestion/event_router.hpp"
#include "./ingestion/file_replay.hpp"
#include "./hawkes/rolling_window.hpp"
#include "./hawkes/intensity_tracker.hpp"
#include <fstream>
#include <filesystem>
#include <ctime>
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

namespace sc {

namespace {

std::string date_dir(const std::string& output_dir, int64_t timestamp_ns) {
    time_t t = timestamp_ns / 1'000'000'000LL;
    char buf[11];
    strftime(buf, sizeof(buf), "%Y-%m-%d", gmtime(&t));
    return output_dir + "/" + std::string(buf);
}

// Format the Python loaders expect: [int32 rows][int32 cols][float64 column-major].
void write_bin(const std::string& path, const double* data, int32_t rows, int32_t cols) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&rows), 4);
    f.write(reinterpret_cast<const char*>(&cols), 4);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(rows) * cols * sizeof(double));
}

}  // namespace

Config Config::from_json_file(const std::string& path) {
    std::ifstream f(path);
    nlohmann::json j = nlohmann::json::parse(f);

    Config cfg;
    cfg.n_instruments = j.value("n_instruments", 59);
    cfg.window_seconds = j.value("window_seconds", 600);
    cfg.update_interval_seconds = j.value("update_interval_seconds", 30);
    cfg.beta = j.value("beta", 10.0);
    cfg.em_max_iter = j.value("em_max_iter", 50);
    cfg.em_epsilon = j.value("em_epsilon", 1e-4);
    cfg.alpha_reg = j.value("alpha_reg", 0.0);
    cfg.output_dir = j.value("output_dir", "output/snapshots");
    cfg.log_level = j.value("log_level", "info");
    cfg.n_threads = j.value("n_threads", 32);
    return cfg;
}

struct Pipeline::Impl {
    std::unique_ptr<EventRouter> router;
    std::unique_ptr<RollingWindow> rolling_window;
    std::unique_ptr<IntensityTracker> tracker;
};

Pipeline::Pipeline(Config config)
    : cfg_(std::move(config)),
      impl_(std::make_unique<Impl>()),
      metrics_(std::make_unique<Metrics>()) {
    impl_->router = std::make_unique<EventRouter>(cfg_.n_instruments);
    impl_->rolling_window = std::make_unique<RollingWindow>(cfg_.n_instruments, cfg_.beta, cfg_.window_seconds, cfg_.update_interval_seconds, *impl_->router, cfg_.alpha_reg);
    impl_->tracker = std::make_unique<IntensityTracker>(cfg_.n_instruments, cfg_.beta);
    fs::create_directories(cfg_.output_dir);
}

Pipeline::~Pipeline() { shutdown(); }

void Pipeline::run_live() {
    running_.store(false);
}

void Pipeline::run_replay(const std::string& file_path, const std::string& /*unused_end*/, double /*unused_speed*/) {
    constexpr int64_t TICK_NS = 30LL * 1'000'000'000LL;
    running_.store(true);

    FileReplay fr;
    fr.replay(file_path, TICK_NS,
        [&](const MarketEvent& ev) {
            if (!running_) return;
            impl_->router->route(ev);
            impl_->tracker->on_event(ev);
            ++metrics_->events_ingested;
        },
        [&](int64_t ts_ns) {
            if (running_) on_tick(ts_ns);
        });

    running_.store(false);
}

void Pipeline::shutdown() {
    running_.store(false);
}

void Pipeline::on_tick(int64_t now_ns) {
    auto t0_em = std::chrono::steady_clock::now();
    Eigen::MatrixXd alpha = impl_->rolling_window->tick(now_ns, cfg_.em_max_iter, cfg_.em_epsilon);
    auto t1_em = std::chrono::steady_clock::now();
    metrics_->last_em_latency_ms = std::chrono::duration<double, std::milli>(t1_em - t0_em).count();
    ++metrics_->em_updates;

    impl_->tracker->set_params(alpha, impl_->rolling_window->last_mu());

    std::string dir = date_dir(cfg_.output_dir, now_ns);
    fs::create_directories(dir);
    write_bin(dir + "/alpha_" + std::to_string(now_ns) + ".bin", alpha.data(), static_cast<int32_t>(alpha.rows()), static_cast<int32_t>(alpha.cols()));
    const Eigen::VectorXd& lam = impl_->tracker->lambda();
    write_bin(dir + "/lambda_" + std::to_string(now_ns) + ".bin", lam.data(), static_cast<int32_t>(lam.size()), 1);
    ++metrics_->snapshots;

    if (metrics_->snapshots % 100 == 0) {
        fprintf(stderr, "[FastLayer] tick %ld lambda_AAPL=%.4f lambda_NVDA=%.4f lambda_TSLA=%.4f\n", (long)metrics_->snapshots.load(), lam.size() > 0 ? lam(0) : 0.0, lam.size() > 11 ? lam(11) : 0.0, lam.size() > 15 ? lam(15) : 0.0);
    }
}

Eigen::VectorXd Pipeline::query_intensity() const {
    return impl_->tracker->lambda();
}

}  // namespace sc
