#include "pipeline.hpp"
// #include "./ingestion/databento_client.hpp"
#include "./ingestion/event_router.hpp"
#include "./ingestion/file_replay.hpp"
#include "./hawkes/rolling_window.hpp"
#include "./hawkes/intensity_tracker.hpp"
#include "./causal/ts_fci.hpp"
#include "./causal/graph_store.hpp"
#include "./changepoint/cusum.hpp"
#include "./changepoint/bocpd.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>

namespace sc {

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
    cfg.fci_alpha_ci = j.value("fci_alpha_ci", 0.01);
    cfg.fci_d_max = j.value("fci_d_max", 3);
    cfg.cusum_threshold = j.value("cusum_threshold", 5.0);
    cfg.cusum_allowance_k = j.value("cusum_allowance_k", 0.5);
    cfg.alpha_reg = j.value("alpha_reg", 0.0);
    cfg.output_dir = j.value("output_dir", "output/snapshots");
    cfg.macro_calendar = j.value("macro_calendar", "");
    cfg.log_level = j.value("log_level", "info");
    cfg.n_threads = j.value("n_threads", 32);
    cfg.alpha_series_history = j.value("alpha_series_history", 200);
    return cfg;
}

struct Pipeline::Impl {
    // std::unique_ptr<DatabentoClient> databento;
    std::unique_ptr<EventRouter> router;
    std::unique_ptr<RollingWindow> rolling_window;
    std::unique_ptr<TSFCI> tsfci;
    std::unique_ptr<GraphStore> graph_store;
    std::unique_ptr<BOCPD> bocpd;
    std::unique_ptr<IntensityTracker> tracker;
    // CUSUM per (i,j) pair.
    std::vector<std::vector<CUSUM>> cusum;
    // In-memory ring of last alpha_series_history alpha_hat matrices.
    std::vector<Eigen::MatrixXd> alpha_series;
    PAG prev_pag;
    int alpha_series_history = 200;
};

Pipeline::Pipeline(Config config)
    : cfg_(std::move(config))
      impl_(std::make_unique<Impl>())
      metrics_(std::make_unique<Metrics>()) {
    // Load macro calendar (empty for now).
    std::vector<MacroCalendarEntry> calendar;

    // Construct sub-components.
    // impl_->databento = std::make_unique<DatabentoClient>(cfg_);
    impl_->router = std::make_unique<EventRouter>(cfg_.n_instruments, calendar, 30);
    impl_->rolling_window = std::make_unique<RollingWindow>(
        cfg_.n_instruments, cfg_.beta, cfg_.window_seconds
        cfg_.update_interval_seconds, *impl_->router, cfg_.alpha_reg);
    impl_->tsfci = std::make_unique<TSFCI>(cfg_.n_instruments
                                            cfg_.fci_alpha_ci, cfg_.fci_d_max);
    impl_->graph_store = std::make_unique<GraphStore>(cfg_.output_dir);
    impl_->bocpd = std::make_unique<BOCPD>();
    impl_->tracker = std::make_unique<IntensityTracker>(cfg_.n_instruments, cfg_.beta);

    // CUSUM array: one per (i,j) pair
    impl_->cusum.resize(cfg_.n_instruments);
    for (int i = 0; i < cfg_.n_instruments; ++i) {
        impl_->cusum[i].resize(cfg_.n_instruments);
        for (int j = 0; j < cfg_.n_instruments; ++j) {
            impl_->cusum[i][j] = CUSUM(cfg_.cusum_threshold, cfg_.cusum_allowance_k);
        }
    }

    impl_->prev_pag = PAG(cfg_.n_instruments, 0);
    impl_->alpha_series_history = cfg_.alpha_series_history;
}

Pipeline::~Pipeline() { shutdown(); }

void Pipeline::run_live() {
    // Requires compiled Databento library binaries
    running_.store(false);
}

void Pipeline::run_replay(const std::string& file_path
                          const std::string& /*unused_end*/
                          double /*unused_speed*/) {
    constexpr int64_t TICK_NS = 30LL * 1'000'000'000LL;
    running_.store(true);

    FileReplay fr;
    fr.replay(
        file_path
        TICK_NS
        [&](const MarketEvent& ev) {
            if (!running_) return;
            impl_->router->route(ev);
            impl_->tracker->on_event(ev);
            ++metrics_->events_ingested;
        }
        [&](int64_t ts_ns) {
            if (running_) on_tick(ts_ns);
        });

    running_.store(false);
}

void Pipeline::shutdown() {
    running_.store(false);
}

void Pipeline::on_tick(int64_t now_ns) {
    // Step 1: Run EM on the rolling window.
    auto t0_em = std::chrono::steady_clock::now();
    Eigen::MatrixXd alpha = impl_->rolling_window->tick(now_ns
        cfg_.em_max_iter, cfg_.em_epsilon);
    auto t1_em = std::chrono::steady_clock::now();
    metrics_->last_em_latency_ms =
        std::chrono::duration<double, std::milli>(t1_em - t0_em).count();

    // Refresh fast-layer tracker with latest parameters.
    impl_->tracker->set_params(alpha, impl_->rolling_window->last_mu());

    // Step 2: Accumulate alpha_hat for ts-FCI.
    impl_->alpha_series.push_back(alpha);
    if (static_cast<int>(impl_->alpha_series.size()) > impl_->alpha_series_history)
        impl_->alpha_series.erase(impl_->alpha_series.begin());

    // Step 3: Run ts-FCI if enough snapshots.
    auto t0_fci = std::chrono::steady_clock::now();
    PAG pag = impl_->tsfci->run(impl_->alpha_series, now_ns);
    auto t1_fci = std::chrono::steady_clock::now();
    metrics_->last_fci_latency_ms =
        std::chrono::duration<double, std::milli>(t1_fci - t0_fci).count();

    // Step 4: Detect structural breaks using BOCPD on graph edit distance.
    double g = pag.graph_edit_distance(impl_->prev_pag);
    double p_break = impl_->bocpd->update(g, now_ns);
    (void)p_break; // Could log or use for alarms

    // Step 5: Run CUSUM on each alpha_hat_{ij}.
    for (int i = 0; i < cfg_.n_instruments; ++i) {
        for (int j = 0; j < cfg_.n_instruments; ++j) {
            if (impl_->cusum[i][j].update(alpha(i, j), now_ns))
                ++metrics_->structural_breaks;
        }
    }

    // Step 6: Persist PAG, alpha_hat, and lambda.
    impl_->graph_store->append(pag, alpha);
    impl_->graph_store->append_lambda(now_ns, impl_->tracker->lambda());
    impl_->prev_pag = pag;
    ++metrics_->graph_snapshots;

    // Log fast-layer lambda every 100 slow ticks for verification.
    if (metrics_->graph_snapshots % 100 == 0) {
        const auto& lam = impl_->tracker->lambda();
        fprintf(stderr, "[FastLayer] tick %ld lambda_AAPL=%.4f lambda_NVDA=%.4f lambda_TSLA=%.4f\n"
                (long)metrics_->graph_snapshots.load()
                lam.size() > 0 ? lam(0) : 0.0
                lam.size() > 11 ? lam(11) : 0.0
                lam.size() > 15 ? lam(15) : 0.0);
    }
}

Eigen::VectorXd Pipeline::query_intensity() const {
    return impl_->tracker->lambda();
}

} // namespace sc
