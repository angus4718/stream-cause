#include "databento_client.hpp"
#include "market_event.hpp"
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <cmath>

namespace sc {

static Action translate_action(databento::Action a) {
    switch (a) {
        case databento::Action::Add: return Action::ADD;
        case databento::Action::Cancel: return Action::CANCEL;
        case databento::Action::Trade: return Action::TRADE;
        case databento::Action::Modify: return Action::MODIFY;
        default: return Action::ADD;
    }
}

// US equity session (ET 09:30-16:00), fixed EST offset, no DST handling.
static bool in_session(int64_t ts_ns) {
    constexpr int64_t ET_OFFSET_NS = -5LL * 3600LL * 1'000'000'000LL;
    int64_t et_ns = ts_ns + ET_OFFSET_NS;
    int64_t ns_of_day = et_ns % (86400LL * 1'000'000'000LL);
    int64_t open_ns = (9LL * 3600 + 30 * 60) * 1'000'000'000LL;
    int64_t close_ns = 16LL * 3600LL * 1'000'000'000LL;
    return ns_of_day >= open_ns && ns_of_day < close_ns;
}

DatabentoClient::DatabentoClient(const Config& /*cfg*/) : impl_(std::make_unique<Impl>()) {
    const char* key = std::getenv("DATABENTO_API_KEY");
    if (!key) {
        throw std::runtime_error("DATABENTO_API_KEY environment variable not set");
    }
    impl_->api_key = key;
}

DatabentoClient::~DatabentoClient() {
    stop();
}

void DatabentoClient::subscribe(const std::string& dataset, const std::string& /*schema*/, const std::vector<std::string>& symbols) {
    if (symbols.empty()) {
        impl_->replay_file_path = dataset;
    }
}

void DatabentoClient::start(EventCallback /*callback*/) {
}

void DatabentoClient::replay(const std::string& /*start_date*/, const std::string& /*end_date*/, EventCallback callback, double speed_multiplier) {
    const std::string& file_path = impl_->replay_file_path;
    if (file_path.empty()) {
        throw std::runtime_error("No file path set; call subscribe() with file path first");
    }

    databento::TsSymbolMap symbol_map;
    auto metadata_callback = [&symbol_map](databento::Metadata metadata) {
        symbol_map = metadata.CreateSymbolMap();
    };

    int64_t first_ts = -1;
    auto wall_start = std::chrono::steady_clock::now();

    auto record_callback = [&](const databento::Record& record) -> databento::KeepGoing {
        if (!impl_->running) return databento::KeepGoing::Stop;

        const auto* mbo = record.GetIf<databento::MboMsg>();
        if (!mbo) return databento::KeepGoing::Continue;

        if (mbo->action != databento::Action::Add && mbo->action != databento::Action::Cancel && mbo->action != databento::Action::Trade) {
            return databento::KeepGoing::Continue;
        }

        int64_t ts_event_ns = static_cast<int64_t>(mbo->hd.ts_event.time_since_epoch().count());
        if (!in_session(ts_event_ns)) {
            return databento::KeepGoing::Continue;
        }

        const std::string& sym = symbol_map.At(*mbo);
        auto it = SYMBOL_TO_ID.find(sym);
        if (it == SYMBOL_TO_ID.end()) {
            return databento::KeepGoing::Continue;
        }

        MarketEvent ev;
        ev.ts_event = ts_event_ns;
        ev.ts_recv = static_cast<int64_t>(mbo->ts_recv.time_since_epoch().count());
        ev.instrument_id = it->second;
        ev.action = translate_action(mbo->action);
        ev.price = mbo->price;
        ev.size = mbo->size;

        if (first_ts < 0) {
            first_ts = ts_event_ns;
        } else if (speed_multiplier > 0.0) {
            int64_t rel_ns = ts_event_ns - first_ts;
            auto target = wall_start + std::chrono::nanoseconds(static_cast<int64_t>(rel_ns / speed_multiplier));
            std::this_thread::sleep_until(target);
        }

        callback(ev);
        return databento::KeepGoing::Continue;
    };

    impl_->running = true;
    try {
        databento::DbnFileStore store{file_path};
        store.Replay(metadata_callback, record_callback);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("DBN replay failed: ") + e.what());
    }
    impl_->running = false;
}

void DatabentoClient::stop() {
    if (impl_) impl_->running = false;
}

}  // namespace sc
