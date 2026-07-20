#include "event_router.hpp"
#include <stdexcept>
#include <algorithm>

namespace sc {

static constexpr int64_t ET_OFFSET_NS = -5LL * 3600LL * 1'000'000'000LL;

EventRouter::EventRouter(int n_instruments) {
    buffers_.reserve(n_instruments);
    for (int i = 0; i < n_instruments; ++i) {
        buffers_.push_back(std::make_unique<DefaultRingBuffer>());
    }
}

Regime EventRouter::route(const MarketEvent& event) {
    if (event.instrument_id >= buffers_.size()) {
        return Regime::UNKNOWN;
    }
    Regime regime = classify_regime(event.ts_event);
    buffers_[event.instrument_id]->push(event);
    return regime;
}

DefaultRingBuffer& EventRouter::get_buffer(int instrument_id) {
    if (instrument_id < 0 || instrument_id >= static_cast<int>(buffers_.size())) {
        throw std::out_of_range("instrument_id out of range");
    }
    return *buffers_[instrument_id];
}

std::vector<MarketEvent> EventRouter::drain_window(int64_t start_ns, int64_t end_ns) {
    std::vector<MarketEvent> result;
    for (auto& buf : buffers_) {
        MarketEvent ev;
        while (buf->pop(ev)) {
            if (ev.ts_event >= start_ns && ev.ts_event < end_ns) {
                result.push_back(ev);
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.ts_event < b.ts_event; });
    return result;
}

Regime EventRouter::classify_regime(int64_t ts_ns) const {
    int64_t et_ns = ts_ns + ET_OFFSET_NS;
    int64_t sod = et_ns % (86400LL * 1'000'000'000LL);
    auto hm = [](int h, int m) -> int64_t { return static_cast<int64_t>(h * 3600 + m * 60) * 1'000'000'000LL; };

    if (sod < hm(8, 0)) return Regime::UNKNOWN;
    if (sod < hm(9, 28)) return Regime::PRE_OPEN;
    if (sod < hm(9, 32)) return Regime::OPEN_AUCTION;
    if (sod < hm(15, 45)) return Regime::REGULAR;
    if (sod < hm(16, 0)) return Regime::CLOSE_AUCTION;
    return Regime::UNKNOWN;
}

}  // namespace sc
