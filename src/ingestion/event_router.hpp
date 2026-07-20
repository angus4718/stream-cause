#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "market_event.hpp"
#include "ring_buffer.hpp"

namespace sc {

// Routes MarketEvents to per-instrument ring buffers and tags each with a Regime.
class EventRouter {
public:
    explicit EventRouter(int n_instruments);

    Regime route(const MarketEvent& event);
    DefaultRingBuffer& get_buffer(int instrument_id);
    std::vector<MarketEvent> drain_window(int64_t start_ns, int64_t end_ns);
    int n_instruments() const { return static_cast<int>(buffers_.size()); }

private:
    Regime classify_regime(int64_t ts_ns) const;

    std::vector<std::unique_ptr<DefaultRingBuffer>> buffers_;
};

}  // namespace sc
