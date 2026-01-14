#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "market_event.hpp"
#include "ring_buffer.hpp"

namespace sc {

struct MacroCalendarEntry {
    int64_t timestamp_ns;
    std::string event_type; // "FOMC", "NFP", "CPI"
};

// Routes normalized MarketEvents to per-instrument lock-free ring buffers.
// Also classifies each event's intraday regime.
//
// - Load macro calendar at construction; build interval lookup for
// +/-macro_window_minutes around each announcement.
// - Tag events with Regime before routing so downstream modules can
// segment by regime without re-reading timestamps.
class EventRouter {
public:
    EventRouter(int n_instruments, const std::vector<MacroCalendarEntry>& calendar
                int macro_window_minutes = 30);

    // Route one event to its instrument's ring buffer.
    // Returns the regime tag assigned to this event.
    Regime route(const MarketEvent& event);

    DefaultRingBuffer& get_buffer(int instrument_id);

    // Drain all ring buffers and return all buffered events for [start_ns, end_ns).
    // Used by RollingWindow::collect_window_events().
    std::vector<MarketEvent> drain_window(int64_t start_ns, int64_t end_ns);

    int n_instruments() const { return static_cast<int>(buffers_.size()); }

private:
    Regime classify_regime(int64_t ts_ns) const;

    std::vector<std::unique_ptr<DefaultRingBuffer>> buffers_;
    std::vector<MacroCalendarEntry> calendar_;
    int macro_window_ns_; // +/-window in nanoseconds
};

} // namespace sc
