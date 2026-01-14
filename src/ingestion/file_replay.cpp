#include "file_replay.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdio>

namespace sc {

void FileReplay::replay(const std::string& path,
                        int64_t           tick_interval_ns,
                        EventCb           event_cb,
                        TickCb            tick_cb) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("FileReplay: cannot open " + path);
    }

    int64_t next_tick_ns = -1;  // set on first record
    uint64_t n_events = 0;
    uint64_t n_ticks  = 0;

    EventRecord rec;
    while (f.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
        const int64_t ts = rec.ts_event_ns;

        // Initialise tick boundary on first record
        if (next_tick_ns < 0) {
            // Align to the next full tick boundary after the first timestamp
            next_tick_ns = (ts / tick_interval_ns + 1) * tick_interval_ns;
        }

        // Fire any elapsed ticks before delivering the event
        while (tick_interval_ns > 0 && ts >= next_tick_ns) {
            tick_cb(next_tick_ns);
            next_tick_ns += tick_interval_ns;
            ++n_ticks;
        }

        // Build MarketEvent
        MarketEvent ev;
        ev.ts_event      = ts;
        ev.ts_recv       = ts;  // no separate recv timestamp in preprocessed file
        ev.instrument_id = rec.instrument_id;
        ev.price         = rec.price;
        ev.size          = rec.size;
        switch (rec.action) {
            case 0:  ev.action = Action::ADD;    break;
            case 1:  ev.action = Action::CANCEL; break;
            case 2:  ev.action = Action::TRADE;  break;
            default: ev.action = Action::ADD;    break;
        }

        event_cb(ev);
        ++n_events;
    }

    // Fire a final tick to flush the last window if any events were processed
    if (n_events > 0 && n_ticks == 0) {
        tick_cb(next_tick_ns);
    }

    std::printf("[FileReplay] %llu events, %llu ticks fired\n",
                (unsigned long long)n_events, (unsigned long long)n_ticks);
}

}  // namespace sc
