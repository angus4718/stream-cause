#pragma once
#include <functional>
#include <string>
#include <cstdint>
#include "market_event.hpp"

namespace sc {

// 32-byte packed record written by the preprocess scripts:
// struct.pack("<qIB3xqII", ts_event_ns, instrument_id, action, price, size, pad2).
#pragma pack(push, 1)
struct EventRecord {
    int64_t ts_event_ns;
    uint32_t instrument_id;
    uint8_t action;  // 0=ADD, 1=CANCEL, 2=TRADE
    uint8_t pad[3];
    int64_t price;
    uint32_t size;
    uint32_t pad2;
};
#pragma pack(pop)
static_assert(sizeof(EventRecord) == 32, "EventRecord must be 32 bytes");

// Replays a preprocessed binary event file. tick_cb fires before the first event
// of each new tick window, so on_tick sees a complete window of prior events.
class FileReplay {
public:
    using EventCb = std::function<void(const MarketEvent&)>;
    using TickCb = std::function<void(int64_t ts_ns)>;

    void replay(const std::string& path, int64_t tick_interval_ns, EventCb event_cb, TickCb tick_cb);
};

}  // namespace sc
