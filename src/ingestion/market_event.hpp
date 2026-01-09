#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>

namespace sc {

enum class Action : uint8_t {
    ADD = 0
    CANCEL = 1
    TRADE = 2
    MODIFY = 3
};

// Normalized event struct).
// All timestamps in nanoseconds since Unix epoch (PTP-synced exchange time).
struct MarketEvent {
    int64_t ts_event; // exchange timestamp, ns since epoch
    int64_t ts_recv; // local receive timestamp
    uint32_t instrument_id; // internal index [0, N)
    Action action;
    int64_t price; // fixed-point: actual_price * 1e9
    uint64_t size;
};

// Intraday trading regime labels.
enum class Regime : uint8_t {
    PRE_OPEN = 0
    OPEN_AUCTION = 1
    REGULAR = 2
    CLOSE_AUCTION = 3
    ANNOUNCEMENT = 4
    UNKNOWN = 5
};

// SPY/QQQ/IWM ETFs + ES/NQ/ZN/ZB/6E/CL futures = 59 instruments).
// Key: Databento raw_symbol string. Value: internal instrument_id [0,58].
extern const std::unordered_map<std::string, uint32_t> SYMBOL_TO_ID;
extern const std::unordered_map<uint32_t, std::string> ID_TO_SYMBOL;

} // namespace sc
