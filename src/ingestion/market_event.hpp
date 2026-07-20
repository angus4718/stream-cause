#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>

namespace sc {

enum class Action : uint8_t {
    ADD = 0,
    CANCEL = 1,
    TRADE = 2,
    MODIFY = 3,
};

struct MarketEvent {
    int64_t ts_event;
    int64_t ts_recv;
    uint32_t instrument_id;
    Action action;
    int64_t price;  // fixed-point: actual_price * 1e9
    uint64_t size;
};

enum class Regime : uint8_t {
    PRE_OPEN = 0,
    OPEN_AUCTION = 1,
    REGULAR = 2,
    CLOSE_AUCTION = 3,
    ANNOUNCEMENT = 4,
    UNKNOWN = 5,
};

extern const std::unordered_map<std::string, uint32_t> SYMBOL_TO_ID;
extern const std::unordered_map<uint32_t, std::string> ID_TO_SYMBOL;

}  // namespace sc
