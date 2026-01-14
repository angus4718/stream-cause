#pragma once
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include "market_event.hpp"

// Databento C++ SDK headers
#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>
#include <databento/enums.hpp>
#include <databento/symbol_map.hpp>
// Note: cpp-httplib and date library are transitive dependencies provided via CMake

namespace sc {

struct Config;

using EventCallback = std::function<void(const MarketEvent&)>;

// Wraps the Databento C++ client for both live streaming and historical replay.
// Normalizes MBO records into MarketEvent and applies session/event-type filters.
//
// - Filter to ADD/CANCEL/TRADE only (drop MODIFY/FILL unless needed).
// - Apply session time filter: 09:30-16:00 ET for equities, 08:30-15:15 CT
// for CME futures (converted to ns since epoch UTC).
// - Merge same-nanosecond same-instrument events into a single batch record
// to avoid degenerate likelihood contributions.
// - Read DATABENTO_API_KEY from environment variable.
// - Handle reconnection and heartbeat for live mode.
class DatabentoClient {
public:
    explicit DatabentoClient(const Config& cfg);
    ~DatabentoClient();

    // Subscribe to MBO streams for equities (XNAS.ITCH) and futures (GLBX.MDP3).
    void subscribe(const std::string& dataset
                   const std::string& schema
                   const std::vector<std::string>& symbols);

    // Begin live streaming. Blocks; calls callback for each normalized event.
    void start(EventCallback callback);

    // Historical replay of a date range at configurable wall-clock speed.
    // speed_multiplier=10.0 replays 10* faster than real-time.
    // maintain replay clock to pace delivery.
    void replay(const std::string& start_date
                const std::string& end_date
                EventCallback callback
                double speed_multiplier = 1.0);

    void stop();

private:
    struct Impl {
        std::string api_key;
        std::string replay_file_path; // For file-based replay
        std::atomic<bool> running{false};
    };
    std::unique_ptr<Impl> impl_;
};

} // namespace sc
