#pragma once
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include "market_event.hpp"

#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>
#include <databento/enums.hpp>
#include <databento/symbol_map.hpp>

namespace sc {

struct Config;

using EventCallback = std::function<void(const MarketEvent&)>;

// Wraps the Databento C++ client for live streaming and historical replay,
// normalizing MBO records into MarketEvent.
class DatabentoClient {
public:
    explicit DatabentoClient(const Config& cfg);
    ~DatabentoClient();

    void subscribe(const std::string& dataset, const std::string& schema, const std::vector<std::string>& symbols);
    void start(EventCallback callback);
    void replay(const std::string& start_date, const std::string& end_date, EventCallback callback, double speed_multiplier = 1.0);
    void stop();

private:
    struct Impl {
        std::string api_key;
        std::string replay_file_path;
        std::atomic<bool> running{false};
    };
    std::unique_ptr<Impl> impl_;
};

}  // namespace sc
