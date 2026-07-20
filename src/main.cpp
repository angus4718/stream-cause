#include <iostream>
#include <string>
#include <csignal>
#include "pipeline/pipeline.hpp"

static sc::Pipeline* g_pipeline = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_pipeline) g_pipeline->shutdown();
}

int main(int argc, char* argv[]) {
    std::string config_path = "config/spy18_allstock.json";
    std::string mode = "replay";
    std::string file_path = "";
    std::string start_date = "2023-01-03";
    std::string end_date = "2023-01-03";
    double speed = 10.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--file" && i + 1 < argc) file_path = argv[++i];
        else if (arg == "--start" && i + 1 < argc) start_date = argv[++i];
        else if (arg == "--end" && i + 1 < argc) end_date = argv[++i];
        else if (arg == "--speed" && i + 1 < argc) speed = std::stod(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: streamcause [--config PATH] [--mode live|replay]\n"
                      << " [--file PREPROCESSED_BIN_PATH]\n"
                      << " [--start YYYY-MM-DD] [--end YYYY-MM-DD]\n"
                      << " [--speed MULTIPLIER]\n";
            return 0;
        }
    }

    sc::Config cfg = sc::Config::from_json_file(config_path);
    sc::Pipeline pipeline(cfg);
    g_pipeline = &pipeline;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (mode == "live") {
        pipeline.run_live();
    } else {
        const std::string& replay_src = file_path.empty() ? start_date : file_path;
        pipeline.run_replay(replay_src, end_date, speed);
    }

    return 0;
}
