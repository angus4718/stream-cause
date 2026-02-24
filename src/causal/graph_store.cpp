#include "graph_store.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace sc {

GraphStore::GraphStore(std::string output_dir) : output_dir_(std::move(output_dir)) {
    fs::create_directories(output_dir_);
}

void GraphStore::append(const PAG& pag, const Eigen::MatrixXd& alpha) {
    std::string dir = date_dir(pag.timestamp_ns());
    fs::create_directories(dir);

    // Write PAG JSON.
    std::ofstream json_file(snapshot_path(pag.timestamp_ns()));
    json_file << pag.to_json().dump(2);
    json_file.close();

    // Write alpha as binary float64 (column-major, with row/col counts).
    std::ofstream bin_file(alpha_path(pag.timestamp_ns()), std::ios::binary);
    int32_t rows = alpha.rows(), cols = alpha.cols();
    bin_file.write(reinterpret_cast<const char*>(&rows), 4);
    bin_file.write(reinterpret_cast<const char*>(&cols), 4);
    bin_file.write(reinterpret_cast<const char*>(alpha.data()),
                   rows * cols * static_cast<std::streamsize>(sizeof(double)));
    bin_file.close();

    // spdlog::info("appended PAG and alpha at {}ns", pag.timestamp_ns());
}

std::vector<PAG> GraphStore::load_range(int64_t start_ns, int64_t end_ns) const {
    std::vector<PAG> result;

    if (!fs::exists(output_dir_)) return result;

    // Collect all snapshot files from all date directories.
    std::vector<std::pair<int64_t, std::string>> files;  // (timestamp, filepath)

    for (const auto& date_entry : fs::directory_iterator(output_dir_)) {
        if (!date_entry.is_directory()) continue;

        for (const auto& file_entry : fs::directory_iterator(date_entry.path())) {
            if (!file_entry.is_regular_file()) continue;
            std::string filename = file_entry.path().filename().string();

            if (filename.find("snapshot_") != 0 || filename.find(".json") == std::string::npos)
                continue;

            // Extract timestamp from filename: snapshot_<timestamp>.json
            size_t start = 9;  // len("snapshot_")
            size_t end = filename.find(".json");
            if (end == std::string::npos) continue;

            try {
                int64_t ts = std::stoll(filename.substr(start, end - start));
                if (ts >= start_ns && ts < end_ns) {
                    files.push_back({ts, file_entry.path().string()});
                }
            } catch (...) {
                continue;
            }
        }
    }

    // Sort by timestamp and deserialize.
    std::sort(files.begin(), files.end());
    for (const auto& [ts, path] : files) {
        std::ifstream f(path);
        try {
            nlohmann::json j = nlohmann::json::parse(f);
            result.push_back(PAG::from_json(j));
        } catch (...) {
            // Skip malformed files
        }
    }

    return result;
}

std::vector<Eigen::MatrixXd> GraphStore::load_alpha_series(int64_t start_ns,
                                                             int64_t end_ns) const {
    std::vector<Eigen::MatrixXd> result;

    if (!fs::exists(output_dir_)) return result;

    // Collect all alpha files from all date directories.
    std::vector<std::pair<int64_t, std::string>> files;  // (timestamp, filepath)

    for (const auto& date_entry : fs::directory_iterator(output_dir_)) {
        if (!date_entry.is_directory()) continue;

        for (const auto& file_entry : fs::directory_iterator(date_entry.path())) {
            if (!file_entry.is_regular_file()) continue;
            std::string filename = file_entry.path().filename().string();

            if (filename.find("alpha_") != 0 || filename.find(".bin") == std::string::npos)
                continue;

            // Extract timestamp from filename: alpha_<timestamp>.bin
            size_t start = 6;  // len("alpha_")
            size_t end = filename.find(".bin");
            if (end == std::string::npos) continue;

            try {
                int64_t ts = std::stoll(filename.substr(start, end - start));
                if (ts >= start_ns && ts < end_ns) {
                    files.push_back({ts, file_entry.path().string()});
                }
            } catch (...) {
                continue;
            }
        }
    }

    // Sort by timestamp and deserialize.
    std::sort(files.begin(), files.end());
    for (const auto& [ts, path] : files) {
        std::ifstream f(path, std::ios::binary);
        int32_t rows, cols;
        f.read(reinterpret_cast<char*>(&rows), 4);
        f.read(reinterpret_cast<char*>(&cols), 4);

        if (rows <= 0 || cols <= 0) continue;

        Eigen::MatrixXd alpha(rows, cols);
        f.read(reinterpret_cast<char*>(alpha.data()),
               rows * cols * static_cast<std::streamsize>(sizeof(double)));
        result.push_back(alpha);
    }

    return result;
}

std::vector<PAG> GraphStore::load_last_k(int k) const {
    std::vector<PAG> result;

    if (!fs::exists(output_dir_)) return result;

    // Collect all snapshot files.
    std::vector<std::pair<int64_t, std::string>> files;  // (timestamp, filepath)

    for (const auto& date_entry : fs::directory_iterator(output_dir_)) {
        if (!date_entry.is_directory()) continue;

        for (const auto& file_entry : fs::directory_iterator(date_entry.path())) {
            if (!file_entry.is_regular_file()) continue;
            std::string filename = file_entry.path().filename().string();

            if (filename.find("snapshot_") != 0 || filename.find(".json") == std::string::npos)
                continue;

            // Extract timestamp from filename: snapshot_<timestamp>.json
            size_t start = 9;  // len("snapshot_")
            size_t end = filename.find(".json");
            if (end == std::string::npos) continue;

            try {
                int64_t ts = std::stoll(filename.substr(start, end - start));
                files.push_back({ts, file_entry.path().string()});
            } catch (...) {
                continue;
            }
        }
    }

    // Sort by timestamp descending, take last k, then reverse to ascending order.
    std::sort(files.rbegin(), files.rend());
    if (static_cast<int>(files.size()) > k)
        files.erase(files.begin() + k, files.end());
    std::sort(files.begin(), files.end());

    // Deserialize.
    for (const auto& [ts, path] : files) {
        std::ifstream f(path);
        try {
            nlohmann::json j = nlohmann::json::parse(f);
            result.push_back(PAG::from_json(j));
        } catch (...) {
            // Skip malformed files
        }
    }

    return result;
}

std::string GraphStore::date_dir(int64_t timestamp_ns) const {
    time_t t = timestamp_ns / 1'000'000'000LL;
    char buf[11];
    strftime(buf, sizeof(buf), "%Y-%m-%d", gmtime(&t));
    return output_dir_ + "/" + std::string(buf);
}

std::string GraphStore::snapshot_path(int64_t timestamp_ns) const {
    return date_dir(timestamp_ns) + "/snapshot_" + std::to_string(timestamp_ns) + ".json";
}

std::string GraphStore::alpha_path(int64_t timestamp_ns) const {
    return date_dir(timestamp_ns) + "/alpha_" + std::to_string(timestamp_ns) + ".bin";
}

std::string GraphStore::lambda_path(int64_t timestamp_ns) const {
    return date_dir(timestamp_ns) + "/lambda_" + std::to_string(timestamp_ns) + ".bin";
}

void GraphStore::append_lambda(int64_t ts_ns, const Eigen::VectorXd& lambda) {
    std::string dir = date_dir(ts_ns);
    fs::create_directories(dir);
    std::ofstream f(lambda_path(ts_ns), std::ios::binary);
    int32_t rows = static_cast<int32_t>(lambda.size()), cols = 1;
    f.write(reinterpret_cast<const char*>(&rows), 4);
    f.write(reinterpret_cast<const char*>(&cols), 4);
    f.write(reinterpret_cast<const char*>(lambda.data()),
            rows * static_cast<std::streamsize>(sizeof(double)));
}

}  // namespace sc
