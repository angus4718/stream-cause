#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include "pag.hpp"

namespace sc {

// Persists time-indexed PAG snapshots and alpha_hat matrices to disk.
// Storage format: JSON files in {output_dir}/{YYYY-MM-DD}/snapshot_{ts_ns}.json.
// Also writes alpha_hat matrices as {output_dir}/{YYYY-MM-DD}/alpha_{ts_ns}.bin (binary float64).
//
class GraphStore {
public:
    explicit GraphStore(std::string output_dir);

    // Append a new PAG snapshot and its associated alpha_hat matrix.
    // Creates date subdirectory if it doesn't exist.
    void append(const PAG& pag, const Eigen::MatrixXd& alpha);

    // Append the per-instrument Hawkes intensity vector lambda at a given tick.
    // Format: [int32 n][int32 1][float64 * n], same as alpha_*.bin.
    void append_lambda(int64_t ts_ns, const Eigen::VectorXd& lambda);

    // Load all PAG snapshots in [start_ns, end_ns).
    std::vector<PAG> load_range(int64_t start_ns, int64_t end_ns) const;

    // Load the time series of alpha_hat matrices for [start_ns, end_ns).
    // Returns vector ordered by timestamp (oldest first).
    std::vector<Eigen::MatrixXd> load_alpha_series(int64_t start_ns
                                                    int64_t end_ns) const;

    // Load the most recent K snapshots.
    std::vector<PAG> load_last_k(int k) const;

    const std::string& output_dir() const { return output_dir_; }

private:
    std::string date_dir(int64_t timestamp_ns) const;
    std::string snapshot_path(int64_t timestamp_ns) const;
    std::string alpha_path(int64_t timestamp_ns) const;
    std::string lambda_path(int64_t timestamp_ns) const;

    std::string output_dir_;
};

} // namespace sc
