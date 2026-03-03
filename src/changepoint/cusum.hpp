#pragma once
#include <cstdint>
#include <utility>

namespace sc {

// Edge-level two-sided CUSUM structural break detector.
// Operates on the normalized branching ratio alpha_bar_{ij}(tau_s) = alpha_hat_{ij}(tau_s) / sigma_hat_{ij}.
// sigma_hat_{ij} is estimated online via Welford's algorithm.
//
// CUSUM statistics:
// S_plus(s) = max(0, S_plus(s-1) + alpha_bar_{ij}(tau_s) - k)
// S_minus(s) = max(0, S_minus(s-1) - alpha_bar_{ij}(tau_s) - k)
// Break declared when S_plus > h or S_minus > h, with h=5 and k=0.5.
class CUSUM {
public:
    CUSUM(double threshold = 5.0, double allowance_k = 0.5);

    // Update with new alpha_hat_{ij} value; returns true if structural break detected.
    // Also updates the rolling sigma_hat_{ij} via Welford's online algorithm.
    //
    // Compute sigma_hat = sqrt(m2_ / (n_ - 1)).
    // Compute normalized: alpha_bar = (alpha_ij - mean_) / sigma_hat.
    // Update S_plus, S_minus; check threshold.
    // Return true and record break_timestamp_ns on breach.
    bool update(double alpha_ij, int64_t timestamp_ns);

    // Reset CUSUM statistics after a confirmed break (start fresh run).
    void reset();

    // Accessors for diagnostics.
    double s_plus() const { return s_plus_; }
    double s_minus() const { return s_minus_; }
    double sigma_hat() const { return sigma_hat_; }
    int n_updates() const { return n_; }
    int64_t last_break_ns() const { return last_break_ns_; }

private:
    double threshold_;
    double k_; // allowance parameter (reference value)

    // CUSUM statistics.
    double s_plus_ = 0.0;
    double s_minus_ = 0.0;

    // Welford online variance estimator.
    int n_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0; // sum of squared deviations
    double sigma_hat_ = 1.0;

    int64_t last_break_ns_ = 0;
};

} // namespace sc
