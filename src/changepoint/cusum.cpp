#include "cusum.hpp"
#include <cmath>
#include <algorithm>

namespace sc {

CUSUM::CUSUM(double threshold, double allowance_k)
    : threshold_(threshold), k_(allowance_k) {}

bool CUSUM::update(double alpha_ij, int64_t timestamp_ns) {
    // Welford online mean/variance update.
    ++n_;
    double delta = alpha_ij - mean_;
    mean_ += delta / n_;
    double delta2 = alpha_ij - mean_;
    m2_ += delta * delta2;
    if (n_ > 1) sigma_hat_ = std::sqrt(m2_ / (n_ - 1));

    // Normalize: skip variance estimates until we have enough samples.
    double alpha_bar = (n_ < 5 || sigma_hat_ < 1e-10)
                       ? 0.0
                       : (alpha_ij - mean_) / sigma_hat_;

    // Two-sided CUSUM update.
    s_plus_  = std::max(0.0, s_plus_  + alpha_bar - k_);
    s_minus_ = std::max(0.0, s_minus_ - alpha_bar - k_);

    // Check threshold and record break.
    if (s_plus_ > threshold_ || s_minus_ > threshold_) {
        last_break_ns_ = timestamp_ns;
        reset();
        return true;
    }

    return false;
}

void CUSUM::reset() {
    s_plus_  = 0.0;
    s_minus_ = 0.0;
    // Intentionally keep Welford state (mean_, m2_, n_) to maintain
    // long-run sigma_hat estimate across break boundaries.
}

}  // namespace sc
