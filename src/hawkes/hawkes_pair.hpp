#pragma once
#include <deque>
#include <cstdint>
#include <vector>

namespace sc {

// Per-(i,j) state for the rolling Hawkes EM estimator.
// Maintains the O(1) recursive sufficient statistic R_ij
// and current parameter estimates.
//
// R_ij(m) := sum_{t_l^j < t_m^i} exp(-beta*(t_m^i - t_l^j))
//
// Recurrence:
// R_ij(m) = exp(-beta*(t_m^i - t_{m-1}^i)) * (R_ij(m-1) + n_j^{m-1})
// where n_j^{m-1} = number of events in instrument j in [t_{m-1}^i, t_m^i).
struct HawkesPair {
    int i = -1;
    int j = -1;
    double beta = 0.0;

    // Current R_ij value (updated O(1) per event in instrument i).
    double R = 0.0;

    // History of R values computed during precompute_R pass.
    // R_history[m] = R_ij at event m in instrument i.
    // Cleared in reset_window(); populated during precompute_R().
    std::vector<double> R_history;

    // Running accumulator for E-step numerator: sum_m R_ij(m) / lambda_i(t_m^i).
    // Reset each EM iteration.
    double e_step_numerator_accum = 0.0;

    // Current parameter estimates (warm-started across windows).
    double mu_hat = 0.1; // baseline intensity for instrument i
    double alpha_hat = 0.01; // branching ratio alpha_{ij}

    // Time of the last event in instrument i (nanoseconds, converted to seconds
    // before passing into Hawkes math to keep floating-point precision).
    double last_t_i = -1.0;

    // Called when a new event arrives in instrument i at time t_new_i (seconds).
    // n_j_since_last = number of events in instrument j since last_t_i.
    void update_R(double t_new_i, int n_j_since_last);

    // Reset per-window state (keep parameter estimates for warm start).
    void reset_window();
};

} // namespace sc
