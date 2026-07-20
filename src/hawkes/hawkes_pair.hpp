#pragma once
#include <deque>
#include <cstdint>
#include <vector>

namespace sc {

// Per-(i,j) state for the rolling Hawkes EM estimator. Maintains the O(1)
// recursive sufficient statistic
//   R_ij(m) = sum_{t_l^j < t_m^i} exp(-beta*(t_m^i - t_l^j))
// via R_ij(m) = exp(-beta*(t_m^i - t_{m-1}^i)) * (R_ij(m-1) + n_j^{m-1}),
// where n_j^{m-1} = number of j-events in [t_{m-1}^i, t_m^i).
struct HawkesPair {
    int i = -1;
    int j = -1;
    double beta = 0.0;
    double R = 0.0;
    std::vector<double> R_history;
    double e_step_numerator_accum = 0.0;
    double mu_hat = 0.1;
    double alpha_hat = 0.01;
    double last_t_i = -1.0;

    void update_R(double t_new_i, int n_j_since_last);
    // Reset per-window state; keeps mu_hat/alpha_hat for warm start.
    void reset_window();
};

}  // namespace sc
