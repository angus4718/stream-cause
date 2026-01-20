#include "hawkes_pair.hpp"
#include <cmath>

namespace sc {

void HawkesPair::update_R(double t_new_i, int n_j_since_last) {
    if (last_t_i < 0.0) {
        // First event in instrument i for this window.
        last_t_i = t_new_i;
        return;
    }
    // Recurrence: R_ij(m) = exp(-beta*dt) * (R_ij(m-1) + n_j_since_last)
    double dt = t_new_i - last_t_i;
    R = std::exp(-beta * dt) * (R + static_cast<double>(n_j_since_last));
    last_t_i = t_new_i;

    // Store R value for E-step access.
    R_history.push_back(R);
}

void HawkesPair::reset_window() {
    R = 0.0;
    e_step_numerator_accum = 0.0;
    last_t_i = -1.0;
    R_history.clear();
    // Intentionally keep mu_hat and alpha_hat for warm start.
}

} // namespace sc
