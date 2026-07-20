#include "hawkes_pair.hpp"
#include <cmath>

namespace sc {

void HawkesPair::update_R(double t_new_i, int n_j_since_last) {
    if (last_t_i < 0.0) {
        last_t_i = t_new_i;
        return;
    }
    double dt = t_new_i - last_t_i;
    R = std::exp(-beta * dt) * (R + static_cast<double>(n_j_since_last));
    last_t_i = t_new_i;
    R_history.push_back(R);
}

void HawkesPair::reset_window() {
    R = 0.0;
    e_step_numerator_accum = 0.0;
    last_t_i = -1.0;
    R_history.clear();
}

}  // namespace sc
