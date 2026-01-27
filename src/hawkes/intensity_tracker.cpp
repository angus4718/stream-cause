#include "intensity_tracker.hpp"
#include <cmath>

namespace sc {

IntensityTracker::IntensityTracker(int n_instruments, double beta)
    : n_(n_instruments), beta_(beta),
      alpha_(Eigen::MatrixXd::Zero(n_instruments, n_instruments)),
      mu_(Eigen::VectorXd::Constant(n_instruments, 0.1)),
      lambda_(Eigen::VectorXd::Zero(n_instruments)),
      live_R_(n_instruments, std::vector<double>(n_instruments, 0.0)),
      last_t_(n_instruments, -1.0),
      pending_(n_instruments, std::vector<int>(n_instruments, 0)) {}

void IntensityTracker::set_params(const Eigen::MatrixXd& alpha,
                                  const Eigen::VectorXd& mu) {
    alpha_ = alpha;
    mu_    = mu;
}

const Eigen::VectorXd& IntensityTracker::on_event(const MarketEvent& ev) {
    int j = static_cast<int>(ev.instrument_id);
    if (j < 0 || j >= n_) return lambda_;

    double t_j = static_cast<double>(ev.ts_event) * 1e-9;  // ns -> seconds

    if (last_t_[j] >= 0.0) {
        double dt    = t_j - last_t_[j];
        double decay = std::exp(-beta_ * dt);
        for (int k = 0; k < n_; ++k) {
            // At this j-event, consume pending counts for pair (j, k).
            // Self pair (k==j): the event at last_t_[j] contributes +1.
            int n_k = (k == j) ? 1 : pending_[j][k];
            live_R_[j][k] = decay * (live_R_[j][k] + n_k);
            pending_[j][k] = 0;
        }
    } else {
        for (int k = 0; k < n_; ++k) {
            live_R_[j][k] = 0.0;
            pending_[j][k] = 0;
        }
    }

    // Mark a j-event pending for all other instruments' R_ij counts.
    for (int i = 0; i < n_; ++i)
        if (i != j) pending_[i][j]++;

    last_t_[j] = t_j;

    // Recompute lambda_j with updated R_j*.
    double lam_j = mu_(j);
    for (int k = 0; k < n_; ++k)
        lam_j += alpha_(j, k) * beta_ * live_R_[j][k];
    lambda_(j) = lam_j;

    return lambda_;
}

}  // namespace sc
