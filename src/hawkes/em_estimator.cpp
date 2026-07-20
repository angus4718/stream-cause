#include "em_estimator.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace sc {

EMEstimator::EMEstimator(int n_instruments, double beta, double alpha_reg)
    : n_(n_instruments), beta_(beta), alpha_reg_(alpha_reg),
      alpha_(Eigen::MatrixXd::Constant(n_instruments, n_instruments, 0.01 / n_instruments)),
      mu_(Eigen::VectorXd::Constant(n_instruments, 0.1)) {
    pairs_.resize(n_);
    for (int i = 0; i < n_; ++i) {
        pairs_[i].resize(n_);
        for (int j = 0; j < n_; ++j) {
            pairs_[i][j].i = i;
            pairs_[i][j].j = j;
            pairs_[i][j].beta = beta_;
        }
    }
}

Eigen::MatrixXd EMEstimator::run(const WindowEvents& events, int max_iter, double epsilon) {
    if (events.empty()) return alpha_;

    double T = 0.0;
    for (const auto& ie : events) {
        if (!ie.times.empty()) T = std::max(T, ie.times.back());
    }
    if (T <= 0.0) return alpha_;

    reset_pairs();
    precompute_R(events);

    Eigen::MatrixXd prev_alpha = alpha_;
    for (int k = 0; k < max_iter; ++k) {
        e_step(events);
        m_step(events, T);
        project_alpha();
        double frob = (alpha_ - prev_alpha).norm();
        if (frob < epsilon) break;
        prev_alpha = alpha_;
    }
    return alpha_;
}

double EMEstimator::log_likelihood(const WindowEvents& events) const {
    double T = 0.0;
    for (const auto& ie : events)
        if (!ie.times.empty()) T = std::max(T, ie.times.back());

    double ll = 0.0;
    for (int i = 0; i < n_; ++i) {
        ll -= mu_(i) * T;
        for (int j = 0; j < n_; ++j) {
            const auto& tj = events[j].times;
            double integral = 0.0;
            for (double tk : tj)
                integral += 1.0 - std::exp(-beta_ * (T - tk));
            ll -= alpha_(i, j) * integral;
        }
    }
    return ll;
}

void EMEstimator::warm_start(const Eigen::MatrixXd& prev_alpha) {
    if (prev_alpha.rows() != n_ || prev_alpha.cols() != n_) return;
    if (prev_alpha.norm() < 1e-12) return;
    alpha_ = prev_alpha;
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
            pairs_[i][j].alpha_hat = prev_alpha(i, j);
}

void EMEstimator::precompute_R(const WindowEvents& events) {
    for (int i = 0; i < n_; ++i) {
        const auto& ti = events[i].times;
        for (int j = 0; j < n_; ++j) {
            pairs_[i][j].reset_window();
            const auto& tj = events[j].times;
            int ptr_j = 0;
            for (double tm : ti) {
                int n_j = 0;
                double prev_tm = (pairs_[i][j].last_t_i < 0) ? 0.0 : pairs_[i][j].last_t_i;
                while (ptr_j < static_cast<int>(tj.size()) && tj[ptr_j] < tm) {
                    if (tj[ptr_j] >= prev_tm) ++n_j;
                    ++ptr_j;
                }
                pairs_[i][j].update_R(tm, n_j);
            }
        }
    }
}

void EMEstimator::e_step(const WindowEvents& events) {
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
            pairs_[i][j].e_step_numerator_accum = 0.0;

    for (int i = 0; i < n_; ++i) {
        const auto& ti = events[i].times;
        for (int m = 0; m < static_cast<int>(ti.size()); ++m) {
            // lambda_i(t_m) = mu_i + sum_j alpha_ij * beta * R_ij(m)
            double lambda_i = mu_(i);
            for (int j = 0; j < n_; ++j) {
                if (m < static_cast<int>(pairs_[i][j].R_history.size())) {
                    lambda_i += alpha_(i, j) * beta_ * pairs_[i][j].R_history[m];
                }
            }
            if (lambda_i < 1e-300) lambda_i = 1e-300;
            for (int j = 0; j < n_; ++j) {
                if (m < static_cast<int>(pairs_[i][j].R_history.size())) {
                    double contrib = alpha_(i, j) * beta_ * pairs_[i][j].R_history[m] / lambda_i;
                    pairs_[i][j].e_step_numerator_accum += contrib;
                }
            }
        }
    }
}

void EMEstimator::m_step(const WindowEvents& events, double T) {
    Eigen::VectorXd bg_resp_sum = Eigen::VectorXd::Zero(n_);
    for (int i = 0; i < n_; ++i) {
        const auto& ti = events[i].times;
        for (int m = 0; m < static_cast<int>(ti.size()); ++m) {
            double lambda_i = mu_(i);
            for (int j = 0; j < n_; ++j) {
                if (m < static_cast<int>(pairs_[i][j].R_history.size())) {
                    lambda_i += alpha_(i, j) * beta_ * pairs_[i][j].R_history[m];
                }
            }
            if (lambda_i < 1e-300) lambda_i = 1e-300;
            bg_resp_sum(i) += mu_(i) / lambda_i;
        }
    }

    for (int i = 0; i < n_; ++i)
        mu_(i) = bg_resp_sum(i) / T;

    for (int j = 0; j < n_; ++j) {
        const auto& tj = events[j].times;
        double denom_j = 0.0;
        for (double tk : tj)
            denom_j += 1.0 - std::exp(-beta_ * (T - tk));

        for (int i = 0; i < n_; ++i) {
            double numer = pairs_[i][j].e_step_numerator_accum;
            if (denom_j > 1e-300)
                alpha_(i, j) = numer / denom_j;
            else
                alpha_(i, j) = 0.0;
            alpha_(i, j) = std::max(0.0, alpha_(i, j));
            // Ridge floor on active cross-terms; inactive instruments stay at 0.
            if (i != j && alpha_reg_ > 0.0 && denom_j > 1e-300)
                alpha_(i, j) += alpha_reg_;
        }
    }
}

void EMEstimator::project_alpha() {
    Eigen::EigenSolver<Eigen::MatrixXd> es(alpha_, false);
    double rho = es.eigenvalues().cwiseAbs().maxCoeff();
    if (rho >= 1.0) {
        alpha_ *= 0.99 / rho;
    }
}

void EMEstimator::reset_pairs() {
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
            pairs_[i][j].reset_window();
}

}  // namespace sc
