#define _USE_MATH_DEFINES  // For MSVC: enable M_PI
#include "bocpd.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace sc {

BOCPD::BOCPD(double hazard_rate, double prior_mean, double prior_var,
             double prior_alpha, double prior_beta)
    : hazard_rate_(hazard_rate),
      base_prior_{prior_mean, 1.0 / prior_var, prior_alpha, prior_beta} {
    // Initialize with run_length = 0 (just started) at probability 1.
    run_length_probs_.push_back(1.0);
    nig_params_.push_back(base_prior_);
}

double BOCPD::update(double g_tau, int64_t timestamp_ns) {
    int R = static_cast<int>(run_length_probs_.size());
    std::vector<double> log_pred(R);

    // Step 1: log predictive P(g_tau | NIG params for each run length r).
    for (int r = 0; r < R; ++r)
        log_pred[r] = std::log(predictive_pdf(g_tau, nig_params_[r]) + 1e-300);

    // Step 2: New run-length log probs (working in log space to avoid underflow).
    std::vector<double> log_new(R + 1);

    // Change point: run_length = 0 gets mass from all existing lengths with hazard.
    double log_sum = -1e300;
    for (int r = 0; r < R; ++r) {
        double lp = std::log(run_length_probs_[r] + 1e-300)
                    + log_pred[r]
                    + std::log(hazard_rate_);
        log_sum = std::max(log_sum, lp) +
                  std::log1p(std::exp(std::min(lp, log_sum) - std::max(lp, log_sum)));
    }
    log_new[0] = log_sum;

    // Continuation: run_length r+1 gets mass from run_length r.
    for (int r = 0; r < R; ++r) {
        log_new[r + 1] = std::log(run_length_probs_[r] + 1e-300)
                       + log_pred[r]
                       + std::log(1.0 - hazard_rate_);
    }

    // Step 3: Normalize.
    double log_total = *std::max_element(log_new.begin(), log_new.end());
    double total = 0.0;
    for (double lp : log_new)
        total += std::exp(lp - log_total);
    log_total += std::log(total);

    run_length_probs_.resize(R + 1);
    for (int r = 0; r <= R; ++r)
        run_length_probs_[r] = std::exp(log_new[r] - log_total);

    // Step 4: Update NIG params (insert base_prior at front for new run length).
    nig_params_.insert(nig_params_.begin(), base_prior_);
    for (int r = 1; r <= R; ++r)
        nig_params_[r] = update_params(nig_params_[r], g_tau);

    // Step 5: Prune negligible run lengths.
    while (!run_length_probs_.empty() && run_length_probs_.back() < 1e-30) {
        run_length_probs_.pop_back();
        nig_params_.pop_back();
    }

    double p_break = run_length_probs_[0];
    p_break_series_.push_back(p_break);
    timestamps_.push_back(timestamp_ns);
    return p_break;
}

int BOCPD::map_run_length() const {
    if (run_length_probs_.empty()) return 0;
    return static_cast<int>(std::max_element(run_length_probs_.begin(),
                                              run_length_probs_.end()) -
                             run_length_probs_.begin());
}

double BOCPD::predictive_pdf(double g_tau, const NIGParams& p) const {
    // Student-t predictive marginal under NIG prior.
    // Degrees of freedom: nu = 2alpha
    double nu    = 2.0 * p.alpha;
    // Scale parameter: sigma = sqrt(beta*(kappa+1)/(alpha*kappa))
    double scale = std::sqrt(p.beta * (p.kappa + 1.0) / (p.alpha * p.kappa));
    double x     = (g_tau - p.mu0) / scale;

    // Student-t PDF: Gamma((nu+1)/2) / (sqrt(nupi)*Gamma(nu/2)*sigma) * (1 + x^2/nu)^{-(nu+1)/2}
    double log_pdf = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
                   - 0.5 * std::log(nu * M_PI) - std::log(scale)
                   - (nu + 1.0) / 2.0 * std::log1p(x * x / nu);
    return std::exp(log_pdf);
}

BOCPD::NIGParams BOCPD::update_params(const NIGParams& p, double g_tau) const {
    // NIG conjugate posterior update.
    double kappa_n = p.kappa + 1.0;
    double mu_n    = (p.kappa * p.mu0 + g_tau) / kappa_n;
    double alpha_n = p.alpha + 0.5;
    double beta_n  = p.beta + 0.5 * p.kappa * (g_tau - p.mu0) * (g_tau - p.mu0) / kappa_n;
    return {mu_n, kappa_n, alpha_n, beta_n};
}

}  // namespace sc
