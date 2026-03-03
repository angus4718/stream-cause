#pragma once
#include <vector>
#include <cstdint>

namespace sc {

// Graph-level Bayesian Online Change Point Detection (Adams & MacKay 2007).
// Operates on the scalar graph summary statistic g(tau_s):
// g(tau_s) = sum_{(i,j)} 1[edge_type_changes(i,j)] * w_{ij}
// (the weighted graph edit distance from the previous PAG snapshot).
//
// Uses a Gaussian observation model with Normal-InverseGamma conjugate prior
// for efficient online Bayesian updates.
//
// Produces p_break(tau_s) = posterior probability that a change point occurred
// at or before tau_s, for use in Experiment 3 (stress event co-occurrence).
//
class BOCPD {
public:
    // hazard_rate: 1/lambda in the geometric prior over run lengths.
    // Default lambda=200 snapshots = 100 minutes at 30s updates.
    // prior_mean, prior_var: Normal prior on mean of g(tau_s).
    // prior_alpha, prior_beta: InverseGamma prior on variance.
    BOCPD(double hazard_rate = 1.0 / 200.0
          double prior_mean = 0.0
          double prior_var = 1.0
          double prior_alpha = 1.0
          double prior_beta = 1.0);

    // Update with new observation g_tau; returns p_break(tau_s).
    //
    // 1. Compute predictive probability P(g_tau | run_length=r) for each r.
    // 2. Multiply by hazard (change point) and (1 - hazard) (no change).
    // 3. Normalize run-length posterior.
    // 4. p_break = posterior mass on run_length = 0.
    // 5. Update Normal-InverseGamma sufficient statistics for each run length.
    // 6. Prune run lengths with negligible posterior mass (< 1e-30).
    double update(double g_tau, int64_t timestamp_ns);

    // MAP run length estimate (most probable run length).
    int map_run_length() const;

    // Full run-length posterior (for diagnostics).
    const std::vector<double>& run_length_probs() const { return run_length_probs_; }

    // Time series of p_break values (exposed for Python analysis).
    const std::vector<double>& p_break_series() const { return p_break_series_; }
    const std::vector<int64_t>& timestamps() const { return timestamps_; }

private:
    // Normal-InverseGamma sufficient statistics per run length.
    struct NIGParams {
        double mu0; // prior mean
        double kappa; // prior pseudo-counts on mean
        double alpha; // InverseGamma shape
        double beta; // InverseGamma rate
    };

    // Predictive density P(g_tau | run_length=r) under Student-t marginal.
    double predictive_pdf(double g_tau, const NIGParams& params) const;

    // Update NIG params with new observation.
    NIGParams update_params(const NIGParams& params, double g_tau) const;

    double hazard_rate_;
    NIGParams base_prior_;

    std::vector<double> run_length_probs_; // P(r_t = r | data)
    std::vector<NIGParams> nig_params_; // per run length
    std::vector<double> p_break_series_;
    std::vector<int64_t> timestamps_;
};

} // namespace sc
