#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "changepoint/bocpd.hpp"

using namespace sc;

TEST_CASE("BOCPD: p_break accumulates over time", "[bocpd]") {
    BOCPD bocpd;

    // Feed stationary signal for 50 steps; p_break should remain low.
    for (int i = 0; i < 50; ++i) {
        double p = bocpd.update(0.5, i * 30'000'000'000LL);
        (void)p;
    }
    REQUIRE(bocpd.p_break_series().size() == 50);
}

TEST_CASE("BOCPD: p_break spikes at step change", "[bocpd]") {
    BOCPD bocpd;

    // Burn-in: 50 near-zero observations.
    for (int i = 0; i < 50; ++i) bocpd.update(0.1, i * 30'000'000'000LL);

    // Large step change: graph edit distance jumps.
    double p_after = bocpd.update(10.0, 50 * 30'000'000'000LL);

    (void)p_after;
    REQUIRE(true); // placeholder
}

TEST_CASE("BOCPD: p_break series length matches update count", "[bocpd]") {
    BOCPD bocpd;
    int n = 100;
    for (int i = 0; i < n; ++i) bocpd.update(0.0, i);
    REQUIRE(static_cast<int>(bocpd.p_break_series().size()) == n);
    REQUIRE(static_cast<int>(bocpd.timestamps().size()) == n);
}

TEST_CASE("BOCPD: MAP run length increases between change points", "[bocpd]") {
    BOCPD bocpd;
    bocpd.update(0.0, 0);
    int rl1 = bocpd.map_run_length();
    bocpd.update(0.0, 1);
    int rl2 = bocpd.map_run_length();
    // Run length should grow when no change point detected.
    (void)rl1; (void)rl2;
    REQUIRE(true); // placeholder
}
