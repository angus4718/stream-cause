# StreamCause

A low-latency C++ streaming pipeline that estimates a time-varying multivariate Hawkes model over live NASDAQ order flow and uses its intensity to gate limit-order-book market making.

## Overview

StreamCause fits a rolling multivariate Hawkes model by EM every 30 seconds, producing at each tick a branching matrix `alpha_hat` (a directed excitation graph over instruments) and a per-instrument arrival-intensity vector `lambda_hat`. A fast layer (`IntensityTracker`) updates `lambda_hat` per trade in O(N) from the frozen slow-layer parameters, while the slow layer (`RollingWindow` -> `EMEstimator`) re-estimates `(alpha_hat, mu_hat)` each tick; both are persisted to disk for offline analysis and backtesting.

The estimated intensity drives the **Hawkes-gate (Hgate)** market-making strategy: post at the best bid/ask when `lambda_i(s) <= tau_i` and sit out otherwise, with `tau_i` a per-instrument quantile calibrated in-sample. Strategies are evaluated two ways: a fast vectorized multi-day backtest that calibrates the per-instrument multipliers and runs block-bootstrap significance tests, and a queue-aware backtest that replays the full MBO stream with FIFO queue-ahead fill logic.

Evaluated on 17 NASDAQ equities plus SPY (18 instruments), calibrated on October-November 2025 and held out on December 2025:

| Strategy | Daily Sharpe |
|----------|-------------|
| **Hgate** (intensity gate) | **4.41** |
| Fixed spread    | 3.36 |
| Hinv (inverse intensity) | 3.26 |
| TOD baseline    | 3.08 |

The gain over Hinv, which consumes the same intensity signal but widens instead of withdrawing, isolates the value of the participation decision under queue-aware fills.

## Layout

- `src/hawkes/`: Hawkes EM estimator, rolling window, per-event intensity tracker
- `src/ingestion/`: MBO event model, lock-free SPSC ring buffer, event router, file replay
- `src/pipeline/`: pipeline orchestration and thread pool
- `scripts/`: the `spy18_1..5` workflow (preprocess trades/LOB, run pipeline, calibrate, backtest) plus `backtest_multiday.py` and `backtest_lob_precise.py`
- `analysis/utils.py`: shared loaders for the `alpha_*.bin` / `lambda_*.bin` outputs
- `tests/`: Catch2 unit tests
- `paper/`: the accompanying write-up

## Build and test

The C++ engine builds with CMake (Eigen, TBB, Boost, spdlog, nlohmann_json; databento is fetched as an external project):

```
cmake --build build-linux -j
./build-linux/tests/streamcause_tests
```

## Pipeline

The end-to-end SPY18 workflow is the numbered `scripts/spy18_*.sh` (SLURM batch jobs): preprocess the trades and LOB binaries, run the C++ pipeline to emit per-tick `alpha`/`lambda`, calibrate per-instrument multipliers with `backtest_multiday.py`, then evaluate under the queue-aware simulator with `backtest_lob_precise.py`.
