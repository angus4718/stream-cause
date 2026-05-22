# StreamCause

A C++ streaming pipeline for real-time causal graph discovery across high-frequency equity markets, applied to limit-order-book market making.

## Overview

StreamCause estimates a time-varying multivariate Hawkes branching matrix at 30-second ticks using a rolling EM algorithm, feeds the result into a time-series FCI causal discovery algorithm, and detects structural breaks in the causal graph via CUSUM and BOCPD. The causal graph drives a Hawkes-gated (Hgate) market-making strategy that posts at the best bid/ask only when estimated arrival intensity is below a calibrated threshold.

Applied to 18 NASDAQ equities + SPY (Oct 2026 - Dec 2026):

| Strategy | Sharpe Ratio |
|----------|-------------|
| Hgate (causal)   | **4.41** |
| Fixed spread     | 3.36 |
| TOD baseline     | 3.08 |
| Hinv (intensity) | 3.26 |
