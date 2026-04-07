#!/usr/bin/env python3
"""
Validate: do PAG causal edges predict 1-tick-ahead (30s) returns better than
rolling correlation?

Usage:
    /ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3.11 \
        scripts/validate_causal_returns.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

# Add project root so analysis/utils.py is importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.utils import Mark, load_snapshots

# ----
# Paths
# ----
TRADE_BIN   = ROOT / "data/test_20251201_allstock_trades.bin"
ID_MAP_FILE = ROOT / "data/test_20251201_allstock_id_map.json"
SNAP_DIR    = ROOT / "output/test_20251201_all"
DATE        = "2025-12-01"
RECORD_FMT  = "<qIB3xqII"   # 32 bytes
RECORD_SIZE = 32
PRICE_SCALE = 1e8
TRAIN_END   = 400            # ticks 0-399 train, 400-778 test
CORR_WINDOW = 60             # rolling correlation look-back in ticks


# ----
# Step 1: VWAP returns
# ----

def compute_vwap_returns(snapshots, n_stocks: int) -> np.ndarray:
    """Return (S, N) array of 30-second log VWAP returns."""
    tick_ts = np.array([s.timestamp_ns for s in snapshots])  # (S,)
    S = len(tick_ts)

    # Tick boundaries: [tick_ts[s-1], tick_ts[s]) for s >= 1; first tick uses
    # everything up to tick_ts[0].
    bounds = np.concatenate([[0], tick_ts])  # len S+1

    # Accumulators: price_sum, size_sum per (tick, stock)
    psum = np.zeros((S, n_stocks))
    ssum = np.zeros((S, n_stocks))

    with open(TRADE_BIN, "rb") as f:
        raw = f.read()

    n_rec = len(raw) // RECORD_SIZE
    for idx in range(n_rec):
        rec = raw[idx * RECORD_SIZE:(idx + 1) * RECORD_SIZE]
        ts_ns, inst_id, _, price_raw, size, _ = struct.unpack(RECORD_FMT, rec)
        if inst_id >= n_stocks:
            continue
        price = price_raw / PRICE_SCALE
        # Find which tick bucket this record belongs to.
        # bounds[s] <= ts_ns < bounds[s+1]  -> bucket s
        s = np.searchsorted(bounds, ts_ns, side="right") - 1
        s = int(np.clip(s, 0, S - 1))
        psum[s, inst_id] += price * size
        ssum[s, inst_id] += size

    # VWAP per tick per stock; use last known VWAP for empty buckets
    vwap = np.full((S, n_stocks), np.nan)
    for s in range(S):
        for j in range(n_stocks):
            if ssum[s, j] > 0:
                vwap[s, j] = psum[s, j] / ssum[s, j]

    # Forward-fill NaNs (carry last traded price)
    for j in range(n_stocks):
        for s in range(1, S):
            if np.isnan(vwap[s, j]):
                vwap[s, j] = vwap[s - 1, j]

    # Log returns: r[s] = log(vwap[s] / vwap[s-1])
    returns = np.zeros((S, n_stocks))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = vwap[1:] / vwap[:-1]
        returns[1:] = np.where(
            np.isfinite(ratio) & (ratio > 0), np.log(ratio), 0.0
        )
    return returns


# ----
# Step 2: Causal predictor signal
# ----

def build_causal_signal(returns: np.ndarray, snapshots) -> np.ndarray:
    """(S, N) causal signal: for each stock j, weighted sum of r[s,i]
    where i has an arrowhead pointing into j in PAG at tick s."""
    S, N = returns.shape
    signal = np.zeros((S, N))
    for s, snap in enumerate(snapshots):
        for e in snap.edges:
            i, j = e.i, e.j
            # Arrowhead at j-side: mark_j == ARROW
            if e.mark_j == Mark.ARROW and not (e.mark_i == Mark.ARROW):
                # i *-> j  (directed or partially oriented into j)
                signal[s, j] += e.weight * returns[s, i]
            # Arrowhead at i-side: mark_i == ARROW (edge stored as i<-* j)
            if e.mark_i == Mark.ARROW and not (e.mark_j == Mark.ARROW):
                signal[s, i] += e.weight * returns[s, j]
    return signal


# ----
# Step 3: Correlation baseline signal
# ----

def build_corr_signal(returns: np.ndarray, window: int = CORR_WINDOW) -> np.ndarray:
    """(S, N) correlation signal: for each stock j, sum of rolling-corr(i,j)*r[s,i]."""
    S, N = returns.shape
    signal = np.zeros((S, N))
    for s in range(1, S):
        lo = max(0, s - window)
        block = returns[lo:s]  # (w, N)
        if block.shape[0] < 2:
            continue
        # Pearson correlation matrix of shape (N, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(block.T)  # (N, N)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        # signal[s, j] = sum_i corr[i, j] * r[s, i]
        signal[s] = corr.T @ returns[s]
    return signal


# ----
# Step 4: OLS walk-forward evaluation
# ----

def ols_predict(x_train, y_train, x_test):
    """Simple OLS with intercept. Returns predictions on test set."""
    X = np.column_stack([np.ones(len(x_train)), x_train])
    beta, _, _, _ = np.linalg.lstsq(X, y_train, rcond=None)
    X_test = np.column_stack([np.ones(len(x_test)), x_test])
    return X_test @ beta


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def directional_accuracy(y_true, y_pred):
    both_nonzero = (y_true != 0) & (y_pred != 0)
    if both_nonzero.sum() == 0:
        return np.nan
    return np.mean(np.sign(y_true[both_nonzero]) == np.sign(y_pred[both_nonzero]))


def diebold_mariano(e1, e2):
    """One-sided DM test: H0 equal MSE, H1 e1 < e2 (causal better).
    Returns t-statistic and p-value."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2  # negative means causal is better
    n = len(d)
    mean_d = np.mean(d)
    # Newey-West variance with lag 1
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.mean((d[1:] - mean_d) * (d[:-1] - mean_d)) if n > 1 else 0
    nw_var = (gamma0 + 2 * gamma1) / n
    if nw_var <= 0:
        return np.nan, np.nan
    t_stat = mean_d / np.sqrt(nw_var)
    p_val = stats.t.cdf(t_stat, df=n - 1)  # one-sided: P(causal better)
    return t_stat, p_val


# ----
# Main
# ----

def main():
    print("Loading symbol map ...", flush=True)
    id_map = {int(k): v for k, v in json.load(open(ID_MAP_FILE)).items()}
    N = len(id_map)
    symbols = [id_map[i] for i in range(N)]

    print("Loading PAG snapshots ...", flush=True)
    snapshots = load_snapshots(SNAP_DIR, DATE)
    S = len(snapshots)
    print(f"  {S} snapshots, {N} stocks", flush=True)

    print("Computing VWAP returns ...", flush=True)
    returns = compute_vwap_returns(snapshots, N)
    print(f"  returns shape: {returns.shape}", flush=True)
    print(f"  non-zero entries: {(returns != 0).sum()} / {returns.size}", flush=True)

    print("Building causal signal ...", flush=True)
    causal_sig = build_causal_signal(returns, snapshots)
    print(f"  non-zero causal entries: {(causal_sig != 0).sum()}", flush=True)

    print("Building correlation signal ...", flush=True)
    corr_sig = build_corr_signal(returns)
    print(f"  non-zero corr entries:   {(corr_sig != 0).sum()}", flush=True)

    # Walk-forward evaluation
    train_idx = slice(TRAIN_END - 1)       # s in [0, TRAIN_END-1] -> predicts s+1
    test_idx  = slice(TRAIN_END - 1, S - 1)  # s in [TRAIN_END-1, S-2]

    y_all     = returns[1:]           # targets: r[s+1]
    c_all     = causal_sig[:-1]       # causal signal at s
    rho_all   = corr_sig[:-1]         # corr signal at s

    y_train, y_test     = y_all[train_idx], y_all[test_idx]
    c_train, c_test     = c_all[train_idx], c_all[test_idx]
    rho_train, rho_test = rho_all[train_idx], rho_all[test_idx]

    header = f"{'Stock':>6}  {'R^2_causal':>10}  {'R^2_corr':>10}  {'DA_causal':>10}  {'DA_corr':>10}  {'DM_p':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    all_e_causal, all_e_corr = [], []

    for j, sym in enumerate(symbols):
        yt = y_test[:, j]
        c_pred  = ols_predict(c_train[:, j],   y_train[:, j], c_test[:, j])
        rho_pred = ols_predict(rho_train[:, j], y_train[:, j], rho_test[:, j])

        r2_c   = r2_score(yt, c_pred)
        r2_rho = r2_score(yt, rho_pred)
        da_c   = directional_accuracy(yt, c_pred)
        da_rho = directional_accuracy(yt, rho_pred)

        e_c   = yt - c_pred
        e_rho = yt - rho_pred
        _, dm_p = diebold_mariano(e_c, e_rho)

        all_e_causal.append(e_c)
        all_e_corr.append(e_rho)

        dm_str = f"{dm_p:.3f}" if dm_p is not None and not np.isnan(dm_p) else "  n/a"
        da_c_s   = f"{da_c:.3f}"   if not np.isnan(da_c)   else "   n/a"
        da_rho_s = f"{da_rho:.3f}" if not np.isnan(da_rho) else "   n/a"
        print(f"{sym:>6}  {r2_c:>10.4f}  {r2_rho:>10.4f}  {da_c_s:>10}  {da_rho_s:>10}  {dm_str:>8}")

    # Pooled DM test across all stocks
    e_c_pool   = np.concatenate(all_e_causal)
    e_rho_pool = np.concatenate(all_e_corr)
    _, dm_p_pool = diebold_mariano(e_c_pool, e_rho_pool)

    # Summary
    r2_causal_vals = [r2_score(y_test[:, j], ols_predict(c_train[:, j], y_train[:, j], c_test[:, j])) for j in range(N)]
    r2_corr_vals   = [r2_score(y_test[:, j], ols_predict(rho_train[:, j], y_train[:, j], rho_test[:, j])) for j in range(N)]

    print(f"\n{'-'*60}")
    print(f"Mean OOS R^2  --  causal: {np.mean(r2_causal_vals):.4f}   corr: {np.mean(r2_corr_vals):.4f}")
    print(f"Pooled DM test p-value (H1: causal MSE < corr MSE): {dm_p_pool:.4f}")
    if dm_p_pool < 0.05:
        print("=> Causal edges significantly outperform correlation baseline (p < 0.05)")
    elif dm_p_pool < 0.10:
        print("=> Causal edges marginally outperform correlation baseline (p < 0.10)")
    elif dm_p_pool > 0.95:
        print("=> Correlation baseline significantly outperforms causal edges")
    else:
        print("=> No significant difference between causal and correlation predictors")


if __name__ == "__main__":
    main()
