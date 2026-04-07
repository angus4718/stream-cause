"""
Market Maker Validation Experiments
====================================
Compares Hawkes-based intensity signals against baseline methods for the
adverse-selection use case that StreamCause is designed to address.

Four experiments:
  Exp 1 -- Intensity Forecasting: Hawkes lambda_hat vs. EWMA for next-30s trade count
  Exp 2 -- Adverse Selection Signal: Hawkes lambda_hat vs. count, correlation with |return|
  Exp 3 -- Cross-Stock Causal Prediction: alpha_hat off-diagonal vs. corr-weighted signal
  Exp 4 -- Structural Break Detection: alpha_hat Frobenius jumps vs. vol-spike alignment

Usage:
  python scripts/validate_market_maker.py [--data_dir data/] [--output_dir output/test_20251201_all] \
      [--date 2025-12-01] [--out output/mm_validation/]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.utils import load_alpha_series, load_lambda_series

# -- Constants ----

TRADE_FMT = "<qIB3xqII"  # int64 ts_ns, uint32 inst_id, uint8 action, 3x pad, int64 price_1e8, uint32 size, uint32 pad
TRADE_SZ  = struct.calcsize(TRADE_FMT)
TICK_NS   = 30_000_000_000  # 30 seconds in nanoseconds
BETA      = 0.05            # must match pipeline config

# -- Data loading ----

def load_trades(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        buf = f.read()
    n = len(buf) // TRADE_SZ
    records = [struct.unpack_from(TRADE_FMT, buf, i * TRADE_SZ) for i in range(n)]
    df = pd.DataFrame(records, columns=["ts_ns", "inst_id", "action", "price_1e8", "size", "_pad"])
    df["price"] = df["price_1e8"] / 1e8
    return df[["ts_ns", "inst_id", "price", "size"]]


def bin_trades(trades: pd.DataFrame, tick_ns: int = TICK_NS) -> pd.DataFrame:
    """Bin trades into (tick, inst_id) buckets -- count trades and compute VWAP."""
    df = trades.copy()
    df["tick"] = df["ts_ns"] // tick_ns
    df["dollar_vol"] = df["price"] * df["size"]
    grp = df.groupby(["tick", "inst_id"]).agg(
        count=("size", "sum"),
        dollar_vol=("dollar_vol", "sum"),
        total_size=("size", "sum"),
    ).reset_index()
    grp["vwap"] = grp["dollar_vol"] / grp["total_size"].replace(0, np.nan)
    return grp[["tick", "inst_id", "count", "vwap"]]


def load_alpha_timestamps(output_dir: Path, date: str):
    """Return sorted list of tick timestamps (in ns) from alpha filenames."""
    paths = sorted(
        glob.glob(str(output_dir / date / "alpha_*.bin")),
        key=lambda p: int(re.search(r"alpha_(\d+)\.bin", p).group(1)),
    )
    return [int(re.search(r"alpha_(\d+)\.bin", p).group(1)) for p in paths]


def build_count_matrix(binned: pd.DataFrame, tick_ids: list[int], n_inst: int) -> np.ndarray:
    """
    Returns count matrix C of shape (S, N) where S = len(tick_ids), N = n_inst.
    C[s, i] = number of trades for instrument i during the tick starting at tick_ids[s].
    """
    tick_to_s = {t // TICK_NS: s for s, t in enumerate(tick_ids)}
    C = np.zeros((len(tick_ids), n_inst), dtype=float)
    for _, row in binned.iterrows():
        s = tick_to_s.get(int(row["tick"]))
        i = int(row["inst_id"])
        if s is not None and 0 <= i < n_inst:
            C[s, i] = row["count"]
    return C


def build_vwap_matrix(binned: pd.DataFrame, tick_ids: list[int], n_inst: int) -> np.ndarray:
    """Returns VWAP matrix V of shape (S, N); NaN where no trades."""
    tick_to_s = {t // TICK_NS: s for s, t in enumerate(tick_ids)}
    V = np.full((len(tick_ids), n_inst), np.nan)
    for _, row in binned.iterrows():
        s = tick_to_s.get(int(row["tick"]))
        i = int(row["inst_id"])
        if s is not None and 0 <= i < n_inst and not np.isnan(row["vwap"]):
            V[s, i] = row["vwap"]
    return V


# -- Helpers ----

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# -- Experiment 1: Intensity Forecasting ----

def run_exp1(C: np.ndarray, alphas: np.ndarray, id_map: dict,
             lambdas: np.ndarray | None = None) -> pd.DataFrame:
    """
    Predict next-tick trade count Y_i(s+1) from signals at tick s.
    Signals:
      - Hawkes: live lambda_i(s) * 30  when lambdas available (events/sec * 30s tick)
                or alpha_ii(s) * C[s,i] as fallback
      - EWMA-0.9: exponentially smoothed past count
      - Rolling mean of last 5 ticks
    Eval on OOS second half (s > S//2).
    """
    S, N = C.shape
    split = S // 2
    use_live = lambdas is not None and lambdas.shape == (S, N)
    rows = []

    for i in range(N):
        Y = C[1:, i]       # target: count at tick s+1, s=0..S-2
        Xh = np.zeros(S - 1)
        Xe = np.zeros(S - 1)
        Xr = np.zeros(S - 1)

        ewma = C[0, i]
        roll_buf = list(C[:5, i])
        for s in range(S - 1):
            Xh[s] = lambdas[s, i] * 30 if use_live else alphas[s, i, i] * C[s, i]
            Xe[s] = ewma
            Xr[s] = np.mean(roll_buf[-5:] if len(roll_buf) >= 5 else roll_buf)
            ewma = 0.9 * ewma + 0.1 * C[s, i]
            roll_buf.append(C[s, i])

        Y_oos  = Y[split:]
        Xh_oos = Xh[split:]
        Xe_oos = Xe[split:]
        Xr_oos = Xr[split:]

        # Scale Hawkes with IS linear fit so units match
        if np.std(Xh[:split]) > 1e-9:
            slope, intercept, *_ = stats.linregress(Xh[:split], Y[:split])
            Xh_pred = slope * Xh_oos + intercept
        else:
            Xh_pred = np.full_like(Y_oos, Y[:split].mean())

        rows.append({
            "inst":        id_map.get(str(i), str(i)),
            "rmse_hawkes": rmse(Y_oos, Xh_pred),
            "rmse_ewma":   rmse(Y_oos, Xe_oos),
            "rmse_roll":   rmse(Y_oos, Xr_oos),
            "r_hawkes":    float(np.corrcoef(Y_oos, Xh_oos)[0, 1]) if np.std(Xh_oos) > 1e-9 else 0.0,
            "r_ewma":      float(np.corrcoef(Y_oos, Xe_oos)[0, 1]) if np.std(Xe_oos) > 1e-9 else 0.0,
        })

    return pd.DataFrame(rows)


# -- Experiment 2: Adverse Selection Signal ----

def run_exp2(C: np.ndarray, V: np.ndarray, alphas: np.ndarray, id_map: dict,
             lambdas: np.ndarray | None = None) -> pd.DataFrame:
    """
    Correlate intensity signals at tick s with realized |return| at tick s+1.
    Signals:
      - Hawkes: live lambda_i(s) when lambdas available, else alpha_ii(s) * C[s,i]
      - Count:  C[s, i]
      - EWMA:   exponentially smoothed past count
    """
    S, N = C.shape
    use_live = lambdas is not None and lambdas.shape == (S, N)
    rows = []

    with np.errstate(invalid="ignore", divide="ignore"):
        R_abs = np.abs(np.diff(V, axis=0) / V[:-1]) * 1e4  # (S-1, N), bps

    for i in range(N):
        y = R_abs[:, i]
        valid = ~np.isnan(y)
        if valid.sum() < 30:
            continue

        ewma_arr   = np.zeros(S - 1)
        hawkes_arr = np.zeros(S - 1)
        ewma_val   = C[0, i]
        for s in range(S - 1):
            hawkes_arr[s] = lambdas[s, i] if use_live else alphas[s, i, i] * C[s, i]
            ewma_arr[s]   = ewma_val
            ewma_val = 0.9 * ewma_val + 0.1 * C[s, i]

        count_arr = C[:S-1, i]

        def safe_corr(x):
            mask = valid & (np.std(x[valid]) > 1e-9)
            if mask.sum() < 10:
                return np.nan
            return float(np.corrcoef(y[mask], x[mask])[0, 1])

        rows.append({
            "inst":         id_map.get(str(i), str(i)),
            "corr_hawkes":  safe_corr(hawkes_arr),
            "corr_count":   safe_corr(count_arr),
            "corr_ewma":    safe_corr(ewma_arr),
            "mean_abs_ret": float(np.nanmean(y)),
        })

    return pd.DataFrame(rows)


# -- Experiment 3: Cross-Stock Causal Prediction ----

def run_exp3(C: np.ndarray, alphas: np.ndarray, id_map: dict) -> pd.DataFrame:
    """
    For each instrument i, predict count_i(s+1) using cross-stock terms.
    Hawkes cross signal: sum_j alpha_ij(s) * C[s,j], j != i
    Corr baseline:       sum_j corr_ij * C[s,j], j != i  (corr from first half)
    Self-only:           alpha_ii(s) * C[s,i]
    Metric: OOS R^2 of (self + cross) over second half.
    """
    S, N = C.shape
    split = S // 2
    rows = []

    # Compute pairwise count correlations on IS half
    C_is = C[:split]
    corr_mat = np.corrcoef(C_is.T)  # (N, N)
    corr_mat = np.where(np.isnan(corr_mat), 0.0, corr_mat)

    Y = C[1:, :]      # shape (S-1, N), target at s+1

    for i in range(N):
        Y_i = Y[split:, i]
        if np.std(Y_i) < 1e-9:
            continue

        # Build signals for s = 0..S-2
        hawkes_cross = np.zeros(S - 1)
        corr_cross   = np.zeros(S - 1)
        hawkes_self  = np.zeros(S - 1)

        for s in range(S - 1):
            a_row = alphas[s, i, :]
            hawkes_self[s]  = a_row[i] * C[s, i]
            hawkes_cross[s] = sum(a_row[j] * C[s, j] for j in range(N) if j != i)
            corr_cross[s]   = sum(corr_mat[i, j] * C[s, j] for j in range(N) if j != i)

        # IS fit for Hawkes (self + cross)
        X_h_is = np.column_stack([hawkes_self[:split], hawkes_cross[:split]])
        X_c_is = np.column_stack([hawkes_self[:split], corr_cross[:split]])
        Y_is   = Y[:split, i]

        def ols_predict(X_is, X_oos, y_is):
            try:
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(y_is)), X_is]),
                    y_is, rcond=None
                )
                pred = coef[0] + X_oos @ coef[1:]
                return pred
            except Exception:
                return np.full(len(X_oos), y_is.mean())

        X_h_oos = np.column_stack([hawkes_self[split:], hawkes_cross[split:]])
        X_c_oos = np.column_stack([hawkes_self[split:], corr_cross[split:]])

        pred_h = ols_predict(X_h_is, X_h_oos, Y_is)
        pred_c = ols_predict(X_c_is, X_c_oos, Y_is)
        pred_self = ols_predict(
            hawkes_self[:split].reshape(-1, 1),
            hawkes_self[split:].reshape(-1, 1), Y_is
        )

        rows.append({
            "inst":             id_map.get(str(i), str(i)),
            "r2_hawkes_cross":  oos_r2(Y_i, pred_h),
            "r2_corr_cross":    oos_r2(Y_i, pred_c),
            "r2_self_only":     oos_r2(Y_i, pred_self),
        })

    return pd.DataFrame(rows)


# -- Experiment 4: Structural Break Detection ----

def run_exp4(C: np.ndarray, V: np.ndarray, alphas: np.ndarray) -> dict:
    """
    Compare Hawkes-jump detector vs. rolling-std baseline for detecting
    high-volatility regime ticks. Target: top-decile realized vol ticks.
    """
    S, _ = C.shape

    # Realized vol at each tick: mean absolute return across all instruments
    with np.errstate(invalid="ignore", divide="ignore"):
        R_abs = np.abs(np.diff(V, axis=0) / V[:-1]) * 1e4  # (S-1, N)
    realized_vol = np.nanmean(R_abs, axis=1)  # (S-1,)

    # True high-vol ticks: top decile
    thresh_vol = np.nanpercentile(realized_vol, 90)
    y_true = (realized_vol >= thresh_vol).astype(int)

    # Alpha Frobenius norm series
    frob = np.array([np.linalg.norm(alphas[s], "fro") for s in range(S)])

    # Hawkes detector: flag tick s if ||alpha_hat(s) - alpha_hat(s-1)||_F > 2sigma of jumps
    jumps = np.abs(np.diff(frob))          # shape (S-1,)
    jump_thresh = jumps.mean() + 2 * jumps.std()
    y_hawkes = (jumps >= jump_thresh).astype(int)

    # Baseline detector: flag tick s if frob(s) > rolling_mean + 2*rolling_std
    window = 10
    y_baseline = np.zeros(S - 1, dtype=int)
    for s in range(S - 1):
        lo = max(0, s - window)
        local = frob[lo:s]
        if len(local) >= 3:
            if frob[s] > local.mean() + 2 * local.std():
                y_baseline[s] = 1

    def f1(y_t, y_p):
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_s = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        return prec, rec, f1_s

    prec_h, rec_h, f1_h = f1(y_true, y_hawkes)
    prec_b, rec_b, f1_b = f1(y_true, y_baseline)

    return {
        "f1_hawkes":      round(f1_h, 4),
        "f1_baseline":    round(f1_b, 4),
        "precision_h":    round(prec_h, 4),
        "recall_h":       round(rec_h, 4),
        "precision_b":    round(prec_b, 4),
        "recall_b":       round(rec_b, 4),
        "n_true_breaks":  int(y_true.sum()),
        "n_hawkes_flags": int(y_hawkes.sum()),
        "n_base_flags":   int(y_baseline.sum()),
    }


# -- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/")
    parser.add_argument("--output_dir", default="output/test_20251201_all")
    parser.add_argument("--date",       default="2025-12-01")
    parser.add_argument("--out",        default="output/mm_validation/")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load ID map
    id_map_path = Path(args.data_dir) / "test_20251201_allstock_id_map.json"
    with open(id_map_path) as f:
        id_map = json.load(f)
    N = len(id_map)

    print(f"Loading trades from {args.data_dir}...")
    trades = load_trades(Path(args.data_dir) / "test_20251201_allstock_trades.bin")
    binned = bin_trades(trades)
    print(f"  Total trade records: {len(trades):,}")

    print(f"Loading alpha series from {args.output_dir}/{args.date}/...")
    alphas = load_alpha_series(Path(args.output_dir), args.date)
    tick_ids = load_alpha_timestamps(Path(args.output_dir), args.date)
    S = len(tick_ids)
    print(f"  {S} ticks, alpha shape: {alphas.shape}")

    C = build_count_matrix(binned, tick_ids, N)
    V = build_vwap_matrix(binned, tick_ids, N)
    print(f"  Count matrix shape: {C.shape}  (mean total trades/tick: {C.sum(1).mean():.1f})")

    print(f"Loading lambda series from {args.output_dir}/{args.date}/...")
    lambdas = load_lambda_series(Path(args.output_dir), args.date)
    has_lambda = lambdas.ndim == 2 and lambdas.shape == (S, N)
    if has_lambda:
        print(f"  lambda shape: {lambdas.shape}  "
              f"mean lambda_AAPL={lambdas[:, 0].mean():.3f}  "
              f"mean lambda_NVDA={lambdas[:, 11].mean():.3f}  "
              f"mean lambda_TSLA={lambdas[:, 15].mean():.3f}")
    else:
        print("  No lambda files found -- falling back to alpha_ii * count")
        lambdas = None

    # -- Experiment 1 ----
    print("\n=== Experiment 1: Intensity Forecasting (next-30s trade count) ===")
    print(f"  Hawkes signal: {'live lambda * 30s' if has_lambda else 'alpha_ii * count (fallback)'}")
    df1 = run_exp1(C, alphas, id_map, lambdas)
    df1["rmse_hawkes_vs_ewma"] = df1["rmse_hawkes"] - df1["rmse_ewma"]
    n_better = (df1["rmse_hawkes"] < df1["rmse_ewma"]).sum()
    print(df1[["inst", "rmse_hawkes", "rmse_ewma", "rmse_roll", "r_hawkes", "r_ewma"]].to_string(index=False))
    print(f"\nHawkes RMSE < EWMA RMSE for {n_better}/{N} instruments")
    print(f"Mean Hawkes r = {df1['r_hawkes'].mean():.4f},  Mean EWMA r = {df1['r_ewma'].mean():.4f}")
    df1.to_csv(out_path / "exp1_intensity_forecast.csv", index=False)

    # -- Experiment 2 ----
    print("\n=== Experiment 2: Adverse Selection Signal vs. |Return| ===")
    print(f"  Hawkes signal: {'live lambda (events/sec)' if has_lambda else 'alpha_ii * count (fallback)'}")
    df2 = run_exp2(C, V, alphas, id_map, lambdas)
    print(df2[["inst", "corr_hawkes", "corr_count", "corr_ewma", "mean_abs_ret"]].to_string(index=False))
    better = (df2["corr_hawkes"] > df2["corr_count"]).sum()
    print(f"\nHawkes corr > Count corr for {better}/{len(df2)} instruments")
    print(f"Mean Hawkes corr = {df2['corr_hawkes'].mean():.4f},  Mean count corr = {df2['corr_count'].mean():.4f}")
    df2.to_csv(out_path / "exp2_adverse_selection.csv", index=False)

    # -- Experiment 3 ----
    print("\n=== Experiment 3: Cross-Stock Causal Prediction ===")
    df3 = run_exp3(C, alphas, id_map)
    print(df3[["inst", "r2_hawkes_cross", "r2_corr_cross", "r2_self_only"]].to_string(index=False))
    better3 = (df3["r2_hawkes_cross"] > df3["r2_corr_cross"]).sum()
    print(f"\nHawkes cross R^2 > Corr cross R^2 for {better3}/{len(df3)} instruments")
    print(f"Mean Hawkes cross R^2 = {df3['r2_hawkes_cross'].mean():.4f}")
    print(f"Mean Corr cross  R^2 = {df3['r2_corr_cross'].mean():.4f}")
    print(f"Mean Self-only   R^2 = {df3['r2_self_only'].mean():.4f}")
    df3.to_csv(out_path / "exp3_cross_stock.csv", index=False)

    # -- Experiment 4 ----
    print("\n=== Experiment 4: Structural Break Detection vs. Vol Spikes ===")
    res4 = run_exp4(C, V, alphas)
    print(f"  True high-vol ticks (top 10%): {res4['n_true_breaks']}")
    print(f"  Hawkes detector flags:          {res4['n_hawkes_flags']}")
    print(f"  Baseline detector flags:        {res4['n_base_flags']}")
    print(f"  Hawkes   -- Precision: {res4['precision_h']:.4f}  Recall: {res4['recall_h']:.4f}  F1: {res4['f1_hawkes']:.4f}")
    print(f"  Baseline -- Precision: {res4['precision_b']:.4f}  Recall: {res4['recall_b']:.4f}  F1: {res4['f1_baseline']:.4f}")
    pd.DataFrame([res4]).to_csv(out_path / "exp4_structural_breaks.csv", index=False)

    print(f"\nResults saved to {out_path}/")


if __name__ == "__main__":
    main()
