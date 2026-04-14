"""
Multi-Day Quote-Width Market Maker Backtest
============================================
IS period : October + November 2025  (calibrate k per instrument)
OOS period: December 2025            (held-out evaluation)

Same 9-policy comparison as backtest_quote_width.py.

For each day we load:
  - data/lob/<YYYYMMDD>_allstock_lob.bin   (LOB binary with BBO)
  - output/multiday/<YYYYMMDD>/<date>/lambda_*.bin  (Hawkes lambda)
  - output/multiday/<YYYYMMDD>/<date>/alpha_*.bin   (alpha matrices)

Usage:
    python scripts/backtest_multiday.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.utils import load_lambda_series, load_alpha_series

# -- Constants ----

LOB_FMT     = "<qIBBxxqqqI4x"
LOB_SZ      = struct.calcsize(LOB_FMT)
TICK_NS     = 30_000_000_000
FIXED_SCALE = 1_000_000_000
EWMA_DECAY  = 0.9
HAWKES_BETA = 0.05
HAWKES_DT   = 30.0
CROSS_IN_THRESHOLD = 0.5   # instruments with IS mean cross_in > this get pure TOD in Hawkes-select

US_HOLIDAYS = {
    datetime.date(2025, 11, 27),
    datetime.date(2025, 11, 28),
    datetime.date(2025, 12, 25),
}


def trading_days(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    days, d = [], start
    while d < end:
        if d.weekday() < 5 and d not in US_HOLIDAYS:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


# -- LOB loading ----

def load_lob(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        buf = f.read()
    n = len(buf) // LOB_SZ
    if n == 0:
        return pd.DataFrame(columns=["ts_ns","inst_id","mid_d","price_d",
                                      "quoted_hs","is_buy","size"])
    recs = [struct.unpack_from(LOB_FMT, buf, i * LOB_SZ) for i in range(n)]
    df = pd.DataFrame(recs, columns=["ts_ns","inst_id","side","_pad",
                                      "price","best_bid","best_ask","size"])
    df["mid_d"]     = (df["best_bid"] + df["best_ask"]) / 2.0 / FIXED_SCALE
    df["price_d"]   = df["price"] / FIXED_SCALE
    df["quoted_hs"] = (df["best_ask"] - df["best_bid"]) / 2.0 / FIXED_SCALE
    df["is_buy"]    = (df["side"] == 65).astype(float)
    return df[["ts_ns","inst_id","mid_d","price_d","quoted_hs","is_buy","size"]]


def load_tick_ids(output_dir: Path, date: str,
                  session_filter: bool = False) -> list[int]:
    date_dir = output_dir / date
    if not date_dir.exists():
        return []
    paths = sorted(date_dir.glob("alpha_*.bin"),
                   key=lambda p: int(re.search(r"alpha_(\d+)\.bin", p.name).group(1)))
    if session_filter:
        from analysis.utils import _filter_session
        paths = _filter_session(paths, date)
    return [int(re.search(r"alpha_(\d+)\.bin", p.name).group(1)) for p in paths]


def aggregate_to_ticks(lob: pd.DataFrame, tick_ids: list[int],
                       n_inst: int) -> dict[str, np.ndarray]:
    S = len(tick_ids)
    tick_to_s = {t // TICK_NS: s for s, t in enumerate(tick_ids)}

    mid   = np.full((S, n_inst), np.nan)
    qhs   = np.full((S, n_inst), np.nan)
    count = np.zeros((S, n_inst))
    vol   = np.zeros((S, n_inst))
    imbal = np.zeros((S, n_inst))

    lob_c = lob.copy()
    lob_c["tick_bin"] = lob_c["ts_ns"] // TICK_NS

    for (tb, inst_id), grp in lob_c.groupby(["tick_bin","inst_id"]):
        s = tick_to_s.get(tb)
        i = int(inst_id)
        if s is None or i < 0 or i >= n_inst:
            continue
        n = len(grp)
        mid[s, i]   = grp["mid_d"].iloc[0]
        qhs[s, i]   = grp["quoted_hs"].median()
        count[s, i] = n
        vol[s, i]   = grp["price_d"].std() if n >= 2 else 0.0
        buys        = grp["is_buy"].sum()
        imbal[s, i] = abs(buys - (n - buys)) / n

    return {
        "mid":   mid,
        "qhs":   np.nan_to_num(qhs, nan=0.0),
        "count": count,
        "vol":   vol,
        "imbal": imbal,
    }


# -- Signal builders ----

def build_ewma_count(count: np.ndarray) -> np.ndarray:
    S, N = count.shape
    out = np.zeros((S, N))
    v   = count[0].copy()
    for s in range(S):
        out[s] = v
        v = EWMA_DECAY * v + (1 - EWMA_DECAY) * count[s]
    return out


def build_ewma_vol(mid: np.ndarray) -> np.ndarray:
    S, N = mid.shape
    dmid = np.abs(np.diff(mid, axis=0, prepend=mid[:1]))
    out  = np.zeros((S, N))
    v    = dmid[0].copy()
    for s in range(S):
        out[s] = v
        v = EWMA_DECAY * v + (1 - EWMA_DECAY) * dmid[s]
    return out


def build_self_hawkes(count: np.ndarray, alpha_series: np.ndarray,
                      lambdas: np.ndarray, n_is_ticks: int) -> np.ndarray:
    """Diagonal-only Hawkes intensity (no cross terms)."""
    S, N = count.shape
    decay = np.exp(-HAWKES_BETA * HAWKES_DT)

    if alpha_series.shape[0] >= n_is_ticks:
        alpha_ii = np.array([alpha_series[:n_is_ticks, i, i].mean() for i in range(N)])
    else:
        alpha_ii = np.zeros(N)
    alpha_ii = np.clip(alpha_ii, 0.0, 0.99)

    mu = lambdas[:n_is_ticks].mean(axis=0) * (1.0 - alpha_ii)
    out = np.zeros((S, N))
    R   = np.zeros(N)
    for s in range(S):
        R      = decay * R + count[s]
        out[s] = mu + alpha_ii * HAWKES_BETA * R
    return out


def build_tod(S: int, N: int) -> np.ndarray:
    s_idx = np.arange(S, dtype=float)
    tod1d = 1.0 + 0.5 * np.cos(2.0 * np.pi * s_idx / S)
    return np.tile(tod1d[:, np.newaxis], (1, N))


def build_hawkes_cross(count: np.ndarray, alpha_series: np.ndarray) -> np.ndarray:
    """Cross-instrument excitation only: sum_{j != i} alpha[s,i,j] * count[s-1,j].
    Isolates the unique StreamCause causal-graph contribution."""
    S, N = count.shape
    A    = alpha_series.shape[0]
    out  = np.zeros((S, N))
    for s in range(1, S):
        prev  = count[s - 1]                          # (N,)
        alpha = alpha_series[min(s, A - 1)]           # (N, N)  alpha[i,j]
        cross = alpha @ prev                           # (N,) = sum_j alpha[i,j]*count[j]
        cross -= alpha.diagonal() * prev              # subtract self term
        out[s] = np.maximum(cross, 0.0)
    return out


def build_hawkes_tod_product(lam_norm: np.ndarray) -> np.ndarray:
    """TOD * normalised-Hawkes: TOD stability combined with Hawkes excess intensity."""
    S, N = lam_norm.shape
    return build_tod(S, N) * lam_norm


# -- Policy evaluation ----

def sharpe(pnl: np.ndarray) -> float:
    if pnl.std() < 1e-12:
        return 0.0
    return float(pnl.mean() / pnl.std())


def backtest_policy(mm_hs: np.ndarray, qhs: np.ndarray,
                    count: np.ndarray, pi: np.ndarray) -> np.ndarray:
    filled = mm_hs <= qhs
    return np.where(filled, count * mm_hs - pi, 0.0)


def calibrate_k(signal: np.ndarray, qhs: np.ndarray,
                count: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Vectorised grid search k per instrument on IS arrays. Returns (N,)."""
    ks_grid = np.logspace(-7, 1, 150)   # (K,)
    N = signal.shape[1]
    ks = np.zeros(N)
    for i in range(N):
        sig_i   = signal[:, i]          # (T,)
        qhs_i   = qhs[:, i]            # (T,)
        count_i = count[:, i]           # (T,)
        pi_i    = pi[:, i]             # (T,)
        # mm_hs for all k at once: (T, K)
        mm = ks_grid[np.newaxis, :] * sig_i[:, np.newaxis]
        filled = mm <= qhs_i[:, np.newaxis]
        pnl = np.where(filled,
                       count_i[:, np.newaxis] * mm - pi_i[:, np.newaxis],
                       0.0)             # (T, K)
        means = pnl.mean(axis=0)
        stds  = pnl.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sr = np.where(stds > 1e-12, means / stds, 0.0)
        ks[i] = ks_grid[sr.argmax()]
    return ks


# -- Per-day data loader ----

def load_day(date: datetime.date, lob_dir: Path, pipeline_root: Path,
             n_inst: int, session_filter: bool = False) -> dict | None:
    date_str     = date.strftime("%Y-%m-%d")
    date_compact = date.strftime("%Y%m%d")

    lob_path   = lob_dir / f"{date_compact}_allstock_lob.bin"
    output_dir = pipeline_root / date_compact

    if not lob_path.exists() or lob_path.stat().st_size == 0:
        return None
    tick_ids = load_tick_ids(output_dir, date_str, session_filter=session_filter)
    if not tick_ids:
        return None
    lambdas = load_lambda_series(output_dir, date_str, session_filter=session_filter)
    if lambdas.shape[0] == 0:
        return None
    alpha_series = load_alpha_series(output_dir, date_str, session_filter=session_filter)

    lob = load_lob(str(lob_path))
    tks = aggregate_to_ticks(lob, tick_ids, n_inst)

    S = len(tick_ids)
    pi = np.zeros_like(tks["mid"])
    pi[:-1] = np.abs(np.diff(tks["mid"], axis=0))
    pi = np.nan_to_num(pi, nan=0.0)

    if lambdas.shape != (S, n_inst):
        return None

    return {
        "date":         date_str,
        "S":            S,
        "mid":          tks["mid"],
        "qhs":          tks["qhs"],
        "count":        tks["count"],
        "vol":          tks["vol"],
        "imbal":        tks["imbal"],
        "price_impact": pi,
        "lambdas":      lambdas,
        "alpha_series": alpha_series,
        "tick_ids":     np.array(tick_ids),
    }


# -- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lob_dir",       default="data/lob")
    parser.add_argument("--pipeline_root", default="output/multiday")
    parser.add_argument("--id_map",        default="data/allstock_id_map.json")
    parser.add_argument("--out",           default="output/mm_backtest_multiday")
    parser.add_argument("--is_end",        default="2025-12-01")
    parser.add_argument("--start",         default="2025-10-01")
    parser.add_argument("--end",           default="2026-01-01")
    parser.add_argument("--session_filter", action="store_true",
                        help="Filter pipeline output to ET session ticks only "
                             "(needed when pipeline ran on a multi-day file)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(args.id_map) as f:
        id_map = json.load(f)
    N = len(id_map)

    start_date  = datetime.date.fromisoformat(args.start)
    end_date    = datetime.date.fromisoformat(args.end)
    is_end_date = datetime.date.fromisoformat(args.is_end)
    all_days    = trading_days(start_date, end_date)

    lob_dir       = Path(args.lob_dir)
    pipeline_root = Path(args.pipeline_root)

    print(f"Loading data for {len(all_days)} trading days...")
    is_days, oos_days = [], []
    for day in all_days:
        d = load_day(day, lob_dir, pipeline_root, N,
                     session_filter=args.session_filter)
        if d is None:
            print(f"  {day}: missing, skip")
            continue
        if day < is_end_date:
            is_days.append(d)
        else:
            oos_days.append(d)
        print(f"  {d['date']}: {d['S']} ticks  ok")

    print(f"\nIS: {len(is_days)} days  |  OOS: {len(oos_days)} days")
    if not is_days or not oos_days:
        print("ERROR: insufficient data"); sys.exit(1)

    # -- Stack IS matrices ----
    def cat(key): return np.concatenate([d[key] for d in is_days], axis=0)

    is_lam   = cat("lambdas")
    is_alpha = np.concatenate([d["alpha_series"] for d in is_days], axis=0)
    is_count = cat("count")
    is_qhs   = cat("qhs")
    is_pi    = cat("price_impact")
    is_mid   = cat("mid")
    is_vol   = cat("vol")
    is_imbal = cat("imbal")
    n_is     = is_lam.shape[0]

    # IS fixed spread per instrument (use full concatenated qhs -- fine for median)
    is_fixed_hs = np.nanmedian(is_qhs, axis=0)

    # TOD baseline for Hawkes normalisation: mean lambda at each intraday tick position.
    # IS days may differ by 1 tick (early closes); truncate to min S before stacking.
    S_per_day = min(d["S"] for d in is_days)
    is_lam_3d  = np.stack([d["lambdas"][:S_per_day] for d in is_days], axis=0)  # (D, S, N)
    is_tod_lam = np.maximum(is_lam_3d.mean(axis=0), 1e-10)                       # (S, N)
    # Build normalised IS signal aligned with the concatenated IS array.
    # Re-concatenate using only the first S_per_day ticks of each IS day so
    # shapes match is_qhs / is_count / is_pi (which were already concatenated).
    is_lam_norm = np.concatenate(
        [d["lambdas"][:S_per_day] / is_tod_lam for d in is_days], axis=0
    )
    # Truncate the other IS arrays to match (drop any extra ticks on long days)
    is_lam_tr    = np.concatenate([d["lambdas"][:S_per_day]       for d in is_days], axis=0)
    is_self_tr   = np.concatenate([build_self_hawkes(
                       d["count"][:S_per_day], d["alpha_series"],
                       d["lambdas"][:S_per_day], S_per_day)        for d in is_days], axis=0)
    is_cross_tr  = np.concatenate([build_hawkes_cross(
                       d["count"][:S_per_day], d["alpha_series"])  for d in is_days], axis=0)
    is_qhs_tr    = np.concatenate([d["qhs"][:S_per_day]           for d in is_days], axis=0)
    is_count_tr  = np.concatenate([d["count"][:S_per_day]         for d in is_days], axis=0)
    is_pi_tr     = np.concatenate([d["price_impact"][:S_per_day]  for d in is_days], axis=0)
    is_mid_tr    = np.concatenate([d["mid"][:S_per_day]           for d in is_days], axis=0)
    is_vol_tr    = np.concatenate([d["vol"][:S_per_day]           for d in is_days], axis=0)
    is_imbal_tr  = np.concatenate([d["imbal"][:S_per_day]         for d in is_days], axis=0)
    n_is_tr      = is_lam_tr.shape[0]
    is_ewma_c_tr = build_ewma_count(is_count_tr)
    is_ewma_v_tr = build_ewma_vol(is_mid_tr)
    is_tod_tr    = build_tod(n_is_tr, N)
    is_htod_tr   = build_hawkes_tod_product(is_lam_norm)

    # -- Alpha cap + instrument selection from IS alpha matrices ----
    # Stack IS alpha series to find 99.9th-pct cap (guards against half-day EM divergence)
    is_alpha_stacked = np.concatenate(
        [d["alpha_series"][:S_per_day] for d in is_days], axis=0)   # (n_is_tr, N, N)
    alpha_cap = np.percentile(is_alpha_stacked, 99.9)
    print(f"\nIS alpha 99.9th-pct cap: {alpha_cap:.3f}  (clips OOS outliers like Dec 24)")

    # Per-instrument IS mean incoming cross-excitation: sum_{j!=i} mean_alpha[i,j]
    is_mean_alpha = is_alpha_stacked.mean(axis=0)                    # (N, N)
    cross_in_IS   = is_mean_alpha.sum(axis=1) - is_mean_alpha.diagonal()  # (N,)
    use_floor     = cross_in_IS <= CROSS_IN_THRESHOLD                # True -> Hawkes-floor
    print("\nInstrument selection (cross_in_IS threshold={:.1f}):".format(CROSS_IN_THRESHOLD))
    for i in range(N):
        tag = "Hawkes-floor" if use_floor[i] else "TOD          "
        print(f"  {id_map[str(i)]:6s}  cross_in={cross_in_IS[i]:.3f}  -> {tag}")

    # -- Calibrate on IS ----
    print(f"\nCalibrating on IS ({n_is_tr} ticks from {len(is_days)} days, S_per_day={S_per_day}):")

    def cal(name, sig):
        print(f"  {name}...", end=" ", flush=True)
        k = calibrate_k(sig, is_qhs_tr, is_count_tr, is_pi_tr)
        print(f"[{k.min():.2e}, {k.max():.2e}]")
        return k

    k_h    = cal("Hawkes-full ", is_lam_tr)
    k_hn   = cal("Hawkes-norm ", is_lam_norm)
    k_hx   = cal("Hawkes-cross", is_cross_tr)
    k_ht   = cal("Hawkes-TOD  ", is_htod_tr)
    k_s    = cal("Hawkes-self ", is_self_tr)
    k_e    = cal("EWMA-count  ", is_ewma_c_tr)
    k_v    = cal("RealVol     ", is_vol_tr)
    k_ev   = cal("EWMA-vol    ", is_ewma_v_tr)
    k_im   = cal("Imbalance   ", is_imbal_tr)
    k_tod  = cal("TOD         ", is_tod_tr)

    # -- Hawkes-pull: calibrate sit-out threshold per instrument on IS ----
    print("  Hawkes-pull threshold...", end=" ", flush=True)
    ks_grid = np.logspace(-7, 1, 150)
    tau_pull = np.zeros(N)
    for i in range(N):
        lam_i = is_lam_tr[:, i]
        quantiles = np.quantile(lam_i, np.linspace(0.5, 1.0, 60))
        best_tau, best_sr = quantiles[0], -np.inf
        for q in quantiles:
            filled = (lam_i <= q) & (is_fixed_hs[i] <= is_qhs_tr[:, i])
            pnl_i = np.where(filled, is_count_tr[:, i] * is_fixed_hs[i] - is_pi_tr[:, i], 0.0)
            sr = sharpe(pnl_i)
            if sr > best_sr:
                best_sr, best_tau = sr, q
        tau_pull[i] = best_tau
    print(f"thresholds at [{np.percentile(is_lam_tr, 50, axis=0).mean():.3f}...]")

    # -- Hawkes-gate: calibrate sit-out threshold for queue-aware posting at best -
    # Strategy: post at market best bid/ask (queue priority) when lambda_i(s) <= tau_i,
    # sit out otherwise.  IS P&L proxy: n(s)*qhs(s) - pi(s) when active.
    print("  Hawkes-gate threshold...", end=" ", flush=True)
    tau_gate = np.zeros(N)
    for i in range(N):
        lam_i = is_lam_tr[:, i]
        qhs_i = is_qhs_tr[:, i]
        cnt_i = is_count_tr[:, i]
        pi_i  = is_pi_tr[:, i]
        quantiles = np.quantile(lam_i, np.linspace(0.3, 0.95, 66))
        best_tau, best_sr = quantiles[-1], -np.inf   # default: always active
        for q in quantiles:
            pnl_i = np.where(lam_i <= q, cnt_i * qhs_i - pi_i, 0.0)
            sr = sharpe(pnl_i)
            if sr > best_sr:
                best_sr, best_tau = sr, q
        tau_gate[i] = best_tau
    print(f"[{tau_gate.min():.3e}, {tau_gate.max():.3e}]")

    # -- Hawkes-rolling: pre-compute expanding-IS k_h for each OOS day ----
    print("  Hawkes-rolling (expanding IS)...")
    rolling_k_h = []
    acc_lam = [is_lam_tr];  acc_qhs = [is_qhs_tr]
    acc_cnt = [is_count_tr]; acc_pi  = [is_pi_tr]
    for d in oos_days:
        aL = np.concatenate(acc_lam); aQ = np.concatenate(acc_qhs)
        aC = np.concatenate(acc_cnt); aP = np.concatenate(acc_pi)
        k_roll = calibrate_k(aL, aQ, aC, aP)
        rolling_k_h.append(k_roll)
        S_d = min(d["S"], S_per_day)
        acc_lam.append(d["lambdas"][:S_d]); acc_qhs.append(d["qhs"][:S_d])
        acc_cnt.append(d["count"][:S_d]);   acc_pi.append(d["price_impact"][:S_d])
        print(f"    {d['date']}: {aL.shape[0]} IS ticks  kin[{k_roll.min():.2e},{k_roll.max():.2e}]", flush=True)

    # -- OOS evaluation ----
    print(f"\nEvaluating OOS ({len(oos_days)} days)...")

    policy_names = ["Hawkes-full","Hawkes-floor","Hawkes-select","Hawkes-pull","Hawkes-rolling",
                    "Hawkes-norm","Hawkes-cross","Hawkes-TOD",
                    "Hawkes-self","EWMA-count","RealVol","EWMA-vol",
                    "Imbalance","TOD","Fixed","Oracle"]
    col_abbr = {
        "Hawkes-full":    "Hfull",  "Hawkes-floor":   "Hfloor",
        "Hawkes-select":  "Hsel",   "Hawkes-pull":    "Hpull",
        "Hawkes-rolling": "Hroll",  "Hawkes-norm":    "Hnorm",
        "Hawkes-cross":   "Hcross", "Hawkes-TOD":     "Htod",
        "Hawkes-self":    "Hself",  "EWMA-count":     "EWMAc",
        "RealVol":        "RVol",   "EWMA-vol":       "EVol",
        "Imbalance":      "Imbel",  "TOD":            "TOD",
        "Fixed":          "Fixed",  "Oracle":         "Oracl",
    }

    day_rows  = []
    all_curve = []
    pnl_stacks   = {p: [] for p in policy_names}
    fills_stacks = {p: [] for p in policy_names}  # count of filled trades per (tick, inst)

    for d_idx, d in enumerate(oos_days):
        S, lam = d["S"], d["lambdas"]
        # Truncate OOS day to IS tick length if it has more ticks (e.g. DST boundary +/-1 tick).
        S = min(S, is_tod_lam.shape[0])
        lam      = lam[:S]
        al       = np.clip(d["alpha_series"][:S], 0, alpha_cap)  # cap outlier days (e.g. Dec 24)
        cnt      = d["count"][:S]
        qhs      = d["qhs"][:S]
        pi       = d["price_impact"][:S]
        mid      = d["mid"][:S]
        tick_ids = d["tick_ids"][:S]

        ewma_c   = build_ewma_count(cnt)
        ewma_v   = build_ewma_vol(mid)
        self_l   = build_self_hawkes(cnt, al, lam, S)
        cross_l  = build_hawkes_cross(cnt, al)
        tod      = build_tod(S, N)
        fixed    = np.tile(is_fixed_hs, (S, 1))
        lam_norm = lam / is_tod_lam[:S]   # (S,N)
        htod_l   = build_hawkes_tod_product(lam_norm)

        def mm(k, sig): return k[np.newaxis, :] * sig

        tod_sig   = mm(k_tod, tod)
        full_sig  = mm(k_h,   lam)
        floor_sig = np.maximum(tod_sig, full_sig)
        # Hawkes-select: Hawkes-floor for low-cross_in instruments, TOD for high-cross_in
        select_sig = np.where(use_floor[np.newaxis, :], floor_sig, tod_sig)
        sigs = {
            "Hawkes-full":    full_sig,
            "Hawkes-floor":   floor_sig,
            "Hawkes-select":  select_sig,
            "Hawkes-pull":    np.where(lam <= tau_pull[np.newaxis, :],
                                       is_fixed_hs[np.newaxis, :], np.inf),
            "Hawkes-rolling": mm(rolling_k_h[d_idx], lam),
            "Hawkes-norm":    mm(k_hn,  lam_norm),
            "Hawkes-cross":   mm(k_hx,  cross_l),
            "Hawkes-TOD":     mm(k_ht,  htod_l),
            "Hawkes-self":    mm(k_s,   self_l),
            "EWMA-count":     mm(k_e,   ewma_c),
            "RealVol":        mm(k_v,   d["vol"]),
            "EWMA-vol":       mm(k_ev,  ewma_v),
            "Imbalance":      mm(k_im,  d["imbal"]),
            "TOD":            tod_sig,
            "Fixed":          fixed,
        }

        # Oracle: best constant half-spread per instrument on this OOS day (cheating).
        ks_grid = np.logspace(-7, 1, 150)
        k_ora = np.zeros(N)
        for i in range(N):
            pnl_k = np.where(
                ks_grid[np.newaxis, :] <= qhs[:, i:i+1],
                cnt[:, i:i+1] * ks_grid[np.newaxis, :] - pi[:, i:i+1],
                0.0)                    # (S, K)
            means = pnl_k.mean(axis=0)
            stds  = pnl_k.std(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                sr = np.where(stds > 1e-12, means / stds, 0.0)
            k_ora[i] = ks_grid[sr.argmax()]
        ones = np.ones((S, N))
        sigs["Oracle"] = mm(k_ora, ones)

        sl = slice(0, S-1)  # exclude last tick (no price_impact)
        pnls_day  = {p: backtest_policy(sigs[p][sl], qhs[sl], cnt[sl], pi[sl])
                     for p in policy_names}
        fills_day = {p: np.where(sigs[p][sl] <= qhs[sl], cnt[sl], 0.0)
                     for p in policy_names}

        for p in policy_names:
            pnl_stacks[p].append(pnls_day[p])
            fills_stacks[p].append(fills_day[p])

        ts = tick_ids[sl]
        curve_row = {"ts_ns": ts, "date": d["date"]}
        for p in policy_names:
            curve_row[f"tickPnL_{col_abbr[p]}"] = pnls_day[p].sum(axis=1)
        all_curve.append(pd.DataFrame(curve_row))

        for p in policy_names:
            tp = pnls_day[p].sum(axis=1)
            day_rows.append({
                "date":      d["date"],
                "policy":    p,
                "total_pnl": tp.sum(),
                "sharpe":    sharpe(tp),
                "fill_rate": ((pnls_day[p] != 0).sum(axis=1) > 0).mean(),
                "win_rate":  (tp > 0).mean(),
                "n_ticks":   len(tp),
            })

    # -- Aggregate summaries ----
    df_days = pd.DataFrame(day_rows)
    df_curve = pd.concat(all_curve, ignore_index=True).sort_values("ts_ns")
    for p in policy_names:
        df_curve[f"cumPnL_{col_abbr[p]}"] = df_curve[f"tickPnL_{col_abbr[p]}"].cumsum()

    print("\n=== OOS Results (all days, aggregate across instruments) ===")
    agg = (df_days.groupby("policy")
           .agg(total_pnl=("total_pnl","sum"),
                mean_daily_pnl=("total_pnl","mean"),
                sharpe_of_daily=("total_pnl", lambda x: sharpe(x.values)),
                mean_fill=("fill_rate","mean"),
                n_days=("date","count"))
           .reset_index()
           .sort_values("sharpe_of_daily", ascending=False))
    print(agg.to_string(index=False, float_format="{:.4f}".format))

    # Per-instrument OOS Sharpe (stack all OOS ticks)
    print("\n=== Per-Instrument OOS Sharpe (stacked OOS) ===")
    stacked = {p: np.concatenate(pnl_stacks[p], axis=0) for p in policy_names}
    inst_rows = []
    for i in range(N):
        sym = id_map.get(str(i), str(i))
        row = {"inst": sym}
        for p in policy_names:
            row[f"sh_{col_abbr[p]}"] = sharpe(stacked[p][:, i])
        row["fill_H"] = (stacked["Hawkes-full"][:, i] != 0).mean()
        row["fill_S"] = (stacked["Hawkes-self"][:, i] != 0).mean()
        row["fill_E"] = (stacked["EWMA-count"][:, i]  != 0).mean()
        inst_rows.append(row)

    df_inst = pd.DataFrame(inst_rows)
    sh_cols = [f"sh_{col_abbr[p]}" for p in policy_names]
    print(df_inst[["inst"] + sh_cols + ["fill_H","fill_S","fill_E"]].to_string(
        index=False, float_format="{:.3f}".format))
    # -- Block-bootstrap significance test (tick-level, day-as-block) ----
    # Resamples whole days with replacement to preserve within-day autocorrelation.
    # Sharpe is computed across all stacked ticks in each bootstrap sample,
    # giving much higher effective N than the 22-day daily Sharpe.

    def block_bootstrap_sharpe_diff(
        pnl_a_days: list, pnl_b_days: list,
        n_boot: int = 10_000, seed: int = 42
    ) -> tuple:
        """
        pnl_a_days / pnl_b_days : list of (S, N) arrays, one per OOS day.
        Returns (obs_sr_a, obs_sr_b, obs_diff, ci_lo, ci_hi, p_one_sided).
        p_one_sided : P(boot_diff >= obs_diff | H0: diff=0) -- fraction of bootstrap
                      samples (centred at 0) that exceed the observed difference.
        """
        rng     = np.random.default_rng(seed)
        n_days  = len(pnl_a_days)

        # Per-tick P&L summed across instruments -- (S,) per day
        tick_a = [d.sum(axis=1) for d in pnl_a_days]
        tick_b = [d.sum(axis=1) for d in pnl_b_days]

        obs_sr_a = sharpe(np.concatenate(tick_a))
        obs_sr_b = sharpe(np.concatenate(tick_b))
        obs_diff = obs_sr_a - obs_sr_b

        boot_diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx         = rng.integers(0, n_days, n_days)
            boot_a      = np.concatenate([tick_a[i] for i in idx])
            boot_b      = np.concatenate([tick_b[i] for i in idx])
            boot_diffs[b] = sharpe(boot_a) - sharpe(boot_b)

        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        # Centre bootstrap distribution at 0 for the null hypothesis
        centred  = boot_diffs - boot_diffs.mean()
        p_val    = float((centred >= obs_diff).mean())
        return obs_sr_a, obs_sr_b, obs_diff, ci_lo, ci_hi, p_val

    print("\n=== Block-Bootstrap Significance (tick-level, day-as-block, 10k samples) ===")
    print(f"{'Policy A':16s}  {'vs':2s}  {'Policy B':16s}  {'DeltaSharpe':>8s}  "
          f"{'95% CI':>16s}  {'p (one-sided)':>14s}  {'Sr_A':>6s}  {'Sr_B':>6s}")
    comparisons = [
        ("Hawkes-floor",  "TOD"),
        ("Hawkes-select", "TOD"),
        ("Hawkes-floor",  "EWMA-count"),
        ("Hawkes-floor",  "RealVol"),
        ("Hawkes-floor",  "Hawkes-full"),
        ("TOD",           "EWMA-count"),
    ]
    bb_results = {}
    for pol_a, pol_b in comparisons:
        sr_a, sr_b, diff, lo, hi, p = block_bootstrap_sharpe_diff(
            pnl_stacks[pol_a], pnl_stacks[pol_b]
        )
        bb_results[(pol_a, pol_b)] = (sr_a, sr_b, diff, lo, hi, p)
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        print(f"{pol_a:16s}  vs  {pol_b:16s}  {diff:+8.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {p:14.4f}  {sr_a:6.4f}  {sr_b:6.4f}  {sig}")

    pd.DataFrame([
        {"policy_a": a, "policy_b": b,
         "sr_a": v[0], "sr_b": v[1], "delta": v[2],
         "ci_lo": v[3], "ci_hi": v[4], "p_onesided": v[5]}
        for (a, b), v in bb_results.items()
    ]).to_csv(out_path / "significance.csv", index=False)

    non_oracle = [p for p in policy_names if p != "Oracle"]
    for focus in ("Hawkes-full","Hawkes-floor","Hawkes-select","Hawkes-pull","TOD"):
        n_beats_all = sum(
            sharpe(stacked[focus][:, i]) > max(
                sharpe(stacked[p][:, i]) for p in non_oracle if p != focus)
            for i in range(N))
        n_beats_tod = sum(sharpe(stacked[focus][:, i]) > sharpe(stacked["TOD"][:, i])
                          for i in range(N))
        print(f"{focus:16s}: beats all non-oracle {n_beats_all:2d}/{N}  |  beats TOD {n_beats_tod:2d}/{N}")

    # -- Per-instrument block-bootstrap: Hawkes-select vs TOD ----
    print("\n=== Per-Instrument Block-Bootstrap: Hawkes-select vs TOD ===")
    print(f"{'Inst':6s}  {'DeltaSharpe':>8s}  {'95% CI':>20s}  {'p':>6s}  {'Sr_sel':>7s}  {'Sr_TOD':>7s}  {'sig':3s}  {'cross_in':>8s}  {'policy':12s}")

    per_inst_bb = []
    # cross_in from IS alpha (already computed earlier as cross_in_IS)
    for i in range(N):
        sym = id_map[str(i)]
        pnl_sel = [d[:, i:i+1] for d in pnl_stacks["Hawkes-select"]]  # keep 2-D (S,1)
        pnl_tod = [d[:, i:i+1] for d in pnl_stacks["TOD"]]
        sr_a, sr_b, diff, lo, hi, p = block_bootstrap_sharpe_diff(pnl_sel, pnl_tod)
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        pol = "Hfloor" if use_floor[i] else "TOD"
        ci = cross_in_IS[i]
        print(f"{sym:6s}  {diff:+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {p:6.4f}  {sr_a:7.4f}  {sr_b:7.4f}  {sig:3s}  {ci:8.3f}  {pol:12s}")
        per_inst_bb.append({"inst": sym, "sr_select": sr_a, "sr_tod": sr_b,
                             "delta": diff, "ci_lo": lo, "ci_hi": hi,
                             "p_onesided": p, "cross_in": ci, "policy": pol})

    pd.DataFrame(per_inst_bb).to_csv(out_path / "per_inst_bootstrap.csv", index=False)
    n_sig = sum(r["p_onesided"] < 0.05 for r in per_inst_bb)
    n_pos = sum(r["delta"] > 0 for r in per_inst_bb)
    print(f"\n{n_pos}/{N} instruments positive DeltaSharpe, {n_sig}/{N} significant at 5%")

    # -- Threshold sensitivity sweep ----
    # For any threshold t, select_pnl[:, i] = floor_pnl[:, i] if cross_in[i]<=t else tod_pnl[:, i].
    # Both are already in pnl_stacks, so no re-calibration needed.
    print("\n=== Threshold Sensitivity: cross_in cutoff vs Hawkes-select OOS ===")
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 5.0, 999.0]
    sweep_rows = []
    print(f"{'thresh':>7s}  {'n_floor':>7s}  {'sr_sel':>7s}  {'sr_tod':>7s}  "
          f"{'delta':>8s}  {'ci_lo':>7s}  {'ci_hi':>7s}  {'p':>6s}  {'sig':3s}")
    for thresh in thresholds:
        uf = cross_in_IS <= thresh
        n_floor = int(uf.sum())
        # Build per-day select pnl by mixing floor and tod columns
        pnl_sel_t = [
            np.where(uf[np.newaxis, :], pf, pt)
            for pf, pt in zip(pnl_stacks["Hawkes-floor"], pnl_stacks["TOD"])
        ]
        sr_a, sr_b, diff, lo, hi, p = block_bootstrap_sharpe_diff(
            pnl_sel_t, pnl_stacks["TOD"])
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        label = f"{thresh:.2f}" if thresh < 100 else "all"
        print(f"{label:>7s}  {n_floor:>7d}  {sr_a:7.4f}  {sr_b:7.4f}  "
              f"{diff:+8.4f}  {lo:+7.4f}  {hi:+7.4f}  {p:6.4f}  {sig}")
        sweep_rows.append({"threshold": thresh, "n_floor": n_floor,
                           "sr_select": sr_a, "sr_tod": sr_b, "delta": diff,
                           "ci_lo": lo, "ci_hi": hi, "p_onesided": p})
    pd.DataFrame(sweep_rows).to_csv(out_path / "threshold_sweep.csv", index=False)

    # -- Transaction cost robustness sweep ----
    # Apply flat $/share cost to all filled trades: pnl_tc = pnl - tc * fill_count.
    # Calibration is unchanged (conservative: strategy was not calibrated with costs).
    # Both policies pay tc, so relative comparison is fair.
    print("\n=== Transaction Cost Robustness: Hawkes-select vs TOD ===")
    tc_values = [0.0, 0.0005, 0.001, 0.002, 0.005]   # $/share ~= 0, 0.5, 1, 2, 5 bps on $100
    tc_rows = []
    print(f"{'tc_$/sh':>8s}  {'tc_bps@$100':>12s}  {'sr_sel':>7s}  {'sr_tod':>7s}  "
          f"{'delta':>8s}  {'ci_lo':>7s}  {'ci_hi':>7s}  {'p':>6s}  {'sig':3s}")
    for tc in tc_values:
        pnl_sel_tc = [pf - tc * ff for pf, ff in
                      zip(pnl_stacks["Hawkes-select"], fills_stacks["Hawkes-select"])]
        pnl_tod_tc = [pt - tc * ft for pt, ft in
                      zip(pnl_stacks["TOD"], fills_stacks["TOD"])]
        sr_a, sr_b, diff, lo, hi, p = block_bootstrap_sharpe_diff(pnl_sel_tc, pnl_tod_tc)
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        bps = tc * 10_000
        print(f"{tc:8.4f}  {bps:>12.1f}  {sr_a:7.4f}  {sr_b:7.4f}  "
              f"{diff:+8.4f}  {lo:+7.4f}  {hi:+7.4f}  {p:6.4f}  {sig}")
        tc_rows.append({"tc_per_share": tc, "tc_bps_at_100": bps,
                        "sr_select": sr_a, "sr_tod": sr_b, "delta": diff,
                        "ci_lo": lo, "ci_hi": hi, "p_onesided": p})
    pd.DataFrame(tc_rows).to_csv(out_path / "tc_sweep.csv", index=False)

    # Save
    agg.to_csv(out_path / "summary.csv", index=False)
    df_days.to_csv(out_path / "per_day.csv", index=False)
    df_inst.to_csv(out_path / "per_inst.csv", index=False)
    df_curve.to_csv(out_path / "curve.csv", index=False)
    pd.DataFrame([{"inst": id_map[str(i)], "k_hawkes": k_h[i], "k_hawkes_norm": k_hn[i],
                   "k_hawkes_self": k_s[i], "k_ewma": k_e[i], "k_rvol": k_v[i],
                   "k_evol": k_ev[i], "k_imbal": k_im[i], "k_tod": k_tod[i],
                   "tau_gate": tau_gate[i]}
                  for i in range(N)]).to_csv(out_path / "calibrated_k.csv", index=False)
    print(f"\nResults saved to {out_path}/")


if __name__ == "__main__":
    main()
