"""
Causal Graph Analysis
=====================
Analyses the time-varying alpha matrices produced by StreamCause.

Outputs (output/causal_graph_analysis/):
  mean_alpha_IS.csv        -- mean alpha matrix over IS period
  mean_alpha_OOS.csv       -- mean alpha matrix over OOS period
  per_inst_summary.csv     -- per-instrument causal metrics + backtest link
  heatmap_IS.png           -- heatmap of IS mean alpha
  heatmap_OOS.png          -- heatmap of OOS mean alpha
  heatmap_diff.png         -- OOS - IS difference heatmap
  stability_scatter.png    -- IS vs OOS alpha values scatter
  top_edges.csv            -- top 20 strongest / most stable causal edges
  intraday_alpha.png       -- mean alpha[i,j] by time-of-day bucket
  report.txt               -- text summary
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.utils import load_alpha_series

US_HOLIDAYS = {
    datetime.date(2025, 11, 27),
    datetime.date(2025, 11, 28),
    datetime.date(2025, 12, 25),
}


def trading_days_local(start, end):
    days, d = [], start
    while d < end:
        if d.weekday() < 5 and d not in US_HOLIDAYS:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


# -- helpers ----

def load_day_alpha(pipeline_root: Path, date: datetime.date) -> np.ndarray | None:
    """Load alpha_series for one day. Returns (S, N, N) or None."""
    date_str     = date.strftime("%Y-%m-%d")
    date_compact = date.strftime("%Y%m%d")
    out_dir      = pipeline_root / date_compact
    alpha        = load_alpha_series(out_dir, date_str)
    if alpha.shape[0] == 0:
        return None
    return alpha   # (S, N, N)


def mean_alpha(days_alpha: list[np.ndarray]) -> np.ndarray:
    """Mean alpha matrix over all ticks across all days. Returns (N, N)."""
    all_ticks = np.concatenate(days_alpha, axis=0)  # (T_total, N, N)
    return all_ticks.mean(axis=0)


def alpha_by_tod(days_alpha: list[np.ndarray], n_bins: int = 13) -> np.ndarray:
    """Mean alpha in each intraday bin (opening -> closing).
    Returns (n_bins, N, N)."""
    bins = [[] for _ in range(n_bins)]
    for alpha_day in days_alpha:
        S = alpha_day.shape[0]
        for s in range(S):
            b = min(int(s / S * n_bins), n_bins - 1)
            bins[b].append(alpha_day[s])
    result = np.zeros((n_bins, days_alpha[0].shape[1], days_alpha[0].shape[2]))
    for b, mats in enumerate(bins):
        if mats:
            result[b] = np.stack(mats).mean(axis=0)
    return result


# -- plotting ----

def plot_heatmap(mat: np.ndarray, syms: list[str], title: str, path: Path,
                 vmin=None, vmax=None, cmap="YlOrRd", center=None):
    N = len(syms)
    fig, ax = plt.subplots(figsize=(9, 8))
    if center is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm)
    else:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N)); ax.set_xticklabels(syms, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(syms, fontsize=8)
    ax.set_xlabel("source j  (alpha[i,j] = excitation of row i from col j)")
    ax.set_ylabel("target i")
    ax.set_title(title)
    # Annotate cells
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=5, color="black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_stability_scatter(is_mat: np.ndarray, oos_mat: np.ndarray,
                           syms: list[str], path: Path):
    N = len(syms)
    is_flat  = is_mat.flatten()
    oos_flat = oos_mat.flatten()
    corr = np.corrcoef(is_flat, oos_flat)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    # Colour diagonal (self) differently from off-diagonal
    colours = []
    for i in range(N):
        for j in range(N):
            colours.append("red" if i == j else "steelblue")
    ax.scatter(is_flat, oos_flat, c=colours, alpha=0.6, s=20)
    lim = max(is_flat.max(), oos_flat.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y=x")
    ax.set_xlabel("IS mean alpha[i,j]")
    ax.set_ylabel("OOS mean alpha[i,j]")
    ax.set_title(f"IS vs OOS alpha stability  (r={corr:.3f})\nblue=off-diag  red=self")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_intraday(tod_alpha: np.ndarray, syms: list[str], path: Path):
    """Plot mean total cross-excitation (sum off-diagonal alpha[i,:]) by intraday bin."""
    n_bins, N, _ = tod_alpha.shape
    # Total incoming cross for each instrument at each time bin
    cross_in = np.zeros((n_bins, N))
    for b in range(n_bins):
        for i in range(N):
            cross_in[b, i] = tod_alpha[b, i, :].sum() - tod_alpha[b, i, i]

    bin_labels = [f"{int(9.5 + b * 6.5/n_bins):d}:{int((9.5 + b*6.5/n_bins) % 1 * 60):02d}"
                  for b in range(n_bins)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sym in enumerate(syms):
        ax.plot(range(n_bins), cross_in[:, i], marker="o", ms=3, label=sym, alpha=0.8)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Intraday bin (ET)")
    ax.set_ylabel("Mean total incoming cross-excitation  sum_{j!=i} alpha[i,j]")
    ax.set_title("Cross-instrument excitation by time of day (IS period)")
    ax.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# -- main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_root", default="output/multiday")
    parser.add_argument("--id_map",        default="data/allstock_id_map.json")
    parser.add_argument("--per_inst_csv",  default="output/mm_backtest_multiday/per_inst.csv")
    parser.add_argument("--out",           default="output/causal_graph_analysis")
    parser.add_argument("--is_end",        default="2025-12-01")
    parser.add_argument("--start",         default="2025-10-01")
    parser.add_argument("--end",           default="2026-01-01")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(args.id_map) as f:
        id_map = json.load(f)
    syms = [id_map[str(i)] for i in range(len(id_map))]
    N = len(syms)

    pipeline_root = Path(args.pipeline_root)
    is_end  = datetime.date.fromisoformat(args.is_end)
    start   = datetime.date.fromisoformat(args.start)
    end     = datetime.date.fromisoformat(args.end)
    all_days = trading_days_local(start, end)

    # -- Load alpha series ----
    print("Loading alpha series...")
    is_alphas, oos_alphas = [], []
    for day in all_days:
        alpha = load_day_alpha(pipeline_root, day)
        if alpha is None:
            print(f"  {day}: missing, skip")
            continue
        if day < is_end:
            is_alphas.append(alpha)
        else:
            oos_alphas.append(alpha)
        print(f"  {day}: {alpha.shape[0]} ticks  alphain[{alpha.min():.3f},{alpha.max():.3f}]")

    print(f"\nIS days: {len(is_alphas)}  OOS days: {len(oos_alphas)}")

    # -- Mean alpha matrices ----
    print("\nComputing mean alpha matrices...")
    is_mean  = mean_alpha(is_alphas)   # (N, N)
    oos_mean = mean_alpha(oos_alphas)  # (N, N)
    diff     = oos_mean - is_mean

    corr_flat = np.corrcoef(is_mean.flatten(), oos_mean.flatten())[0, 1]
    print(f"  IS-OOS correlation (all {N*N} entries): r={corr_flat:.4f}")

    # off-diagonal only
    mask_off = ~np.eye(N, dtype=bool)
    corr_off = np.corrcoef(is_mean[mask_off], oos_mean[mask_off])[0, 1]
    print(f"  IS-OOS correlation (off-diagonal only):  r={corr_off:.4f}")

    # -- Per-instrument metrics ----
    print("\nPer-instrument causal metrics (IS mean):")
    cross_in_IS  = np.array([is_mean[i, :].sum() - is_mean[i, i] for i in range(N)])
    cross_out_IS = np.array([is_mean[:, j].sum() - is_mean[j, j] for j in range(N)])
    self_IS      = is_mean.diagonal()

    # IS-OOS per-instrument stability: correlation of alpha[i,:] across IS and OOS
    row_stability = np.array([np.corrcoef(is_mean[i], oos_mean[i])[0, 1] for i in range(N)])
    col_stability = np.array([np.corrcoef(is_mean[:, j], oos_mean[:, j])[0, 1] for j in range(N)])

    # Load per-instrument backtest results
    df_inst = pd.read_csv(args.per_inst_csv)
    hfloor_sharpe = dict(zip(df_inst["inst"].str.strip(), df_inst["sh_Hfloor"]))
    tod_sharpe    = dict(zip(df_inst["inst"].str.strip(), df_inst["sh_TOD"]))
    hfull_sharpe  = dict(zip(df_inst["inst"].str.strip(), df_inst["sh_Hfull"]))

    delta_floor_tod = np.array([hfloor_sharpe.get(s, np.nan) - tod_sharpe.get(s, np.nan)
                                for s in syms])
    delta_full_tod  = np.array([hfull_sharpe.get(s, np.nan) - tod_sharpe.get(s, np.nan)
                                for s in syms])

    inst_df = pd.DataFrame({
        "inst":          syms,
        "self_IS":       self_IS,
        "cross_in_IS":   cross_in_IS,
        "cross_out_IS":  cross_out_IS,
        "row_stability": row_stability,
        "col_stability": col_stability,
        "sh_Hfloor":     [hfloor_sharpe.get(s, np.nan) for s in syms],
        "sh_TOD":        [tod_sharpe.get(s, np.nan)    for s in syms],
        "sh_Hfull":      [hfull_sharpe.get(s, np.nan)  for s in syms],
        "delta_floor_tod": delta_floor_tod,
        "delta_full_tod":  delta_full_tod,
    })
    print(inst_df.to_string(index=False, float_format="{:.3f}".format))

    # Correlations with backtest benefit
    valid = ~np.isnan(delta_floor_tod)
    for metric, vals in [("cross_in_IS",   cross_in_IS),
                          ("cross_out_IS",  cross_out_IS),
                          ("self_IS",       self_IS),
                          ("row_stability", row_stability)]:
        r = np.corrcoef(vals[valid], delta_floor_tod[valid])[0, 1]
        print(f"  corr({metric}, Delta_floor-TOD) = {r:.3f}")

    # -- Top edges ----
    edges = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            edges.append({
                "source": syms[j], "target": syms[i],
                "alpha_IS":  is_mean[i, j],
                "alpha_OOS": oos_mean[i, j],
                "abs_diff":  abs(diff[i, j]),
                "rel_change": (oos_mean[i, j] - is_mean[i, j]) / (is_mean[i, j] + 1e-8),
            })
    df_edges = pd.DataFrame(edges).sort_values("alpha_IS", ascending=False)
    print("\nTop 15 causal edges by IS strength (source -> target):")
    print(df_edges.head(15).to_string(index=False, float_format="{:.4f}".format))

    # -- Intraday alpha by TOD bin ----
    print("\nComputing intraday alpha profile...")
    tod_alpha = alpha_by_tod(is_alphas, n_bins=13)

    # -- Save outputs ----
    print("\nSaving outputs...")
    pd.DataFrame(is_mean,  index=syms, columns=syms).to_csv(out_path / "mean_alpha_IS.csv")
    pd.DataFrame(oos_mean, index=syms, columns=syms).to_csv(out_path / "mean_alpha_OOS.csv")
    pd.DataFrame(diff,     index=syms, columns=syms).to_csv(out_path / "mean_alpha_diff.csv")
    inst_df.to_csv(out_path / "per_inst_summary.csv", index=False)
    df_edges.to_csv(out_path / "top_edges.csv", index=False)

    vmax = max(is_mean.max(), oos_mean.max())
    plot_heatmap(is_mean,  syms, "Mean alpha[i,j] -- IS (Oct-Nov 2025)",
                 out_path / "heatmap_IS.png",  vmin=0, vmax=vmax)
    plot_heatmap(oos_mean, syms, "Mean alpha[i,j] -- OOS (Dec 2025)",
                 out_path / "heatmap_OOS.png", vmin=0, vmax=vmax)
    plot_heatmap(diff, syms, "Delta alpha[i,j] = OOS - IS",
                 out_path / "heatmap_diff.png",
                 vmin=diff.min(), vmax=diff.max(), center=0.0)

    plot_stability_scatter(is_mean, oos_mean, syms, out_path / "stability_scatter.png")
    plot_intraday(tod_alpha, syms, out_path / "intraday_alpha.png")

    # Text report
    lines = [
        "=" * 60,
        "StreamCause Causal Graph Analysis",
        f"IS: Oct-Nov 2025 ({len(is_alphas)} days)   OOS: Dec 2025 ({len(oos_alphas)} days)",
        "=" * 60,
        f"\nIS-OOS stability (all entries):    r={corr_flat:.4f}",
        f"IS-OOS stability (off-diagonal):   r={corr_off:.4f}",
        "",
        "IS mean alpha (diagonal = self-excitation):",
    ]
    for i in range(N):
        row_str = "  ".join(f"{is_mean[i,j]:.3f}" for j in range(N))
        lines.append(f"  {syms[i]:6s}  {row_str}")
    lines += [
        "",
        "Per-instrument summary (sorted by Delta Hawkes-floor vs TOD Sharpe):",
        inst_df.sort_values("delta_floor_tod", ascending=False).to_string(index=False),
        "",
        "Top 20 causal edges by IS mean alpha:",
        df_edges.head(20).to_string(index=False),
    ]
    report = "\n".join(lines)
    (out_path / "report.txt").write_text(report)
    print(report)
    print(f"\nAll outputs written to {out_path}/")


if __name__ == "__main__":
    main()
