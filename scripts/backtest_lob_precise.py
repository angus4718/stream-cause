#!/usr/bin/env python3
"""Queue-aware MM backtest over reconstructed LOBs, comparing spread policies."""

import argparse
import datetime
import json
import re
import struct
import sys
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lob import Book # noqa: E402

FIXED_SCALE = 1_000_000_000
TICK_PX = int(1e7) # $0.01 tick
CROSS_IN_THRESHOLD = 0.5
ET = pytz.timezone("America/New_York")
SESSION_S = datetime.time(9, 30)
SESSION_E = datetime.time(16, 0)

POLICIES = ["sel", "tod", "fix", "qjn", "hgt", "hmin", "hfg", "htg", "hinv", "hfixfl"]
(POL_SEL, POL_TOD, POL_FIX, POL_QJN, POL_HGT, POL_HMIN, POL_HFG, POL_HTG, POL_HINV, POL_HFIXFL) = range(10)
NPOL = len(POLICIES)

# Policies that post at best bid/ask rather than mid +/- mm_hs
QJOIN_POLS = {POL_QJN, POL_HGT, POL_HFG, POL_HTG}


def round_tick(px_fixed: int) -> int:
    return round(px_fixed / TICK_PX) * TICK_PX


def ts_to_et_date_time(ts_ns: int):
    dt_utc = datetime.datetime.utcfromtimestamp(ts_ns / 1e9).replace(tzinfo=datetime.timezone.utc)
    dt_et = dt_utc.astimezone(ET)
    return dt_et.date(), dt_et.time().replace(microsecond=0)


def in_session(t: datetime.time) -> bool:
    return SESSION_S <= t < SESSION_E


def build_tod(S: int) -> np.ndarray:
    return 1.0 + 0.5 * np.cos(2 * np.pi * np.arange(S) / S)


def load_tick_ids(output_dir: Path, date_str: str) -> list:
    date_dir = output_dir / date_str
    if not date_dir.exists():
        return []
    paths = sorted(date_dir.glob("alpha_*.bin"), key=lambda p: int(re.search(r"alpha_(\d+)\.bin", p.name).group(1)))
    return [int(re.search(r"alpha_(\d+)\.bin", p.name).group(1)) for p in paths]


def load_lambda_series(output_dir: Path, date_str: str) -> np.ndarray:
    date_dir = output_dir / date_str
    if not date_dir.exists():
        return np.empty((0, 0))
    paths = sorted(date_dir.glob("lambda_*.bin"), key=lambda p: int(re.search(r"lambda_(\d+)\.bin", p.name).group(1)))
    if not paths:
        return np.empty((0, 0))
    vecs = []
    for p in paths:
        with open(p, "rb") as f:
            n_rows, n_cols = struct.unpack("ii", f.read(8))
            data = np.frombuffer(f.read(n_rows * n_cols * 8), dtype=np.float64)
            vecs.append(data.reshape((n_rows, n_cols), order="F").flatten())
    return np.stack(vecs, axis=0)


def load_alpha_mean(pipeline_root: Path, is_days: list, N: int) -> np.ndarray:
    from analysis.utils import load_alpha_series as _load_alpha
    alpha_acc = None
    cnt = 0
    for d in is_days:
        date_str = d.strftime("%Y-%m-%d")
        date_compact = d.strftime("%Y%m%d")
        al = _load_alpha(pipeline_root / date_compact, date_str)
        if al.shape[0] == 0:
            continue
        m = al.mean(axis=0)
        alpha_acc = m if alpha_acc is None else alpha_acc + m
        cnt += 1
    return (alpha_acc / cnt) if cnt else np.zeros((N, N))


def trading_days(s: datetime.date, e: datetime.date) -> list:
    US_HOLIDAYS = {datetime.date(2025, 11, 27), datetime.date(2025, 11, 28), datetime.date(2025, 12, 25)}
    days, d = [], s
    while d < e:
        if d.weekday() < 5 and d not in US_HOLIDAYS:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std()) if x.std() > 1e-12 else 0.0


def simulate_instrument_oos(dbn_file: Path, oos_days: list, pipeline_root: Path, inst_col: int, k_h: float, k_tod: float, k_fix: float, use_floor: bool, tau_gate: float, mm_size: int = 100) -> list:
    """Single-pass simulation for one instrument across all OOS days."""

    day_data = {}
    for d in oos_days:
        date_str = d.strftime("%Y-%m-%d")
        date_compact = d.strftime("%Y%m%d")
        tids = load_tick_ids(pipeline_root / date_compact, date_str)
        lam = load_lambda_series(pipeline_root / date_compact, date_str)
        if not tids or lam.shape[0] < len(tids):
            continue
        S = len(tids)
        tod = build_tod(S)
        lam_i = lam[:, inst_col]

        mm_full = k_h * lam_i
        mm_tod_s = k_tod * tod
        mm_floor = np.maximum(mm_tod_s, mm_full)
        mm_sel = mm_floor if use_floor else mm_tod_s

        # hmin: min(Hawkes, TOD) -- never wider than TOD
        mm_hmin = np.minimum(mm_full, mm_tod_s)

        # hinv: k_fix^2/(k_h*lambda+epsilon), clamped to [0.3*k_fix, 3*k_fix]
        lam_eps = max(np.percentile(lam_i, 5), 1e-9)
        mm_hinv = np.clip(k_fix * k_fix / (k_h * lam_i + lam_eps), 0.3 * k_fix, 3.0 * k_fix)

        # hfixfl: max(k_h*lambda, k_fix) -- Fixed as the floor instead of TOD
        mm_hfixfl = np.maximum(mm_full, k_fix)

        day_data[d] = {
            "tick_ids": tids, "S": S, "lam_i": lam_i,
            "mm_sel": mm_sel, "mm_tod": mm_tod_s,
            "mm_fix": np.full(S, k_fix),
            "mm_hmin": mm_hmin,
            "mm_hinv": mm_hinv,
            "mm_hfixfl": mm_hfixfl,
        }

    if not day_data:
        return []

    book = Book()

    results = {}
    for d, dd in day_data.items():
        S1 = dd["S"] - 1
        results[d] = {f"pnl_{p}": np.zeros(S1) for p in POLICIES}
        results[d].update({f"fills_{p}": np.zeros(S1) for p in POLICIES})

    bid_px = np.zeros(NPOL, dtype=np.int64)
    ask_px = np.zeros(NPOL, dtype=np.int64)
    qa_bid = np.zeros(NPOL, dtype=np.int64)
    qa_ask = np.zeros(NPOL, dtype=np.int64)
    # consumed_bid/ask: total size removed from orders that were ahead of us at
    # tick start, counting both fills (which arrive as "C" after "T") and plain
    # cancellations. This replaces the old vol_bid/vol_ask which only counted "T".
    consumed_bid = np.zeros(NPOL, dtype=np.int64)
    consumed_ask = np.zeros(NPOL, dtype=np.int64)
    # traded_bid/ask: True if any "T" occurred at our price during this tick.
    # Fill requires both consumed >= qa AND a trade actually happened.
    traded_bid = np.zeros(NPOL, dtype=bool)
    traded_ask = np.zeros(NPOL, dtype=bool)
    mid_place = np.zeros(NPOL, dtype=np.int64)
    # snap_bid/snap_ask[p]: set of order_ids resting at bid_px[p]/ask_px[p] at
    # tick start; any "C" for these order_ids reduces consumed_bid/ask[p].
    snap_bid: list[set] = [set() for _ in range(NPOL)]
    snap_ask: list[set] = [set() for _ in range(NPOL)]

    def reset_vorders():
        bid_px[:] = ask_px[:] = 0
        qa_bid[:] = qa_ask[:] = 0
        consumed_bid[:] = consumed_ask[:] = 0
        traded_bid[:] = traded_ask[:] = False
        mid_place[:] = 0
        for p in range(NPOL):
            snap_bid[p].clear()
            snap_ask[p].clear()

    def place_orders(s: int, dd: dict, bbo_bid: int, bbo_ask: int):
        mid = (bbo_bid + bbo_ask) // 2
        lam_s = dd["lam_i"][s]
        gate = (lam_s <= tau_gate)

        # Spread-width policies
        spread_pols = [
            (POL_SEL, dd["mm_sel"]),
            (POL_TOD, dd["mm_tod"]),
            (POL_FIX, dd["mm_fix"]),
            (POL_HMIN, dd["mm_hmin"]),
            (POL_HINV, dd["mm_hinv"]),
            (POL_HFIXFL, dd["mm_hfixfl"]),
        ]
        for p, sig_arr in spread_pols:
            mm_fixed = int(sig_arr[s] * FIXED_SCALE)
            bq = round_tick(mid - mm_fixed)
            aq = round_tick(mid + mm_fixed)
            bid_px[p] = bq; ask_px[p] = aq
            mid_place[p] = mid
            consumed_bid[p] = consumed_ask[p] = 0
            traded_bid[p] = traded_ask[p] = False
            # snapshot order IDs ahead of our virtual order (FIFO queue position)
            bl = book._get_or_insert_level(bq, "B") if bq in book.bids else None
            al = book._get_or_insert_level(aq, "A") if aq in book.offers else None
            if bl:
                snap_bid[p] = {o.order_id for o in bl.orders}
                qa_bid[p] = sum(o.size for o in bl.orders)
            else:
                snap_bid[p] = set(); qa_bid[p] = 0
            if al:
                snap_ask[p] = {o.order_id for o in al.orders}
                qa_ask[p] = sum(o.size for o in al.orders)
            else:
                snap_ask[p] = set(); qa_ask[p] = 0

        # Queue-join policies (post at best bid/ask when active)
        qjn_pols = [
            (POL_QJN, True),
            (POL_HGT, gate), # gate at best bid/ask
            (POL_HFG, gate), # gate at Fixed spread
            (POL_HTG, gate), # gate at TOD spread
        ]
        for p, active in qjn_pols:
            if not active:
                mid_place[p] = 0
                bid_px[p] = ask_px[p] = 0; qa_bid[p] = qa_ask[p] = 0
                consumed_bid[p] = consumed_ask[p] = 0
                traded_bid[p] = traded_ask[p] = False
                snap_bid[p].clear(); snap_ask[p].clear()
                continue

            if p in (POL_HFG,):
                # post at Fixed spread (not best bid/ask)
                mm_fixed = int(dd["mm_fix"][s] * FIXED_SCALE)
                bq = round_tick(mid - mm_fixed)
                aq = round_tick(mid + mm_fixed)
            elif p in (POL_HTG,):
                # post at TOD spread
                mm_fixed = int(dd["mm_tod"][s] * FIXED_SCALE)
                bq = round_tick(mid - mm_fixed)
                aq = round_tick(mid + mm_fixed)
            else:
                # post at best bid/ask (Qjoin style)
                bq = bbo_bid; aq = bbo_ask

            bid_px[p] = bq; ask_px[p] = aq
            mid_place[p] = mid
            consumed_bid[p] = consumed_ask[p] = 0
            traded_bid[p] = traded_ask[p] = False
            bl = book._get_or_insert_level(bq, "B") if bq in book.bids else None
            al = book._get_or_insert_level(aq, "A") if aq in book.offers else None
            if bl:
                snap_bid[p] = {o.order_id for o in bl.orders}
                qa_bid[p] = sum(o.size for o in bl.orders)
            else:
                snap_bid[p] = set(); qa_bid[p] = 0
            if al:
                snap_ask[p] = {o.order_id for o in al.orders}
                qa_ask[p] = sum(o.size for o in al.orders)
            else:
                snap_ask[p] = set(); qa_ask[p] = 0

    def settle_tick(s: int, d: datetime.date, next_mid: int):
        res = results[d]
        for p, pol in enumerate(POLICIES):
            if mid_place[p] == 0:
                continue
            # Fill iff the queue ahead of us was fully consumed (trades + cancels)
            # AND at least one trade occurred at our price (someone to fill against).
            bf = mm_size if (traded_bid[p] and consumed_bid[p] >= qa_bid[p]) else 0
            af = mm_size if (traded_ask[p] and consumed_ask[p] >= qa_ask[p]) else 0
            pnl = 0.0
            tot = 0
            if bf > 0 and next_mid > 0:
                pnl += bf * (next_mid - bid_px[p]) / FIXED_SCALE; tot += bf
            if af > 0 and next_mid > 0:
                pnl += af * (ask_px[p] - next_mid) / FIXED_SCALE; tot += af
            res[f"pnl_{pol}"][s] += pnl
            res[f"fills_{pol}"][s] += tot

    data = db.DBNStore.from_file(str(dbn_file))
    instrument_map = db.common.symbology.InstrumentMap()
    instrument_map.insert_metadata(data.metadata)

    cur_date = None; cur_dd = None; cur_tick = -1; next_tick_idx = 0

    for mbo in data:
        ts = mbo.ts_event
        evt_date, evt_time = ts_to_et_date_time(ts)

        if evt_date != cur_date:
            reset_vorders()
            cur_date = evt_date
            cur_dd = day_data.get(evt_date)
            cur_tick = -1; next_tick_idx = 0

        book.apply(mbo)

        if cur_dd is None:
            continue

        tids = cur_dd["tick_ids"]
        S = cur_dd["S"]

        while next_tick_idx < S and ts >= tids[next_tick_idx]:
            s = next_tick_idx
            bl = book.get_bid_level()
            al = book.get_ask_level()
            next_mid_fixed = ((bl.price + al.price) // 2) if (bl and al) else 0

            if s > 0 and cur_tick == s - 1:
                settle_tick(s - 1, evt_date, next_mid_fixed)

            reset_vorders()
            if bl and al and in_session(evt_time):
                place_orders(s, cur_dd, bl.price, al.price)

            cur_tick = s; next_tick_idx += 1

        if cur_tick >= 0 and cur_tick < S - 1 and in_session(evt_time):
            if mbo.action == "T":
                # Record that a trade occurred at this price (needed for fill guard).
                tpx = mbo.price
                for p in range(NPOL):
                    if mbo.side == "B" and tpx == bid_px[p] and bid_px[p] > 0:
                        traded_bid[p] = True
                    elif mbo.side == "A" and tpx == ask_px[p] and ask_px[p] > 0:
                        traded_ask[p] = True

            elif mbo.action == "C":
                # A cancel reduces the queue of any order that was ahead of us at
                # tick start. This covers both fill-driven cancels (which follow a
                # "T" message) and plain order withdrawals. We read order_id and
                # size from the MBO message; book.apply() has already been called
                # above but mbo.order_id / mbo.size are unchanged.
                oid = mbo.order_id; csz = mbo.size
                for p in range(NPOL):
                    if oid in snap_bid[p]:
                        consumed_bid[p] += csz
                    if oid in snap_ask[p]:
                        consumed_ask[p] += csz

    return [{"date": d.isoformat(), **res} for d, res in results.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="/ocean/projects/cis260122p/shared/data/raw")
    parser.add_argument("--pipeline_root", default="output/multiday")
    parser.add_argument("--id_map", default="data/allstock_id_map.json")
    parser.add_argument("--calibrated_k", default="output/mm_backtest_multiday/calibrated_k.csv")
    parser.add_argument("--out", default="output/mm_backtest_precise")
    parser.add_argument("--is_end", default="2025-12-01")
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--mm_size", type=int, default=1)
    parser.add_argument("--syms", nargs="*", default=None,
                        help="Restrict to these symbols only (default: all)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    pipeline_root = Path(args.pipeline_root)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(args.id_map) as f:
        id_map = json.load(f)
    N = len(id_map)

    cal_k = pd.read_csv(args.calibrated_k).set_index("inst")

    is_end_date = datetime.date.fromisoformat(args.is_end)
    start_date = datetime.date.fromisoformat(args.start)
    end_date = datetime.date.fromisoformat(args.end)
    all_days = trading_days(start_date, end_date)
    is_days = [d for d in all_days if d < is_end_date]
    oos_days = [d for d in all_days if d >= is_end_date]

    is_mean_alpha = load_alpha_mean(pipeline_root, is_days, N)
    cross_in_IS = is_mean_alpha.sum(axis=1) - is_mean_alpha.diagonal()
    use_floor = cross_in_IS <= CROSS_IN_THRESHOLD

    all_day_results = []
    for i in range(N):
        sym = id_map[str(i)]
        if args.syms is not None and sym not in args.syms:
            continue
        if sym not in cal_k.index:
            continue

        k_h_i = float(cal_k.loc[sym, "k_hawkes"])
        k_tod_i = float(cal_k.loc[sym, "k_tod"])
        k_fix_i = k_tod_i * 1.0 # Fixed = k_tod * TOD(mid-day weight)
        uf_i = bool(use_floor[i])
        tau_gate_i = float(cal_k.loc[sym, "tau_gate"]) if "tau_gate" in cal_k.columns else np.inf

        oos_month_start = datetime.date(2025, 12, 1)
        oos_month_end = datetime.date(2026, 1, 1)
        dbn_file = (raw_dir / sym /
                    f"XNAS_ITCH_{sym}_mbo_"
                    f"{oos_month_start.strftime('%Y%m%d')}_"
                    f"{oos_month_end.strftime('%Y%m%d')}.dbn.zst")

        if not dbn_file.exists():
            # Fallback: look for any file that spans the OOS period (e.g. 3-month file)
            sym_dir = raw_dir / sym
            fallback = None
            if sym_dir.exists():
                for f in sorted(sym_dir.glob(f"XNAS_ITCH_{sym}_mbo_*.dbn.zst")):
                    m = re.search(r"_(\d{8})_(\d{8})\.dbn\.zst$", f.name)
                    if m:
                        f_start = datetime.date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:]))
                        f_end = datetime.date(int(m.group(2)[:4]), int(m.group(2)[4:6]), int(m.group(2)[6:]))
                        if f_start <= oos_month_start and f_end >= oos_month_end:
                            fallback = f
                            break
            if fallback is None:
                continue
            dbn_file = fallback

        try:
            day_res = simulate_instrument_oos(
                dbn_file=dbn_file, oos_days=oos_days,
                pipeline_root=pipeline_root, inst_col=i,
                k_h=k_h_i, k_tod=k_tod_i, k_fix=k_fix_i,
                use_floor=uf_i, tau_gate=tau_gate_i,
                mm_size=args.mm_size,
            )
        except Exception as e:
            print(f" {sym}: ERROR -- {e}")
            import traceback; traceback.print_exc()
            continue

        for r in day_res:
            r["inst"] = sym
        all_day_results.extend(day_res)

    if not all_day_results:
        print("No results produced."); sys.exit(1)

    rows = []
    for r in all_day_results:
        inst = r["inst"]; date = r["date"]
        for pol in POLICIES:
            pnl_arr = r[f"pnl_{pol}"]
            fill_arr = r[f"fills_{pol}"]
            rows.append({
                "date": date, "inst": inst, "policy": pol,
                "total_pnl": float(pnl_arr.sum()),
                "fill_rate": float((fill_arr > 0).mean()),
                "total_fills": float(fill_arr.sum()),
                "n_ticks": len(pnl_arr),
            })

    df = pd.DataFrame(rows)
    csv_path = out_path / "day_results.csv"
    if args.syms is not None and csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[~existing["inst"].isin(args.syms)]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)

    pol_labels = [
        ("sel", "Hselect "),
        ("tod", "TOD "),
        ("fix", "Fixed "),
        ("qjn", "Qjoin "),
        ("hgt", "Hgate "),
        ("hmin", "Hmin "),
        ("hfg", "Hfix-g "),
        ("htg", "Htod-g "),
        ("hinv", "Hinv "),
        ("hfixfl", "HfixFl "),
    ]

    print("\n=== Precise LOB Backtest -- OOS Summary ===")
    print(f"{'Policy':10s} {'Total PnL':>12s} {'Mean/Day':>10s} {'Sharpe':>7s} "
          f"{'Fill Rate':>9s}")
    for pol, label in pol_labels:
        sub = df[df.policy == pol]
        daily = sub.groupby("date")["total_pnl"].sum()
        fill = sub["fill_rate"].mean()
        sr = sharpe(daily.values)
        print(f"{label} {daily.sum():12.2f} {daily.mean():10.2f} {sr:7.4f} {fill:9.4f}")

    print("\n=== Per-Instrument OOS Sharpes ===")
    header = f"{'Inst':6s}" + "".join(f" {p[1].strip():>8s}" for p in pol_labels)
    print(header)
    for i in range(N):
        sym = id_map[str(i)]
        line = f"{sym:6s}"
        for pol, _ in pol_labels:
            sub = df[(df.inst == sym) & (df.policy == pol)]
            if sub.empty:
                line += " n/a"
                continue
            d_pol = sub.groupby("date")["total_pnl"].sum()
            line += f" {sharpe(d_pol.values):8.4f}"
        print(line)

    print(f"\nResults saved to {out_path}/")


if __name__ == "__main__":
    main()
