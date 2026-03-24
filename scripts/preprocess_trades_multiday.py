#!/usr/bin/env python3
"""
Multi-day allstock trades binary for the StreamCause C++ pipeline.

Streams DBN files per symbol, collects trade events into a numpy structured
array (32 bytes/event vs ~200 bytes for Python tuples), sorts by timestamp,
and writes a single packed binary covering the full date range.

Parallelism: --workers N  (one worker per symbol, reads in parallel)
Memory: ~32 bytes * total_events (e.g. 82M events * 32 B = 2.6 GB for 63 days)

Output: data/allstock_trades_<start>_<end>.bin
        data/allstock_id_map.json (created from existing id_map if absent)

Usage:
    python scripts/preprocess_trades_multiday.py \
        [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--workers 8]
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pytz

RAW_DIR    = "/ocean/projects/cis260122p/shared/data/raw"
RECORD_FMT = "<qIB3xqII"   # 32 bytes
RECORD_DT  = np.dtype([    # same layout as struct, zero-copy compatible
    ("ts_ns",   "<i8"),
    ("inst_id", "<u4"),
    ("action",  "u1"),
    ("_pad",    "3u1"),
    ("price",   "<i8"),
    ("size",    "<u4"),
    ("_pad2",   "<u4"),
])
assert struct.calcsize(RECORD_FMT) == 32
assert RECORD_DT.itemsize == 32

ET = pytz.timezone("America/New_York")
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


def session_ns(date: datetime.date) -> tuple[int, int]:
    open_dt  = ET.localize(datetime.datetime(date.year, date.month, date.day, 9, 30))
    close_dt = ET.localize(datetime.datetime(date.year, date.month, date.day, 16, 0))
    return (int(open_dt.timestamp() * 1_000_000_000),
            int(close_dt.timestamp() * 1_000_000_000))


def discover_files(sym: str) -> list[tuple[str, str, str]]:
    sym_dir = os.path.join(RAW_DIR, sym)
    entries = []
    for f in sorted(glob.glob(f"{sym_dir}/*.dbn.zst")):
        fname = os.path.basename(f)
        parts = fname.replace(".dbn.zst", "").split("_")
        if len(parts) >= 2:
            entries.append((parts[-2], parts[-1], f))
    return sorted(entries)


def load_sym_events(sym: str, inst_id: int,
                    sessions: list[tuple[datetime.date, int, int]]) -> np.ndarray:
    """Worker: read all monthly DBN files for one symbol, return structured array."""
    import databento as db

    file_list = discover_files(sym)
    chunks: list[np.ndarray] = []

    for file_start, file_end, filepath in file_list:
        file_sessions = [
            (d, o, c) for (d, o, c) in sessions
            if file_start <= d.strftime("%Y%m%d") < file_end
        ]
        if not file_sessions:
            continue

        last_close = file_sessions[-1][2] + 60_000_000_000
        rows = []
        store = db.DBNStore.from_file(filepath)

        for mbo in store:
            ts  = mbo.ts_event
            act = mbo.action.value if hasattr(mbo.action, "value") else str(mbo.action)

            if ts > last_close and act == "T":
                break
            if act != "T":
                continue

            for (_, open_ns, close_ns) in file_sessions:
                if open_ns <= ts < close_ns:
                    rows.append((ts, inst_id, 2, int(mbo.price), int(mbo.size)))
                    break

        if rows:
            arr = np.zeros(len(rows), dtype=RECORD_DT)
            for k, (ts, iid, act, price, size) in enumerate(rows):
                arr[k]["ts_ns"]   = ts
                arr[k]["inst_id"] = iid
                arr[k]["action"]  = act
                arr[k]["price"]   = price
                arr[k]["size"]    = size
            chunks.append(arr)
            n = sum(len(c) for c in chunks)
            print(f"  {sym}: {os.path.basename(filepath)} -> {len(rows):,} trades "
                  f"(total {n:,})", flush=True)

    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=RECORD_DT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default="2025-10-01")
    parser.add_argument("--end",     default="2026-01-01")
    parser.add_argument("--id_map",  default="data/allstock_id_map.json")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load or copy id_map
    if not os.path.exists(args.id_map):
        src = "data/test_20251201_allstock_id_map.json"
        with open(src) as f:
            id_map = json.load(f)
        with open(args.id_map, "w") as f:
            json.dump(id_map, f, indent=2)
    with open(args.id_map) as f:
        id_map = json.load(f)
    sym_to_id = {v: int(k) for k, v in id_map.items()}
    print(f"Universe: {len(sym_to_id)} symbols  workers={args.workers}")

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)
    days       = trading_days(start_date, end_date)
    print(f"Trading days: {len(days)}  ({days[0]} -> {days[-1]})")

    sessions = [(d, *session_ns(d)) for d in days]

    # Parallel reads: one worker per symbol
    sym_arrays: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(load_sym_events, sym, inst_id, sessions): sym
            for sym, inst_id in sym_to_id.items()
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                sym_arrays[sym] = fut.result()
                print(f"  [OK] {sym}: {len(sym_arrays[sym]):,} events collected", flush=True)
            except Exception as e:
                print(f"  [FAIL] {sym}: ERROR {e}", flush=True)
                sym_arrays[sym] = np.zeros(0, dtype=RECORD_DT)

    print("\nMerging and sorting...")
    all_arrays = [a for a in sym_arrays.values() if len(a) > 0]
    merged = np.concatenate(all_arrays)
    print(f"  Total events: {len(merged):,}")

    order = np.argsort(merged["ts_ns"], kind="stable")
    merged = merged[order]

    start_compact = args.start.replace("-", "")
    end_compact   = args.end.replace("-", "")
    out_path = os.path.join(args.out_dir,
                            f"allstock_trades_{start_compact}_{end_compact}.bin")

    merged.tofile(out_path)
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Written: {out_path}  ({mb:.1f} MB, {len(merged):,} records * 32 bytes)")

    # Per-day summary
    day_map: dict[str, int] = {}
    for (d, open_ns, close_ns) in sessions:
        mask = (merged["ts_ns"] >= open_ns) & (merged["ts_ns"] < close_ns)
        day_map[d.strftime("%Y-%m-%d")] = int(mask.sum())
    print("\nPer-day trade counts:")
    for d, cnt in sorted(day_map.items()):
        print(f"  {d}: {cnt:>8,}")


if __name__ == "__main__":
    main()
