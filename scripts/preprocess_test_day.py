#!/usr/bin/env python3
"""
Preprocess one trading day of ITCH MBO data for the StreamCause smoke test.

Streams DBN records for AAPL, MSFT, TSLA, filters to a single trading day,
merge-sorts, and writes a packed 32-byte binary event file.

Usage:
    /ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3.11 \
        scripts/preprocess_test_day.py
"""

import struct
import sys
import os
import glob

import databento as db

# ----
# Config
# ----

# 2025-12-01 09:30-16:00 ET = UTC-5 (December is EST)
# 09:30 ET = 14:30 UTC = 1764599400 seconds = 1764599400_000000000 ns
# 16:00 ET = 21:00 UTC = 1764622800 seconds = 1764622800_000000000 ns
OPEN_NS  = 1_764_599_400_000_000_000
CLOSE_NS = 1_764_622_800_000_000_000

RAW_DIR = "/ocean/projects/cis260122p/shared/data/raw"
TARGET_DATE = "20251201"

def discover_stocks():
    """Find all symbols that have a DBN file covering TARGET_DATE, sorted alphabetically."""
    stocks = {}
    for sym_dir in sorted(glob.glob(f"{RAW_DIR}/*/")):
        sym = os.path.basename(sym_dir.rstrip("/"))
        # Find any file whose date range spans TARGET_DATE
        for f in sorted(glob.glob(f"{sym_dir}*.dbn.zst")):
            fname = os.path.basename(f)
            # Filename pattern: XNAS_ITCH_<SYM>_mbo_<START>_<END>.dbn.zst
            parts = fname.replace(".dbn.zst", "").split("_")
            if len(parts) >= 2:
                start, end = parts[-2], parts[-1]
                if start <= TARGET_DATE < end:
                    stocks[sym] = f
                    break
    return stocks

# Action encoding: Trade=2 only (ADD/CANCEL skipped -- too self-exciting, no cross-asset signal)
ACTION_STR = {"T": 2}

OUTPUT_PATH = f"data/test_{TARGET_DATE}_allstock_trades.bin"

# Binary record: int64 ts_event_ns | uint32 instrument_id | uint8 action |
#                uint8 pad[3]      | int64 price           | uint32 size  | uint32 pad2
RECORD_FMT = "<qIB3xqII"  # 32 bytes
assert struct.calcsize(RECORD_FMT) == 32


# ----
# Streaming loader
# ----

def load_events_streaming(symbol: str, sc_id: int, filepath: str) -> list:
    """Stream MBO records for one symbol, return only target-day session trade events."""
    print(f"  Streaming {symbol} (id={sc_id}) from {os.path.basename(filepath)} ...", flush=True)

    events = []

    def on_record(rec):
        ts = rec.ts_event
        if ts < OPEN_NS or ts >= CLOSE_NS:
            return
        act_char = rec.action.value if hasattr(rec.action, "value") else str(rec.action)
        action_byte = ACTION_STR.get(act_char)
        if action_byte is None:
            return
        events.append((ts, sc_id, action_byte, int(rec.price), int(rec.size)))

    store = db.DBNStore.from_file(filepath)
    store.replay(on_record)

    print(f"  {symbol}: {len(events):,} trade events on {TARGET_DATE}", flush=True)
    return events


# ----
# Main
# ----

def main():
    print(f"Session window: {OPEN_NS} - {CLOSE_NS} ns UTC")
    print(f"  = {TARGET_DATE} 09:30-16:00 ET\n")

    stocks = discover_stocks()
    if not stocks:
        print("ERROR: no stocks found in raw directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(stocks)} stocks: {', '.join(stocks.keys())}\n")

    # Write ID mapping so downstream code knows which id = which symbol
    id_to_sym = {i: sym for i, sym in enumerate(stocks.keys())}
    sym_to_id = {sym: i for i, sym in id_to_sym.items()}

    os.makedirs("data", exist_ok=True)
    import json
    with open(f"data/test_{TARGET_DATE}_allstock_id_map.json", "w") as f:
        json.dump({str(i): sym for i, sym in id_to_sym.items()}, f, indent=2)
    print(f"ID map written to data/test_{TARGET_DATE}_allstock_id_map.json\n")

    all_events = []
    for sym, filepath in stocks.items():
        sc_id = sym_to_id[sym]
        events = load_events_streaming(sym, sc_id, filepath)
        all_events.extend(events)

    if not all_events:
        print("ERROR: no events loaded.", file=sys.stderr)
        sys.exit(1)

    all_events.sort(key=lambda e: e[0])
    print(f"\nTotal events after merge-sort: {len(all_events):,}")

    with open(OUTPUT_PATH, "wb") as f:
        for ts, iid, act, price, size in all_events:
            f.write(struct.pack(RECORD_FMT, ts, iid, act, price, size, 0))

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"Written: {OUTPUT_PATH}  ({file_size / 1024 / 1024:.1f} MB, "
          f"{len(all_events):,} records x 32 bytes)")

    from collections import Counter
    counts = Counter(e[1] for e in all_events)
    print(f"\nPer-symbol trade counts:")
    for iid, cnt in sorted(counts.items()):
        print(f"  {id_to_sym[iid]:6s} (id={iid}): {cnt:,}")

    ts_min, ts_max = all_events[0][0], all_events[-1][0]
    print(f"\nTime span: {(ts_max - ts_min) / 1e9 / 3600:.2f} hours")


if __name__ == "__main__":
    main()
