#!/usr/bin/env python3
"""
Run the StreamCause C++ pipeline for every trading day in a date range.

For each day:
  1. Writes a per-day config JSON (output_dir = output/lob/<YYYYMMDD>)
  2. Runs streamcause in replay mode on the pre-built allstock trades binary
  3. Skips days where lambda_*.bin files already exist

Usage:
    /ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3 \
        scripts/run_pipeline_all_days.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

BINARY     = "build-linux/streamcause"
TRADES_BIN = "data/test_20251201_allstock_trades.bin"   # the shared multi-day trades file
BASE_CONFIG = "config/test_allstock.json"
OUTPUT_ROOT = "output/multiday"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",      default="2025-10-01")
    parser.add_argument("--end",        default="2026-01-01")
    parser.add_argument("--trades_bin", default=TRADES_BIN)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--config",     default=BASE_CONFIG)
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs("config/days", exist_ok=True)

    with open(args.config) as f:
        base_cfg = json.load(f)

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)
    days       = trading_days(start_date, end_date)
    print(f"Pipeline run: {len(days)} trading days  ({days[0]} -> {days[-1]})")

    done, skipped, failed = 0, 0, 0
    for day in days:
        date_str = day.strftime("%Y-%m-%d")
        date_compact = day.strftime("%Y%m%d")

        out_dir   = os.path.join(args.output_root, date_compact)
        day_subdir = os.path.join(out_dir, date_str)

        # Skip if already has lambda files
        if os.path.isdir(day_subdir):
            lam_files = list(Path(day_subdir).glob("lambda_*.bin"))
            if lam_files:
                print(f"  {date_str}: skip ({len(lam_files)} lambda files exist)")
                skipped += 1
                continue

        os.makedirs(out_dir, exist_ok=True)

        # Write per-day config
        cfg = dict(base_cfg)
        cfg["output_dir"] = out_dir
        cfg_path = f"config/days/config_{date_compact}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        next_day = (day + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        cmd = [
            BINARY,
            "--config", cfg_path,
            "--mode", "replay",
            "--file", args.trades_bin,
            "--start", date_str,
            "--end", next_day,
        ]

        if args.dry_run:
            print(f"  {date_str}: DRY RUN  {' '.join(cmd)}")
            continue

        print(f"  {date_str}: running...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[-500:]}")
            failed += 1
        else:
            lam_files = list(Path(day_subdir).glob("lambda_*.bin"))
            print(f"    OK  ({len(lam_files)} lambda files)")
            done += 1

    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()
