#!/bin/bash
#SBATCH --job-name=streamcause_s34b
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=06:00:00
#SBATCH --output=logs/steps34b_%j.out
#SBATCH --error=logs/steps34b_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
START=2025-10-01
END=2026-01-01
OUTPUT_ROOT=output/multiday
ID_MAP=data/allstock_id_map.json
TRADES_BY_DAY=data/trades_by_day

mkdir -p logs config/days output/multiday output/mm_backtest_multiday "$TRADES_BY_DAY"

echo "========================================"
echo "StreamCause Steps 3b+4 (per-day binaries)"
echo "Start: $START  End: $END"
echo "========================================"
date

# ── Step 3a: Split full trades binary into per-day files ──────────────────────
echo ""
echo "=== Step 3a: Split trades binary by day ==="
$PYTHON scripts/split_trades_by_day.py \
    --input data/allstock_trades_20251001_20260101.bin \
    --outdir "$TRADES_BY_DAY" \
    --start "$START" \
    --end "$END"
echo "Split done." && date

# ── Step 3b: C++ pipeline per day ────────────────────────────────────────────
echo ""
echo "=== Step 3b: C++ pipeline (63 days × per-day binary) ==="

export OUTPUT_ROOT START END TRADES_BY_DAY

$PYTHON - <<'PYEOF'
import json, datetime, os, subprocess, glob as _glob

BINARY      = "build-linux/streamcause"
BASE_CONFIG = "config/test_allstock.json"
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/multiday")
START       = os.environ.get("START", "2025-10-01")
END         = os.environ.get("END",   "2026-01-01")
TRADES_DIR  = os.environ.get("TRADES_BY_DAY", "data/trades_by_day")

US_HOLIDAYS = {datetime.date(2025,11,27), datetime.date(2025,11,28), datetime.date(2025,12,25)}

def trading_days(s, e):
    days, d = [], s
    while d < e:
        if d.weekday() < 5 and d not in US_HOLIDAYS:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days

with open(BASE_CONFIG) as f:
    base_cfg = json.load(f)

days = trading_days(datetime.date.fromisoformat(START), datetime.date.fromisoformat(END))
done, skip, fail = 0, 0, 0

for day in days:
    date_str     = day.strftime("%Y-%m-%d")
    date_compact = day.strftime("%Y%m%d")
    out_dir      = os.path.join(OUTPUT_ROOT, date_compact)
    day_subdir   = os.path.join(out_dir, date_str)

    if os.path.isdir(day_subdir) and _glob.glob(os.path.join(day_subdir, "lambda_*.bin")):
        n = len(_glob.glob(os.path.join(day_subdir, "lambda_*.bin")))
        print(f"  {date_str}: skip ({n} λ files)")
        skip += 1
        continue

    trades_bin = os.path.join(TRADES_DIR, f"{date_compact}.bin")
    if not os.path.exists(trades_bin):
        print(f"  {date_str}: SKIP — no per-day trades binary")
        skip += 1
        continue

    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(base_cfg)
    cfg["output_dir"] = out_dir
    cfg_path = f"config/days/config_{date_compact}.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    next_day = (day + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [BINARY, "--config", cfg_path, "--mode", "replay",
           "--file", trades_bin, "--start", date_str, "--end", next_day]

    print(f"  {date_str}: running...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        print(f"FAILED\n    stderr: {result.stderr[-400:]}")
        fail += 1
    else:
        lam = _glob.glob(os.path.join(day_subdir, "lambda_*.bin"))
        print(f"OK ({len(lam)} λ files)")
        done += 1

print(f"\nPipeline: {done} done, {skip} skipped, {fail} failed")
PYEOF

echo "C++ pipeline done." && date

# ── Step 4: Multi-day backtest ────────────────────────────────────────────────
echo ""
echo "=== Step 4: Multi-day backtest ==="
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause

$PYTHON scripts/backtest_multiday.py \
    --lob_dir data/lob \
    --pipeline_root "$OUTPUT_ROOT" \
    --id_map "$ID_MAP" \
    --out output/mm_backtest_multiday \
    --is_end 2025-12-01 \
    --start "$START" \
    --end "$END"

echo ""
echo "========================================"
echo "All steps complete."
date
echo "========================================"
