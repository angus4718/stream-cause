#!/bin/bash
#SBATCH --job-name=spy18_pipeline
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=06:00:00
#SBATCH --output=logs/spy18_pipeline_%j.out
#SBATCH --error=logs/spy18_pipeline_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

date

# Clean any leftover files from aborted runs, then recreate empty dir.
rm -rf output/multiday_spy18
mkdir -p output/multiday_spy18

# Write config pointing to the flat output dir
python3 -c "
import json
cfg = json.load(open('config/spy18_allstock.json'))
cfg['output_dir'] = 'output/multiday_spy18'
json.dump(cfg, open('config/spy18_run.json','w'), indent=2)
"

build-linux/streamcause \
    --config config/spy18_run.json \
    --mode   replay \
    --file   data/spy18/allstock_trades_20251001_20260101.bin \
    --start  2025-10-01 \
    --end    2026-01-01

date

# Build per-day symlink structure:
#   output/multiday_spy18/{YYYYMMDD}/{YYYY-MM-DD} -> ../../../multiday_spy18/{YYYY-MM-DD}
# so that load_lambda_series(pipeline_root/YYYYMMDD, date_str) resolves correctly.
python3 - << 'PYEOF'
import os, datetime

def trading_days(start, end):
    days, d = [], start
    holidays = {datetime.date(2025,11,27), datetime.date(2025,11,28), datetime.date(2025,12,25)}
    while d < end:
        if d.weekday() < 5 and d not in holidays:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days

base = "output/multiday_spy18"
for d in trading_days(datetime.date(2025,10,1), datetime.date(2026,1,1)):
    date_str     = d.strftime("%Y-%m-%d")
    date_compact = d.strftime("%Y%m%d")
    src   = os.path.abspath(f"{base}/{date_str}")
    link_parent = f"{base}/{date_compact}"
    link_path   = f"{link_parent}/{date_str}"
    if not os.path.isdir(src):
        print(f"  WARNING: {src} missing, skipping symlink")
        continue
    os.makedirs(link_parent, exist_ok=True)
    if not os.path.exists(link_path):
        os.symlink(src, link_path)
PYEOF

date
