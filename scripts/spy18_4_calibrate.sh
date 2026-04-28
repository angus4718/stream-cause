#!/bin/bash
#SBATCH --job-name=spy18_cal
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=02:00:00
#SBATCH --output=logs/spy18_cal_%j.out
#SBATCH --error=logs/spy18_cal_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause
export PYTHONUNBUFFERED=1

echo "=== SPY18 Step 4: IS Calibration ===" && date

$PYTHON scripts/backtest_multiday.py \
    --lob_dir      data/lob_spy18 \
    --pipeline_root output/multiday_spy18 \
    --id_map       data/spy18_id_map.json \
    --out          output/mm_backtest_spy18 \
    --is_end       2025-12-01 \
    --start        2025-10-01 \
    --end          2026-01-01 \
    --session_filter

echo "Done." && date
