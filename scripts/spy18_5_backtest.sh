#!/bin/bash
#SBATCH --job-name=spy18_bt
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=48:00:00
#SBATCH --output=logs/spy18_bt_%j.out
#SBATCH --error=logs/spy18_bt_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause
export PYTHONUNBUFFERED=1

echo "=== SPY18 Step 5: OOS Backtest (all 18 instruments) ===" && date

$PYTHON scripts/backtest_lob_precise.py \
    --raw_dir      /ocean/projects/cis260122p/shared/data/raw \
    --pipeline_root output/multiday_spy18 \
    --id_map       data/spy18_id_map.json \
    --calibrated_k output/mm_backtest_spy18/calibrated_k.csv \
    --out          output/mm_backtest_precise_spy18 \
    --is_end       2025-12-01 \
    --start        2025-10-01 \
    --end          2026-01-01 \
    --mm_size      1

echo "Done." && date
