#!/bin/bash
#SBATCH --job-name=mm_backtest
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=01:00:00
#SBATCH --output=logs/backtest_%j.out
#SBATCH --error=logs/backtest_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause

echo "=== Multi-day backtest ===" && date

$PYTHON scripts/backtest_multiday.py \
    --lob_dir data/lob \
    --pipeline_root output/multiday \
    --id_map data/allstock_id_map.json \
    --out output/mm_backtest_multiday \
    --is_end 2025-12-01 \
    --start 2025-10-01 \
    --end 2026-01-01

echo "Done." && date
