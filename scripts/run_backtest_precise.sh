#!/bin/bash
#SBATCH --job-name=mm_precise
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=04:00:00
#SBATCH --output=logs/precise_%j.out
#SBATCH --error=logs/precise_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause

echo "=== Precise LOB MM Backtest ===" && date

$PYTHON scripts/backtest_lob_precise.py \
    --raw_dir     /ocean/projects/cis260122p/shared/data/raw \
    --pipeline_root output/multiday \
    --id_map      data/allstock_id_map.json \
    --calibrated_k output/mm_backtest_multiday/calibrated_k.csv \
    --out         output/mm_backtest_precise \
    --is_end      2025-12-01 \
    --start       2025-10-01 \
    --end         2026-01-01 \
    --mm_size     1

echo "Done." && date
