#!/bin/bash
#SBATCH --job-name=spy18_trades
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=01:30:00
#SBATCH --output=logs/spy18_trades_%j.out
#SBATCH --error=logs/spy18_trades_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause

echo "=== SPY18 Step 1: Preprocess Trades ===" && date
mkdir -p data/spy18

$PYTHON scripts/preprocess_trades_multiday.py \
    --start   2025-10-01 \
    --end     2026-01-01 \
    --id_map  data/spy18_id_map.json \
    --out_dir data/spy18 \
    --workers 8

echo "Done." && date
