#!/bin/bash
#SBATCH --job-name=spy18_lob
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=02:00:00
#SBATCH --output=logs/spy18_lob_%j.out
#SBATCH --error=logs/spy18_lob_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause

echo "=== SPY18 Step 2: Preprocess LOB ===" && date

# Only run the per-symbol worker for SPY (reuse existing lob_tmp for others),
# then merge all 18 symbols into data/lob_spy18/
$PYTHON scripts/preprocess_lob_multiday.py \
    --start     2025-10-01 \
    --end       2026-01-01 \
    --id_map    data/spy18_id_map.json \
    --out       data/lob_spy18 \
    --tmp       data/lob_tmp \
    --workers   4 \
    --only_syms SPY

echo "Done." && date
