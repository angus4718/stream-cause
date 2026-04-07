#!/bin/bash
#SBATCH --job-name=causal_graph
#SBATCH --account=cis260122p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=00:30:00
#SBATCH --output=logs/causal_%j.out
#SBATCH --error=logs/causal_%j.err

set -euo pipefail
cd /ocean/projects/cis260122p/ccheung1/stream-cause

PYTHON=/ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3
export PYTHONPATH=/ocean/projects/cis260122p/ccheung1/stream-cause

echo "=== Causal Graph Analysis ===" && date

$PYTHON scripts/analyze_causal_graph.py \
    --pipeline_root output/multiday \
    --id_map        data/allstock_id_map.json \
    --per_inst_csv  output/mm_backtest_multiday/per_inst.csv \
    --out           output/causal_graph_analysis \
    --is_end        2025-12-01 \
    --start         2025-10-01 \
    --end           2026-01-01

echo "Done." && date
