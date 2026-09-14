#!/bin/bash
# CORE PLOTS for the grouped-CV run (Actual_vs_Predicted, Learning_Curves,
# Permutation_Importance, Importance_Cluster_Heatmap, AutoGluon_Best_Model_Mix).
# CPU-only. Runs after train_merge (reads the canonical <RUN_DIR>/ CSVs).
#
#SBATCH --job-name=core_plots
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/core_plots_%j.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

# --cv-scheme MUST be passed: run.py rebuilds and overwrites <RUN_DIR>/fold_summary.csv
python -u run.py --stage plot --run-dir "${RUN_DIR}" --cv-scheme "${CV_SCHEME}"
