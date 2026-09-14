#!/bin/bash
# ROBUSTNESS sweep, AutoGluon, on CPU (Glacier). 4 targets x 7 settings = 0-27.
# Cheap relative to TabPFN; run concurrently.
#
# Submit:  sbatch slurm/shap_robustness_ag.sbatch
#
#SBATCH --job-name=shap_rob_ag
#SBATCH --partition=glacier         # CPU partition (38 nodes)
#SBATCH --array=0-27%28
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/shap_rob_ag_%A_%a.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"   # SLURM_SUBMIT_DIR = dir you ran sbatch from
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use slurm/submit_robustness.sh)}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

# --cv-scheme MUST match the scheme the explained models were trained with; the
# fold-perturbation settings (rob_fold1/rob_fold2) are meaningless otherwise.
python -u run_shap_cell.py --manifest manifests/rob_ag.csv \
                           --row "${SLURM_ARRAY_TASK_ID}" \
                           --run-dir "${RUN_DIR}" \
                           --cv-scheme "${CV_SCHEME}"
