#!/bin/bash
# MAIN run, AutoGluon, on CPU (Glacier). 12 targets = array 0-11. AutoGluon SHAP
# is CPU-bound and GPU-independent (it reuses the saved AG fold models), so run
# it concurrently with the TabPFN GPU array.
#
# Submit:  sbatch slurm/shap_main_ag.sbatch
#
#SBATCH --job-name=shap_main_ag
#SBATCH --partition=glacier         # CPU partition (38 nodes)
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/shap_main_ag_%A_%a.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"   # SLURM_SUBMIT_DIR = dir you ran sbatch from
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use slurm/submit_shap.sh)}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

# --cv-scheme MUST match the scheme the explained models were trained with, or
# the reconstructed fold-0 training split won't match the saved fold-0 model.
python -u run_shap_cell.py --manifest manifests/main_ag.csv \
                           --row "${SLURM_ARRAY_TASK_ID}" \
                           --run-dir "${RUN_DIR}" \
                           --cv-scheme "${CV_SCHEME}"
