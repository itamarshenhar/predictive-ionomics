#!/bin/bash
# ROBUSTNESS sweep, TabPFN, on GPU. 4 targets x 7 settings = array 0-27.
# Longest cell is K=20 x 2048 at n_explain=150 (~12 h L40 / ~8 h A100 / ~3 h
# H200); the 18 h walltime covers L40. %8 caps concurrency (use %1 for H200).
#
# Submit:  sbatch slurm/shap_robustness_tabpfn.sbatch
#
#SBATCH --job-name=shap_rob_pfn
#SBATCH --partition=salmon          # overridden by submit_robustness.sh
#SBATCH --gres=gpu:l40s:1           # overridden by submit_robustness.sh
#SBATCH --array=0-27%28
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/shap_rob_pfn_%A_%a.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"   # SLURM_SUBMIT_DIR = dir you ran sbatch from
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use slurm/submit_robustness.sh)}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

# --cv-scheme MUST match the scheme the explained models were trained with; the
# fold-perturbation settings (rob_fold1/rob_fold2) are meaningless otherwise.
python -u run_shap_cell.py --manifest manifests/rob_tabpfn.csv \
                           --row "${SLURM_ARRAY_TASK_ID}" \
                           --run-dir "${RUN_DIR}" \
                           --cv-scheme "${CV_SCHEME}"
