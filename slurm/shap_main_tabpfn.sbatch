#!/bin/bash
# MAIN run, TabPFN, on GPU. 12 targets = array 0-11, one target/task.
# Per-target ~5 h on L40 / ~3.4 h A100 / ~1.4 h H200 (250 / 2048 / K5 / 32-est).
# %8 caps concurrency at 8 (fine for plentiful L40; lower to %1 if using the
# single H200, or %2 for A100).
#
# Submit:  sbatch slurm/shap_main_tabpfn.sbatch
#
#SBATCH --job-name=shap_main_pfn
#SBATCH --partition=salmon          # overridden by submit_shap.sh (SHAP_GPU_PARTITION)
#SBATCH --gres=gpu:l40s:1           # overridden by submit_shap.sh (SHAP_GPU_GRES)
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/shap_main_pfn_%A_%a.log
# NOTE: #SBATCH lines cannot read env vars. Either edit the two VERIFY lines
# above once, or override at submit time:
#   sbatch --partition=$GPU_PARTITION --gres=$GPU_GRES slurm/shap_main_tabpfn.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"   # SLURM_SUBMIT_DIR = dir you ran sbatch from
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use slurm/submit_shap.sh)}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

# --cv-scheme MUST match the scheme the explained models were trained with, or
# the reconstructed fold-0 training split won't match the saved fold-0 model.
python -u run_shap_cell.py --manifest manifests/main_tabpfn.csv \
                           --row "${SLURM_ARRAY_TASK_ID}" \
                           --run-dir "${RUN_DIR}" \
                           --cv-scheme "${CV_SCHEME}"
