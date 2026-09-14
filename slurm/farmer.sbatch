#!/bin/bash
# EXTERNAL (commercial-cohort) validation — Fig. 8. Retrains AutoGluon + TabPFN
# on the 15 features the grower can record (Strategy A) and reads the
# controlled-CV cloud from <RUN_DIR>/{metrics,predictions}, so this MUST run
# after train_merge. Needs FARMER_PATH (control table) — set in env.sh.
#
# GPU node: the TabPFN refit wants CUDA.
#
#SBATCH --job-name=farmer_val
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/farmer_val_%j.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported}"

python -u validate_farmer.py --run-dir "${RUN_DIR}"
