#!/bin/bash
# GPU half of the training sweep: TabPFN v3 only — the sole learner that needs a
# GPU. One task per target (5 folds x 11 fractions of pure inference-style fits),
# which is minutes per target, so a GPU is held only while it is actually used
# instead of idling through hours of AutoGluon.
#
# Submit via slurm/submit_fast.sh (needs RUN_DIR + CV_SCHEME exported).
#
#SBATCH --job-name=train_gpu
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_gpu_%A_%a.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use submit_fast.sh, not a bare sbatch)}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

python -u run_train_cell.py --target-index "${SLURM_ARRAY_TASK_ID}" \
                           --shard-tag gpu \
                           --models "TabPFN" \
                           --run-dir "${RUN_DIR}" \
                           --cv-scheme "${CV_SCHEME}" \
                           --no-keep-models
