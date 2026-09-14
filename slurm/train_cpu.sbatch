#!/bin/bash
# CPU half of the training sweep: AutoGluon. It does not use a GPU and is ~95%
# of the runtime — so it runs on `glacier` (38 nodes x 128 cores), sharded by
# (target, fold) for maximum parallelism.
#
#   array index -> target = idx / 5 , fold = idx % 5     (12 targets x 5 folds = 60)
#
# 60 tasks x 32 CPUs = 1920 cores, well inside glacier's 4864, so ALL 60 run
# concurrently and wall-clock collapses to a single (target, fold) cell
# (AutoGluon: 3600 s budget at the full step, 900 s at each of the 9 sub-steps).
#
# Submit via slurm/submit_fast.sh (needs RUN_DIR + CV_SCHEME exported).
#
#SBATCH --job-name=train_cpu
#SBATCH --partition=glacier
#SBATCH --array=0-59%60
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_cpu_%A_%a.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use submit_fast.sh, not a bare sbatch)}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

TARGET_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))

python -u run_train_cell.py --target-index "${TARGET_IDX}" \
                           --fold "${FOLD}" \
                           --shard-tag cpu \
                           --models "AutoGluon" \
                           --run-dir "${RUN_DIR}" \
                           --cv-scheme "${CV_SCHEME}" \
                           --no-keep-models
