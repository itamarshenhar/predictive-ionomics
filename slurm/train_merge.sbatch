#!/bin/bash
# MERGE the 12 per-target training shards into the canonical <RUN_DIR>/ tree
# (metrics/predictions/architecture/importance CSVs + models/). CPU-only, quick.
# Runs after both training arrays finish OK (chained by submit_fast.sh).
#
#SBATCH --job-name=train_merge
#SBATCH --partition=glacier         # CPU partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_merge_%j.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported}"
CV_SCHEME="${CV_SCHEME:-dobscv}"

python -u merge_train_shards.py --run-dir "${RUN_DIR}" --cv-scheme "${CV_SCHEME}"
