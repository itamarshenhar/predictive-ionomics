#!/bin/bash
# Regenerate the publication SHAP figures (HANDOFF §11) AFTER both main arrays
# (TabPFN + AutoGluon) have finished. Reads the canonical <RUN_DIR>/shap/ tree.
# CPU-only (just plotting).
#
# Submit chained on the main jobs (see submit_all.sh), or standalone:
#   sbatch slurm/make_figures.sbatch
#
#SBATCH --job-name=shap_figs
#SBATCH --partition=glacier         # CPU partition (38 nodes)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/shap_figs_%j.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"   # SLURM_SUBMIT_DIR = dir you ran sbatch from
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use slurm/submit_shap.sh)}"

python - <<PY
import os
from shap_module import render_article_figures
render_article_figures(os.environ["RUN_DIR"])      # Figs. 5, 6, 7
print("FIGURES DONE")
PY
