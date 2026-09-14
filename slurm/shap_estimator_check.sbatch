#!/bin/bash
# TabPFN SHAP ensemble-size stability check (supplementary): compare per-feature
# mean|SHAP| at 8 vs 32 estimators on the SAME fold-0 training split, for the two
# representative targets (TRN, Ca).
#
# Self-contained: TabPFN is fitted from scratch, so this needs only the code and
# the dataset — no saved AutoGluon models, hence no run dir to restore.
#
# Deliberately a single job rather than an array: the script produces one
# two-panel figure and one combined CSV, so splitting by target would require a
# merge step for ~1 h of saving. Both targets serially is ~2 h on an L40S.
#
# Submit:  RUN_DIR=<dir> bash slurm/submit_estimator_check.sh
#
#SBATCH --job-name=shap_estcheck
#SBATCH --partition=salmon          # overridden by the submit script
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/shap_estcheck_%j.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

: "${RUN_DIR:?RUN_DIR must be exported (use slurm/submit_estimator_check.sh)}"
export CV_SCHEME="${CV_SCHEME:-dobscv}"
export OUT_DIR="${RUN_DIR}/plots"
mkdir -p "${OUT_DIR}"

echo "cv_scheme=${CV_SCHEME}  out=${OUT_DIR}"
python -u shap_estimator_check.py
