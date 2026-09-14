#!/bin/bash
# Submit the SHAP stage for a finished training run — or for BOTH CV schemes.
#
#   bash slurm/submit_shap.sh both       # DOB-SCV + SGK
#   bash slurm/submit_shap.sh dobscv     # one scheme
#   RUN_DIR=<explicit dir> CV_SCHEME=dobscv bash slurm/submit_shap.sh
#
# Run this only AFTER that scheme's training chain has completed (it needs the
# saved AutoGluon Fold_0_Step_1.00 models). The run dir is read from the marker
# slurm/CURRENT_RUN_DIR_<scheme> written by submit_fast.sh.
#
#   shap_main_pfn[0-11]  (GPU) ─┐
#                               ├─afterok─> make_figures
#   shap_main_ag[0-11]   (CPU) ─┘
#
# Config is the heavy publication setting from config.py
# (250 explained / 2048 coalitions / K=5 / TabPFN-32). AutoGluon and TabPFN use
# the SAME config so the model-comparison figure stays valid.
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "both" ]; then
  echo "======================= SHAP: DOB-SCV ========================"
  bash "${HERE}/submit_shap.sh" dobscv
  echo
  echo "======================= SHAP: SGK ============================"
  bash "${HERE}/submit_shap.sh" grouped
  exit 0
fi

source "${HERE}/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

CV_SCHEME="${1:-${CV_SCHEME:-dobscv}}"
if [ -z "${RUN_DIR:-}" ]; then
  MARKER="${PROJECT}/slurm/CURRENT_RUN_DIR_${CV_SCHEME}"
  [ -s "${MARKER}" ] || { echo "ERROR: ${MARKER} missing/empty — has ${CV_SCHEME} training run?" >&2; exit 1; }
  RUN_DIR="$(cat "${MARKER}")"
fi
# The saved fold-0 AutoGluon models are the input SHAP explains.
if ! ls "${RUN_DIR}"/models/*/Fold_0_Step_1.00/predictor.pkl >/dev/null 2>&1; then
  echo "ERROR: no Fold_0_Step_1.00 models under ${RUN_DIR}/models — training incomplete?" >&2
  exit 1
fi
echo "CV scheme = ${CV_SCHEME}"
echo "RUN_DIR   = ${RUN_DIR}"

# Manifests are scheme-independent (they only enumerate target x model cells).
# Generate once here so two concurrent scheme submits never race on the files.
python make_manifests.py >/dev/null
echo "manifests regenerated"

EXP="ALL,RUN_DIR=${RUN_DIR},CV_SCHEME=${CV_SCHEME}"
sid() { sbatch "$@" | awk '{print $NF}'; }

SHAP_GPU_PARTITION="${SHAP_GPU_PARTITION:-salmon}"
SHAP_GPU_GRES="${SHAP_GPU_GRES:-gpu:l40s:1}"
SHAP_CPU_PARTITION="${SHAP_CPU_PARTITION:-glacier}"

J_PFN=$(sid --export="${EXP}" --partition="${SHAP_GPU_PARTITION}" \
            --gres="${SHAP_GPU_GRES}" slurm/shap_main_tabpfn.sbatch)
echo "  shap TabPFN  array ${J_PFN} (${SHAP_GPU_PARTITION})"
J_AG=$(sid --export="${EXP}" --partition="${SHAP_CPU_PARTITION}" \
           slurm/shap_main_ag.sbatch)
echo "  shap AG      array ${J_AG} (${SHAP_CPU_PARTITION})"

J_FIG=$(sid --export="${EXP}" --partition="${SHAP_CPU_PARTITION}" \
            --dependency=afterok:${J_PFN}:${J_AG} slurm/make_figures.sbatch)
echo "  shap figures job ${J_FIG} (after both arrays)"
echo
echo "Track:  squeue -u \$USER   |   tail -f ${PROJECT}/logs/shap_main_pfn_${J_PFN}_*.log"
