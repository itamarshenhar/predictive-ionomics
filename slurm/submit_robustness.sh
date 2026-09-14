#!/bin/bash
# Submit the SHAP robustness sweep (supplementary Table S6) for a finished run.
#
#   RUN_DIR=<run dir> bash slurm/submit_robustness.sh
#   bash slurm/submit_robustness.sh dobscv     # read the run dir from its marker
#
# Prerequisite: the fold-0/1/2 AutoGluon models for the four representative
# targets (TRN, Ca, Fe, Mo) must be present under <RUN_DIR>/models. If the run
# dir was cleaned up, restore them first with stage_robustness_models.sh (Mac).
#
#   shap_rob_pfn[0-27]  (GPU) ─┐
#                              ├─afterok─> analyze_robustness -> Table S6
#   shap_rob_ag[0-27]   (CPU) ─┘
#
# 28 cells per model = 4 targets x 7 settings, perturbing one axis at a time from
# a 150-instance / 2048-coalition / K=5 / fold-0 baseline:
#   coalitions 1024 | background K=10, K=20 | explained set B | folds 1, 2
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

CV_SCHEME="${1:-${CV_SCHEME:-dobscv}}"
if [ -z "${RUN_DIR:-}" ]; then
  MARKER="${PROJECT}/slurm/CURRENT_RUN_DIR_${CV_SCHEME}"
  [ -s "${MARKER}" ] || { echo "ERROR: set RUN_DIR, or populate ${MARKER}" >&2; exit 1; }
  RUN_DIR="$(cat "${MARKER}")"
fi

# The four robustness targets need folds 0,1,2 — fail early and specifically.
MISSING=0
for t in TRN_output Ca_output Fe_output Mo_output; do
  for f in 0 1 2; do
    [ -f "${RUN_DIR}/models/${t}/Fold_${f}_Step_1.00/predictor.pkl" ] || {
      echo "MISSING: ${t} fold ${f}"; MISSING=1; }
  done
done
if [ "${MISSING}" = "1" ]; then
  echo >&2
  echo "ERROR: required models absent under ${RUN_DIR}/models." >&2
  echo "Restore them from the local archive:  bash stage_robustness_models.sh" >&2
  exit 1
fi
echo "CV scheme = ${CV_SCHEME}"
echo "RUN_DIR   = ${RUN_DIR}"
echo "models    = all 12 required (4 targets x folds 0,1,2) present"

python make_manifests.py >/dev/null
echo "manifests regenerated (rob_tabpfn.csv, rob_ag.csv: 28 cells each)"

EXP="ALL,RUN_DIR=${RUN_DIR},CV_SCHEME=${CV_SCHEME}"
sid() { sbatch "$@" | awk '{print $NF}'; }

ROB_GPU_PARTITION="${ROB_GPU_PARTITION:-salmon}"
ROB_GPU_GRES="${ROB_GPU_GRES:-gpu:l40s:1}"
ROB_CPU_PARTITION="${ROB_CPU_PARTITION:-glacier}"

J_PFN=$(sid --export="${EXP}" --partition="${ROB_GPU_PARTITION}" \
            --gres="${ROB_GPU_GRES}" slurm/shap_robustness_tabpfn.sbatch)
echo "  rob TabPFN   array ${J_PFN} (${ROB_GPU_PARTITION})"
J_AG=$(sid --export="${EXP}" --partition="${ROB_CPU_PARTITION}" \
           slurm/shap_robustness_ag.sbatch)
echo "  rob AG       array ${J_AG} (${ROB_CPU_PARTITION})"

J_ANA=$(sid --export="${EXP}" --partition="${ROB_CPU_PARTITION}" \
            --dependency=afterok:${J_PFN}:${J_AG} \
            --job-name=shap_robana --cpus-per-task=4 --mem=16G --time=00:20:00 \
            --output=logs/shap_robana_%j.log \
            --wrap="cd ${PROJECT} && source ${VENV}/bin/activate && \
                    RUN_DIR=${RUN_DIR} python analyze_robustness.py")
echo "  analysis     job ${J_ANA} (after both arrays)"
echo
echo "Table S6 will be written to:"
echo "  ${RUN_DIR}/shap_runs/robustness_rank_correlations.csv"
echo "Track:  squeue -u \$USER   |   tail -f ${PROJECT}/logs/shap_rob_pfn_${J_PFN}_*.log"
