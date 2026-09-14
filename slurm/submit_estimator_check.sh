#!/bin/bash
# Submit the TabPFN 8-vs-32 estimator stability check (supplementary Table S7 /
# SHAP_Estimator_Stability figure).
#
#   bash slurm/submit_estimator_check.sh              # writes into a fresh dir
#   RUN_DIR=<existing dir> bash slurm/submit_estimator_check.sh
#
# Needs only code + dataset (TabPFN is fitted from scratch), so it runs even
# after the training run dirs have been cleaned off the cluster.
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

CV_SCHEME="${CV_SCHEME:-dobscv}"
# No run dir needed as input — just somewhere to put the outputs.
if [ -z "${RUN_DIR:-}" ]; then
  RUN_DIR="${PROJECT}/estimator_check_${CV_SCHEME}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${RUN_DIR}/plots"
  echo "No RUN_DIR given — created ${RUN_DIR}"
fi
echo "CV scheme = ${CV_SCHEME}"
echo "outputs   -> ${RUN_DIR}/plots"

EST_GPU_PARTITION="${EST_GPU_PARTITION:-salmon}"
EST_GPU_GRES="${EST_GPU_GRES:-gpu:l40s:1}"

JID=$(sbatch --export="ALL,RUN_DIR=${RUN_DIR},CV_SCHEME=${CV_SCHEME}" \
             --partition="${EST_GPU_PARTITION}" --gres="${EST_GPU_GRES}" \
             slurm/shap_estimator_check.sbatch | awk '{print $NF}')
echo "  shap_estcheck job ${JID} (${EST_GPU_PARTITION})"
echo
echo "Produces:"
echo "  ${RUN_DIR}/plots/SHAP_Estimator_Stability.{pdf,png}"
echo "  ${RUN_DIR}/plots/shap_estimator_check_values.csv"
echo "Track:  tail -f ${PROJECT}/logs/shap_estcheck_${JID}.log"
