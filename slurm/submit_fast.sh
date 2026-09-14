#!/bin/bash
# Submit the training sweep for one or both CV schemes, split by what each
# learner needs:
#   train_cpu  glacier, 60 tasks (12 targets x 5 folds)  AutoGluon
#   train_gpu  salmon,  12 tasks (1 per target)          TabPFN only
# 60 x 32 = 1920 cores fits inside glacier's 4864, so every cell runs at once and
# wall-clock is one (target, fold) cell (~1 h) instead of a 12-task queue.
#
#   train_cpu[0-59] ─┐
#                    ├─afterok─> merge ─> plots ─> farmer
#   train_gpu[0-11] ─┘
#
#   bash slurm/submit_fast.sh                    # DOB-SCV, full pipeline
#   CV_SCHEME=grouped bash slurm/submit_fast.sh  # the SGK comparison run
#   bash slurm/submit_fast.sh both               # both schemes back to back
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "both" ]; then
  echo "===================== DOB-SCV (headline) ====================="
  CV_SCHEME=dobscv bash "${HERE}/submit_fast.sh"
  echo
  echo "================ StratifiedGroupKFold (reserve) ==============="
  CV_SCHEME=grouped bash "${HERE}/submit_fast.sh"
  exit 0
fi

source "${HERE}/env.sh"
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p logs

CV_SCHEME="${CV_SCHEME:-dobscv}"
RUN_DIR="$(python -c "import config; print(config.make_run_dir(prefix='Ionomic_Prediction_Run_${CV_SCHEME}'))")"
echo "Fresh RUN_DIR = ${RUN_DIR}"
echo "CV scheme     = ${CV_SCHEME}"
echo "${RUN_DIR}" > "${PROJECT}/slurm/CURRENT_RUN_DIR_${CV_SCHEME}"

EXP="ALL,RUN_DIR=${RUN_DIR},CV_SCHEME=${CV_SCHEME}"
sid() { sbatch "$@" | awk '{print $NF}'; }

# Overridable at submit time, e.g. to fall back to goldfish/H200:
#   TRAIN_GPU_PARTITION=goldfish TRAIN_GPU_GRES=gpu:h200:1 bash slurm/submit_fast.sh
# (salmon = 10 nodes x 8 L40S: far more schedulable slots than goldfish's single
#  node, and TabPFN here is short — GPU speed is not the constraint.)
TRAIN_CPU_PARTITION="${TRAIN_CPU_PARTITION:-glacier}"
TRAIN_GPU_PARTITION="${TRAIN_GPU_PARTITION:-salmon}"
TRAIN_GPU_GRES="${TRAIN_GPU_GRES:-gpu:l40s:1}"

J_CPU=$(sid --export="${EXP}" --partition="${TRAIN_CPU_PARTITION}" \
            slurm/train_cpu.sbatch)
echo "  train_cpu  array ${J_CPU}  (60 tasks: AutoGluon, ${TRAIN_CPU_PARTITION})"
J_GPU=$(sid --export="${EXP}" --partition="${TRAIN_GPU_PARTITION}" \
            --gres="${TRAIN_GPU_GRES}" slurm/train_gpu.sbatch)
echo "  train_gpu  array ${J_GPU}  (12 tasks: TabPFN, ${TRAIN_GPU_PARTITION})"

J_MERGE=$(sid --export="${EXP}" --dependency=afterok:${J_CPU}:${J_GPU} \
              slurm/train_merge.sbatch)
echo "  train_merge job ${J_MERGE} (after BOTH arrays)"

J_PLOT=$(sid --export="${EXP}" --dependency=afterok:${J_MERGE} slurm/plots.sbatch)
echo "  core_plots  job ${J_PLOT}"

J_FARM=$(sid --export="${EXP}" --partition="${TRAIN_GPU_PARTITION}" \
             --gres="${TRAIN_GPU_GRES}" --dependency=afterok:${J_MERGE} \
             slurm/farmer.sbatch)
echo "  farmer_val  job ${J_FARM} (${TRAIN_GPU_PARTITION})"

echo
echo "SHAP is NOT submitted here (heavy; separate step once approved)."
echo "Track:  squeue -u \$USER   |   tail -f ${PROJECT}/logs/train_cpu_${J_CPU}_*.log"
