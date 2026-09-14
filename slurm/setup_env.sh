#!/bin/bash
# One-time environment setup on a HURCS LOGIN node (moriah-gw). Creates a venv in
# lab storage (NOT home) and installs the pinned stack with the CUDA build of
# torch. Run once:   bash slurm/setup_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

echo "LAB_ROOT = ${LAB_ROOT}"
mkdir -p "${LAB_ROOT}/envs" "${LAB_ROOT}/data"

# Keep ALL python/HF/torch caches off the 5 GB home: symlink ~/.cache -> lab.
# (Belt-and-suspenders alongside the XDG_CACHE_HOME export in env.sh.)
if [ ! -L "${HOME}/.cache" ]; then
  rm -rf "${HOME}/.cache" 2>/dev/null || true
  ln -s "${XDG_CACHE_HOME}" "${HOME}/.cache"
  echo "symlinked ~/.cache -> ${XDG_CACHE_HOME}"
fi

# Python: use a 3.11 interpreter. If the cluster exposes modules, load one;
# otherwise the system python3.11 is fine. VERIFY with: module avail python
# module load python/3.11   # <-- uncomment + fix name if your cluster needs it
PYBIN="$(command -v python3.11 || command -v python3)"
echo "using ${PYBIN} ($(${PYBIN} --version))"

"${PYBIN}" -m venv "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

# CUDA torch + torchvision FIRST, from the CUDA index (NOT the default/MPS wheel).
# torch 2.9.1 is NOT on the cu124 index (max there is 2.6.0); 2.6.0+cu124 runs on
# all HURCS GPUs (H200/A100/L40S/L4) and is what this run was validated on.
# torchvision must match torch and be installed from the same index, else fastai
# pulls a CPU torch and breaks CUDA. Bump cu124 if the cluster CUDA differs.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# The rest of the stack. fastai is required to LOAD AutoGluon's FastAI NN
# ensemble members; seaborn is required by plotting.py for the figure grids.
pip install \
  autogluon.tabular==1.5.0 \
  tabpfn==8.0.3 \
  shap==0.51.0 \
  scikit-learn==1.7.2 \
  numpy==2.1.3 \
  pandas==2.3.3 \
  lightgbm==4.6.0 \
  xgboost==3.1.3 \
  catboost==1.2.10 \
  openpyxl==3.1.5 \
  fastai \
  seaborn

echo
echo "Installed. Quick CUDA check (run on a GPU node, not here):"
echo "  srun -p \$GPU_PARTITION --gres=\$GPU_GRES --pty python -c \\"
echo "    'import torch;print(torch.__version__, torch.cuda.is_available())'"
