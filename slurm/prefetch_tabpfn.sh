#!/bin/bash
# Pre-download the TabPFN v3 checkpoint on a LOGIN node into lab-storage cache,
# so the GPU compute nodes (which may lack internet) don't try to download it.
# Run once after setup_env.sh:   bash slurm/prefetch_tabpfn.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
source "${VENV}/bin/activate"

python - <<'PY'
from tabpfn import TabPFNRegressor
m = TabPFNRegressor.create_default_for_version("v3")
print("TabPFN v3 checkpoint cached at:", m.model_path)
PY

echo
echo "Confirm nothing landed in home (~/.cache should be a symlink to lab):"
ls -ld "${HOME}/.cache"
du -sh "${XDG_CACHE_HOME}" 2>/dev/null || true
