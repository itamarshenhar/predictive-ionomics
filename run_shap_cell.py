"""Compute ONE SHAP cell from a manifest row — the unit of a SLURM array task.

Usage (inside an sbatch):
    python -u run_shap_cell.py --manifest manifests/main_tabpfn.csv \
                               --row ${SLURM_ARRAY_TASK_ID}

It reads the row, reconstructs the exact fold split, and calls
shap_module.compute_shap_for_target for the single (model, target, setting) in
that row. Output lands in <RUN_DIR>/shap_runs/<setting_id>/<target>/.

Resumable: a per-cell marker .done_<model> is written on success and skipped on
restart, so an array that hits the walltime can simply be resubmitted.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import warnings

warnings.filterwarnings("ignore")

# The authoritative run directory (holds the saved AutoGluon fold models). Must
# come from the environment or --run-dir: NO fallback, so a misconfigured job
# cannot silently explain a different run's models.
RUN_DIR = os.environ.get("RUN_DIR") or None


def _read_row(manifest: str, row_id: int) -> dict:
    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))
    match = [r for r in rows if int(r["row_id"]) == row_id]
    if not match:
        raise SystemExit(f"row_id {row_id} not in {manifest} "
                         f"(has {len(rows)} rows, 0-{len(rows) - 1})")
    return match[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--row", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--cv-scheme",
                    choices=["dobscv", "grouped", "stratified", "loeo"],
                    default=os.environ.get("CV_SCHEME", "dobscv"),
                    help="MUST match the scheme the explained models were trained "
                         "with, or the reconstructed fold-0 training split will "
                         "not correspond to the saved fold-0 model.")
    args = ap.parse_args()
    if not args.run_dir:
        raise SystemExit("--run-dir (or RUN_DIR env) is required.")

    row = _read_row(args.manifest, args.row)
    model = row["model"]
    target = row["target"]
    fold = int(row["fold"])
    setting_id = row["setting_id"]
    # Main run lands in the canonical shap/ tree so the existing publication
    # figure functions work unchanged; robustness settings go to a side tree.
    shap_subdir = ("shap" if setting_id == "main"
                   else os.path.join("shap_runs", setting_id))

    out_dir = os.path.join(args.run_dir, shap_subdir, target)
    marker = os.path.join(out_dir, f".done_{model}")
    if os.path.exists(marker):
        print(f"[skip] {setting_id}/{target}/{model} already done", flush=True)
        return

    # Import heavy modules only after the fast skip check.
    import config
    from data import build_folds, load_dataset
    from shap_module import compute_shap_for_target

    print(f"device={config.DEVICE}  cores={config.NUM_CORES}  "
          f"cv_scheme={args.cv_scheme}  run_dir={args.run_dir}", flush=True)
    print(f"[cell] phase={row['phase']} setting={setting_id} target={target} "
          f"model={model} fold={fold} n_explain={row['n_explain']} "
          f"nsamples={row['nsamples']} K={row['k']} "
          f"n_est={row['n_estimators']} set={row['sample_set']}", flush=True)

    df = load_dataset()
    df_strat, folds = build_folds(df, n_splits=config.N_SPLITS,
                                  scheme=args.cv_scheme)
    train_idx = folds[fold][0]

    t0 = time.time()
    compute_shap_for_target(
        args.run_dir, df_strat, target,
        fold_idx=fold,
        n_background=int(row["n_background"]),
        n_explain=int(row["n_explain"]),
        models=[model],
        train_idx=train_idx,
        k=int(row["k"]),
        nsamples=int(row["nsamples"]),
        n_estimators=int(row["n_estimators"]),
        seed=int(row["seed"]),
        shap_subdir=shap_subdir,
    )
    os.makedirs(out_dir, exist_ok=True)
    open(marker, "w").close()
    print(f"[done] {setting_id}/{target}/{model} in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
