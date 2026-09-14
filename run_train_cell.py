"""Train ONE target's full sweep — the unit of a SLURM array task.

Usage (inside an sbatch):
    python -u run_train_cell.py --target-index ${SLURM_ARRAY_TASK_ID} \
                                --run-dir "$RUN_DIR" --cv-scheme grouped

One array task runs the complete 5-fold x 11-fraction x 2-model sweep for a
single target, writing into an ISOLATED per-target shard directory:
    <RUN_DIR>/_shards/<target>/{metrics,predictions,architecture,models}/

Isolation is deliberate. training.run_training streams to shared CSVs with a
plain append (not concurrency-safe), so 12 targets writing to one directory
would corrupt the CSVs. Each target writes its own shard; merge_train_shards.py
concatenates the shards into the canonical <RUN_DIR>/ tree afterwards.

Resumable twice over: run_training skips already-complete (fold, frac, model)
rows from the shard's own learning_curve.csv, and a per-target .done marker lets
the whole array be resubmitted after a walltime kill without redoing finished
targets.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-index", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ap.add_argument("--run-dir", default=os.environ.get("RUN_DIR"),
                    help="Shared run dir all array tasks agree on (from env.sh / submit script).")
    ap.add_argument("--cv-scheme",
                    choices=["dobscv", "grouped", "stratified", "loeo"],
                    default=os.environ.get("CV_SCHEME", "dobscv"))
    ap.add_argument("--no-keep-models", action="store_true",
                    help="Keep only step==1.0 model dirs (needed for SHAP/farmer); "
                         "purge intermediate sweep steps.")
    ap.add_argument("--fold", type=int, default=None,
                    help="Run ONLY this outer fold (shards a target by fold for "
                         "more parallelism). Default: all folds.")
    ap.add_argument("--models", default=None,
                    help="Comma-separated subset of learners: 'TabPFN' (GPU "
                         "array) or 'AutoGluon' (CPU array). Default: both.")
    ap.add_argument("--shard-tag", default=None,
                    help="Suffix that keeps concurrent shards of the same target "
                         "in separate directories (e.g. 'cpu' / 'gpu').")
    ap.add_argument("--smoke", action="store_true",
                    help="Trivial sweep (2 folds, {0,1} fractions, short AG "
                         "budgets) to prove the path end-to-end in minutes.")
    args = ap.parse_args()

    if not args.run_dir:
        raise SystemExit("--run-dir (or RUN_DIR env) is required so all array "
                         "tasks share one run directory.")

    import config
    if args.smoke:
        config.LEARNING_CURVE_FRACTIONS = [0.0, 1.0]
        config.N_SPLITS = 2
        config.AG_TIME_LIMIT_FULL = 60
        config.AG_TIME_LIMIT_SUB = 30
        print("[smoke] 2 folds, fractions {0.0, 1.0}, AG time 60/30s", flush=True)
    targets = config.TARGETS_AVAILABLE
    if not 0 <= args.target_index < len(targets):
        raise SystemExit(f"--target-index {args.target_index} out of range "
                         f"(0-{len(targets) - 1} for {len(targets)} targets)")
    target = targets[args.target_index]

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else None)
    fold_subset = None if args.fold is None else {args.fold}

    # One shard directory per concurrently-running unit. training.run_training
    # appends to shared CSVs and is NOT concurrency-safe, so every unit that can
    # run at the same time must own its own directory; merge_train_shards.py
    # stitches them back together.
    shard_name = target
    if args.fold is not None:
        shard_name += f"__f{args.fold}"
    if args.shard_tag:
        shard_name += f"__{args.shard_tag}"
    shard_dir = os.path.join(args.run_dir, "_shards", shard_name)
    marker = os.path.join(shard_dir, ".done")
    if os.path.exists(marker):
        print(f"[skip] {target}: shard already complete", flush=True)
        return

    # training.run_training writes into these subdirs but does not create them.
    for sub in ("metrics", "predictions", "architecture", "models"):
        os.makedirs(os.path.join(shard_dir, sub), exist_ok=True)

    from data import build_folds, load_dataset
    from training import run_training, verify_versions

    verify_versions()
    print(f"device={config.DEVICE}  cores={config.NUM_CORES}  "
          f"cv_scheme={args.cv_scheme}", flush=True)
    print(f"[cell] target={target} (index {args.target_index}) "
          f"fold={args.fold if args.fold is not None else 'all'} "
          f"models={models or 'all'} -> {shard_dir}", flush=True)

    df = load_dataset()
    df_strat, folds = build_folds(df, n_splits=config.N_SPLITS,
                                  scheme=args.cv_scheme)
    print(f"[cell] {len(folds)} folds, test sizes "
          f"{[len(te) for _, te in folds]}", flush=True)

    t0 = time.time()
    run_training(
        df_strat, folds, shard_dir,
        targets=[target],
        fractions=config.LEARNING_CURVE_FRACTIONS,
        keep_models=not args.no_keep_models,
        models=models,
        fold_subset=fold_subset,
    )
    open(marker, "w").close()
    print(f"[done] {target} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
