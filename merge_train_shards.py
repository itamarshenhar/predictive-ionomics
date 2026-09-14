"""Merge per-target training shards into the canonical run-dir tree.

Each SLURM array task (run_train_cell.py) trained ONE target into
<RUN_DIR>/_shards/<target>/. This concatenates those shards into the single
files the plotting / farmer / SHAP stages expect:

    <RUN_DIR>/metrics/learning_curve.csv
    <RUN_DIR>/predictions/full_data_preds.csv
    <RUN_DIR>/architecture/autogluon_leaderboard.csv
    <RUN_DIR>/models/<target>/...            (moved up from each shard)

Idempotent: rebuilds the four CSVs from scratch each call, and moves any model
dirs not yet relocated. Also writes fold_summary.csv for provenance.

Usage:
    python merge_train_shards.py --run-dir "$RUN_DIR" --cv-scheme grouped
"""
from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd

CSV_PARTS = {
    "metrics": "learning_curve.csv",
    "predictions": "full_data_preds.csv",
    "architecture": "autogluon_leaderboard.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("RUN_DIR"), required=False)
    ap.add_argument("--cv-scheme",
                    choices=["dobscv", "grouped", "stratified", "loeo"],
                    default=os.environ.get("CV_SCHEME", "dobscv"))
    args = ap.parse_args()
    if not args.run_dir:
        raise SystemExit("--run-dir (or RUN_DIR env) is required.")

    run_dir = args.run_dir
    shards_root = os.path.join(run_dir, "_shards")
    if not os.path.isdir(shards_root):
        raise SystemExit(f"No shards dir at {shards_root} — did the training array run?")

    shards = sorted(d for d in os.listdir(shards_root)
                    if os.path.isdir(os.path.join(shards_root, d)))
    done = [d for d in shards
            if os.path.exists(os.path.join(shards_root, d, ".done"))]
    missing = [d for d in shards if d not in done]
    print(f"Shards found: {len(shards)}  complete: {len(done)}", flush=True)
    if missing:
        print(f"WARNING — incomplete/missing .done markers: {missing}", flush=True)

    # 1) Concatenate the four CSVs across shards.
    for subdir, fname in CSV_PARTS.items():
        frames = []
        for d in shards:
            p = os.path.join(shards_root, d, subdir, fname)
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
        out = os.path.join(run_dir, subdir, fname)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
            print(f"  {subdir}/{fname}: {sum(len(f) for f in frames)} rows "
                  f"from {len(frames)} shards -> {out}", flush=True)
        else:
            print(f"  {subdir}/{fname}: NO shard data found", flush=True)

    # 2) Relocate saved model dirs into the canonical models/ tree.
    models_root = os.path.join(run_dir, "models")
    os.makedirs(models_root, exist_ok=True)
    # Shard dir names are <target>[__f<fold>][__<tag>], so scan the shard's own
    # models/ tree rather than assuming shard name == target name. Merge at the
    # FOLD-dir level so several shards of one target combine cleanly.
    moved = 0
    for d in shards:
        sh_models = os.path.join(shards_root, d, "models")
        if not os.path.isdir(sh_models):
            continue
        for tgt in sorted(os.listdir(sh_models)):
            src_t = os.path.join(sh_models, tgt)
            if not os.path.isdir(src_t):
                continue
            dst_t = os.path.join(models_root, tgt)
            os.makedirs(dst_t, exist_ok=True)
            for fold_dir in sorted(os.listdir(src_t)):
                src_f, dst_f = os.path.join(src_t, fold_dir), os.path.join(dst_t, fold_dir)
                if os.path.isdir(src_f) and not os.path.isdir(dst_f):
                    shutil.move(src_f, dst_f); moved += 1
    print(f"  models: relocated {moved} fold model dirs", flush=True)

    # 3) fold_summary.csv for provenance (matches run.py's normal output).
    from data import build_folds, fold_summary, load_dataset
    import config
    df = load_dataset()
    df_strat, folds = build_folds(df, n_splits=config.N_SPLITS,
                                  scheme=args.cv_scheme)
    fold_summary(df_strat, folds).to_csv(
        os.path.join(run_dir, "fold_summary.csv"), index=False)
    print(f"Wrote fold_summary.csv (cv_scheme={args.cv_scheme}, "
          f"{len(folds)} folds)", flush=True)

    # 4) QC gate. training.py catches per-model exceptions and records NaN metric
    #    rows, so a target can "succeed" (exit 0, .done marker written) while
    #    having produced no usable numbers. Surface that loudly here rather than
    #    letting NaNs flow into the figures.
    lc_path = os.path.join(run_dir, "metrics", "learning_curve.csv")
    if os.path.exists(lc_path):
        lc = pd.read_csv(lc_path)
        full = lc[lc["Training_Frac"] == 1.0]
        bad = full[full["R2"].isna()]
        if len(bad):
            print(f"\n*** QC WARNING: {len(bad)} NaN R2 rows at Training_Frac==1.0 "
                  f"({len(bad)}/{len(full)}) ***", flush=True)
            cols = [c for c in ("Target", "Fold", "Model", "Set", "error")
                    if c in bad.columns]
            print(bad[cols].to_string(index=False), flush=True)
            if len(missing) == 0:
                print("(all shards reported .done — these are per-model failures, "
                      "not an interrupted run)", flush=True)
        else:
            print(f"QC: no NaN R2 at full training size "
                  f"({full['Model'].nunique()} models x {len(full)} rows).", flush=True)
    print("MERGE DONE", flush=True)


if __name__ == "__main__":
    main()
