"""Pipeline entrypoint. Stages can be run independently with `--stage`.

Examples
--------
Smoke test (3 targets, 3 fractions, 2 folds, short AG budget):
    python run.py --stage all --smoke

Full benchmark:
    python run.py --stage train
    python run.py --stage plot   --run-dir Ionomic_Prediction_Run_<ts>
    python run.py --stage shap   --run-dir Ionomic_Prediction_Run_<ts>
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import config
from data import build_folds, fold_summary, load_dataset


def _maybe_smoke(args) -> None:
    """Shrink the sweep in-place on the config module for a fast smoke test.

    Smoke keeps the full every-10% learning-curve grid (0.00 .. 1.00) so the
    plot retains its resolution; speed comes from shorter AG time budgets,
    fewer targets, and fewer folds.
    """
    if not args.smoke:
        return
    config.LEARNING_CURVE_FRACTIONS = np.round(np.arange(0.00, 1.0001, 0.10), 4)
    config.TARGETS_AVAILABLE = ["N_output", "P_output", "K_output"]
    config.AG_TIME_LIMIT_FULL = 60
    config.AG_TIME_LIMIT_SUB = 30
    config.N_SPLITS = 2
    print(f"[smoke] fractions={config.LEARNING_CURVE_FRACTIONS}, "
          f"targets={config.TARGETS_AVAILABLE}, n_splits={config.N_SPLITS}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "train", "plot", "shap"],
                   default="all")
    p.add_argument("--run-dir", default=None,
                   help="Existing run dir for plot/shap stages.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny sweep for fast end-to-end verification.")
    p.add_argument("--no-keep-models", action="store_true",
                   help="Delete AutoGluon model dirs after each step (saves disk).")
    p.add_argument("--cv-scheme",
                   choices=["dobscv", "grouped", "stratified", "loeo"],
                   default="dobscv",
                   help="Fold scheme: dobscv (HEADLINE, deterministic round-robin, "
                        "leakage-free), grouped (SGK secondary/comparison), "
                        "stratified (legacy interpolation upper bound), "
                        "loeo (transfer estimate).")
    args = p.parse_args()

    _maybe_smoke(args)

    from training import verify_versions
    verify_versions()
    print(f"Device: {config.DEVICE} | CPU cores: {config.NUM_CORES}")
    df = load_dataset()
    print(f"Loaded {len(df)} rows | Experiments: "
          f"{sorted(df['Experiment_ID'].unique().tolist())}")

    run_dir = args.run_dir or config.make_run_dir()
    print(f"Run dir: {run_dir}")

    df_strat, folds = build_folds(df, n_splits=config.N_SPLITS, scheme=args.cv_scheme)
    fs = fold_summary(df_strat, folds)
    fs.to_csv(os.path.join(run_dir, "fold_summary.csv"), index=False)
    print(f"CV scheme: {args.cv_scheme} | {len(folds)} folds. Summary -> fold_summary.csv")

    if args.stage in ("all", "train"):
        from training import run_training
        run_training(
            df_strat, folds, run_dir,
            targets=config.TARGETS_AVAILABLE,
            fractions=config.LEARNING_CURVE_FRACTIONS,
            keep_models=not args.no_keep_models,
        )

    if args.stage in ("all", "plot"):
        from plotting import generate_all_plots
        paths = generate_all_plots(run_dir)
        print("Plots:")
        for k, v in paths.items():
            print(f"  {k:24s} -> {v}")

    if args.stage in ("all", "shap"):
        from shap_module import run_shap
        run_shap(run_dir, df_strat, targets=config.TARGETS_AVAILABLE,
                 fold_idx=0, n_background=80, n_explain=200,
                 cv_scheme=args.cv_scheme)

    print(f"\nDone. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
