"""Re-draw Fig. 8 from the CSVs the farmer-validation stage already wrote.

No cluster, no retraining, no GPU — pure plotting. Use this when only the
figure needs to change and the underlying predictions are unchanged.

Reads  <run_dir>/farmer_validation/farmer_predictions.csv   (Strategy A rows)
       <run_dir>/predictions/full_data_preds.csv            (controlled-CV cloud)
Writes <run_dir>/plots/fig2_strategy_A.{pdf,png}

    python regen_farmer_fig2.py --run-dir /path/to/run
    RUN_DIR=/path/to/run python regen_farmer_fig2.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import config
from farmer_plots import fig2_single_strategy

HEADLINE_MODELS = ("AutoGluon", "TabPFN")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("RUN_DIR"),
                    help="Run directory containing farmer_validation/ and "
                         "predictions/ (or set RUN_DIR).")
    args = ap.parse_args()
    run_dir = args.run_dir
    if not run_dir:
        raise SystemExit("pass --run-dir or set RUN_DIR")

    fv = os.path.join(run_dir, "farmer_validation", "farmer_predictions.csv")
    if not os.path.exists(fv):
        raise SystemExit(f"Not found: {fv}")
    preds = pd.read_csv(fv)
    if "A" not in set(preds.get("Strategy", [])):
        raise SystemExit(f"{fv} has no Strategy A rows")

    # Controlled-CV background clouds (optional: figure still renders without).
    cv_path = os.path.join(run_dir, "predictions", "full_data_preds.csv")
    if os.path.exists(cv_path):
        cv = pd.read_csv(cv_path)
        cv = cv[cv["Model"].isin(HEADLINE_MODELS)]
        print(f"CV cloud: {len(cv)} rows from {cv['Model'].nunique()} models")
    else:
        cv = pd.DataFrame(columns=["Target", "Actual", "Predicted", "Model"])
        print(f"[warn] {cv_path} missing — panels will have no grey CV cloud")

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out = fig2_single_strategy(cv, preds, "A",
                               os.path.join(plot_dir, "fig2_strategy_A.pdf"),
                               targets=config.TARGETS_SPEC)
    print(f"  -> {out}")
    print("DONE")


if __name__ == "__main__":
    main()
