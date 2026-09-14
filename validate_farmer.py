"""Commercial-cohort (external) validation — Fig. 8 of the article.

Usage
-----
    # After the main training run finishes:
    python validate_farmer.py --run-dir <run_dir>

Both AutoGluon 1.5 and TabPFN v3 are retrained from scratch on the controlled
dataset using only the 15 features available in the commercial cohort
(Strategy A, Section 2.5.7), then predict the 99 commercial plants.

Outputs under `<run-dir>/farmer_validation/`:
    farmer_clean.csv            — coerced, renamed commercial-cohort table
    missingness.csv             — which features are missing & how often
    parsimony_models/<target>/  — the retrained AutoGluon predictors
    farmer_predictions.csv      — long-format predictions (Target, Model, row,
                                  Actual, Predicted); source of the Fig. 8
                                  bias insets
    metrics_per_target.csv      — controlled-CV R2/RMSE/MAE vs commercial-cohort
    plots/fig2_strategy_A.pdf   — Fig. 8
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)

import config
from data import load_dataset
from farmer_data import AVAILABLE_FEATURES, FARMER_PATH, load_farmer, missingness_report
from farmer_plots import fig2_single_strategy
from farmer_strategies import run_strategy_a


# ---------------------------------------------------------------------------
def _metric_row(target: str, source: str, model: str, y_true, y_pred) -> dict:
    return {
        "Target": target, "Source": source, "Model": model,
        "N": int(len(y_true)),
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


def _save_plot(p): print(f"  -> {p}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="An existing Ionomic_Prediction_Run_* directory.")
    ap.add_argument("--farmer-path", default=FARMER_PATH)
    ap.add_argument("--strategy-a-time", type=int, default=600,
                    help="Per-target AutoGluon time budget (s).")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = os.path.join(run_dir, "farmer_validation")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ----- Load both sets -------------------------------------------------
    print("Loading commercial-cohort set ...")
    farmer = load_farmer(args.farmer_path)
    print(f"  rows: {len(farmer)}")
    miss = missingness_report(farmer)
    miss.to_csv(os.path.join(out_dir, "missingness.csv"), index=False)
    farmer.to_csv(os.path.join(out_dir, "farmer_clean.csv"), index=False)
    print("Missingness report:")
    print(miss.to_string(index=False))

    print("\nLoading controlled set ...")
    controlled = load_dataset()
    print(f"  rows: {len(controlled)}")

    targets_available = [t for t in config.TARGETS_AVAILABLE
                         if t in farmer.columns]
    print(f"\nValidating on {len(targets_available)} targets present in the "
          f"commercial cohort: {targets_available}")

    # ----- Strategy A — parsimony retraining ------------------------------
    print(f"\n[Strategy A] parsimony retraining on {len(AVAILABLE_FEATURES)} "
          "features ...")
    preds_df = run_strategy_a(
        controlled, farmer, targets_available,
        os.path.join(out_dir, "parsimony_models"),
        time_limit=args.strategy_a_time,
    )
    preds_df.to_csv(os.path.join(out_dir, "farmer_predictions.csv"), index=False)

    # ----- Per-target metrics ------------------------------------------------
    metrics_rows = []
    cv_path = os.path.join(run_dir, "metrics", "learning_curve.csv")
    if os.path.exists(cv_path):
        cv = pd.read_csv(cv_path)
        full = cv[(cv["Set"] == "Test") & (cv["Training_Pct"] == 100.0)]
        cv_per = (full.groupby(["Target", "Model"])
                     [["R2", "RMSE", "MAE"]].mean().reset_index())
        cv_per["Source"] = "Controlled_CV"
        metrics_rows.append(cv_per)
    for (tgt, strat, mdl), sub in preds_df.groupby(
            ["Target", "Strategy", "Model"]):
        m = _metric_row(tgt, f"Farmer_{strat}", mdl,
                        sub["Actual"], sub["Predicted"])
        metrics_rows.append(pd.DataFrame([m]))
    metrics_per_target = pd.concat(metrics_rows, ignore_index=True)
    metrics_per_target.to_csv(os.path.join(out_dir, "metrics_per_target.csv"),
                              index=False)

    # ----- Fig. 8 ----------------------------------------------------------
    cv_preds_path = os.path.join(run_dir, "predictions", "full_data_preds.csv")
    if os.path.exists(cv_preds_path):
        cv_preds = pd.read_csv(cv_preds_path)
        cv_preds = cv_preds[cv_preds["Model"].isin(("AutoGluon", "TabPFN"))]
    else:
        cv_preds = pd.DataFrame(columns=["Target", "Actual", "Predicted", "Model"])

    _save_plot(fig2_single_strategy(
        cv_preds, preds_df, "A",
        os.path.join(plot_dir, "fig2_strategy_A.pdf"),
        targets=config.TARGETS_SPEC,
    ))

    print(f"\n✓ Commercial-cohort validation complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
