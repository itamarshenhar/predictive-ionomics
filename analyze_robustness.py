"""Rank-correlation robustness report for the supplementary SHAP run.

For each robustness target and model, load the mean|SHAP| feature-importance
vector produced under each setting and report Spearman rho (rank correlation)
and Pearson r between settings along each axis:

    coalitions   1024  vs 2048
    K            5 vs 10, 5 vs 20, 10 vs 20
    explained    set A vs set B
    fold         fold1 vs fold2  (and base/fold0 vs each)

Reads <RUN_DIR>/shap_runs/<setting_id>/<target>/shap_values_<model>.csv.
Writes <RUN_DIR>/shap_runs/robustness_rank_correlations.csv and prints a table.

Run after the robustness arrays finish:
    python analyze_robustness.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RUN_DIR = os.environ.get("RUN_DIR") or ""
if not RUN_DIR:
    raise SystemExit("RUN_DIR must be set (no fallback: a wrong run dir would "
                     "silently summarise a different partition's SHAP values).")
ROB_TARGETS = ["TRN_output", "Ca_output", "Fe_output", "Mo_output"]
MODELS = ["TabPFN", "AutoGluon"]

# (label, setting_a, setting_b) — settings match make_manifests.py setting_ids.
COMPARISONS = [
    ("coalitions 1024 vs 2048", "rob_coal1024", "rob_base"),
    ("K 5 vs 10",               "rob_base",     "rob_k10"),
    ("K 5 vs 20",               "rob_base",     "rob_k20"),
    ("K 10 vs 20",              "rob_k10",      "rob_k20"),
    ("explained set A vs B",    "rob_base",     "rob_setB"),
    ("fold 1 vs fold 2",        "rob_fold1",    "rob_fold2"),
    ("fold 0 vs fold 1",        "rob_base",     "rob_fold1"),
    ("fold 0 vs fold 2",        "rob_base",     "rob_fold2"),
]


def _mean_abs_shap(setting_id: str, target: str, model: str):
    path = os.path.join(RUN_DIR, "shap_runs", setting_id, target,
                        f"shap_values_{model}.csv")
    if not os.path.exists(path):
        return None
    vals = pd.read_csv(path).values
    return np.abs(vals).mean(axis=0)               # length = n_features


def main():
    out_rows = []
    for model in MODELS:
        for target in ROB_TARGETS:
            for label, sa, sb in COMPARISONS:
                va = _mean_abs_shap(sa, target, model)
                vb = _mean_abs_shap(sb, target, model)
                if va is None or vb is None:
                    out_rows.append(dict(model=model, target=target,
                                         comparison=label, spearman_rho=np.nan,
                                         pearson_r=np.nan, note="missing"))
                    continue
                rho = spearmanr(va, vb).correlation
                r = pearsonr(va, vb)[0]
                out_rows.append(dict(model=model, target=target,
                                     comparison=label,
                                     spearman_rho=round(float(rho), 4),
                                     pearson_r=round(float(r), 4), note=""))

    out_path = os.path.join(RUN_DIR, "shap_runs",
                            "robustness_rank_correlations.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "target", "comparison",
                                           "spearman_rho", "pearson_r", "note"])
        w.writeheader()
        w.writerows(out_rows)

    df = pd.DataFrame(out_rows)
    print(f"\nWrote {out_path}\n")
    # Summary: mean Spearman rho per (model, comparison) across the 4 targets.
    if not df.empty:
        piv = (df.dropna(subset=["spearman_rho"])
                 .groupby(["model", "comparison"])["spearman_rho"]
                 .mean().round(3).reset_index())
        print("Mean Spearman rho across the 4 robustness targets:")
        print(piv.to_string(index=False))
    print("\n(Per-target detail is in the CSV. rho>=~0.95 => rankings stable.)")


if __name__ == "__main__":
    main()
