"""Verify a CV scheme's fold assignment before any heavy run.

Reproduces the claims in the DOB-SCV figure caption for the chosen scheme:
  - same-condition leakage  (whole conditions never split -> 0%)
  - experiment coverage     (every experiment present in every fold's TRAIN set)
  - dose balance            (chi-square of dose-tier x fold; large p = balanced)
  - fold test sizes / #conditions per fold

    python verify_cv.py                 # dobscv (headline)
    python verify_cv.py grouped         # SGK secondary
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import config
from data import (_condition_key, _per_experiment_dose_tier, build_folds,
                  load_dataset)


def main() -> None:
    scheme = sys.argv[1] if len(sys.argv) > 1 else "dobscv"
    df = load_dataset()
    df_strat, folds = build_folds(df, n_splits=config.N_SPLITS, scheme=scheme)
    cond = _condition_key(df_strat).to_numpy()
    tier = _per_experiment_dose_tier(df_strat).to_numpy()
    exp = df_strat["Experiment_ID"].to_numpy()
    n = len(df_strat)

    print(f"scheme = {scheme} | {len(folds)} folds | {n} plants | "
          f"{len(set(cond))} conditions | {len(set(exp))} experiments\n")

    # --- 1. same-condition leakage (whole condition never split across tr/te) ---
    leak = tot = 0
    for tr, te in folds:
        tr_conds = set(cond[tr])
        for c in cond[te]:
            tot += 1
            if c in tr_conds:
                leak += 1
    print(f"same-condition leakage : {leak}/{tot} = {100*leak/tot:.2f}%  "
          f"({'PASS' if leak == 0 else 'FAIL'})")

    # --- 2. experiment coverage: every experiment in every fold's TRAIN set ---
    all_exp = set(exp.tolist())
    missing = 0
    for i, (tr, _) in enumerate(folds):
        miss = all_exp - set(exp[tr].tolist())
        if miss:
            missing += 1
            print(f"    fold {i}: missing experiments in train -> {sorted(miss)}")
    print(f"experiment coverage    : every experiment in every fold train "
          f"({'PASS' if missing == 0 else f'FAIL ({missing} folds)'})")

    # --- 3. dose balance: chi-square of (dose tier x fold) over TEST conditions --
    rows = []
    for i, (_, te) in enumerate(folds):
        seen = {}
        for c, t in zip(cond[te], tier[te]):
            seen[c] = t
        for c, t in seen.items():
            rows.append((i, t))
    ct = pd.crosstab(pd.Series([r[0] for r in rows], name="fold"),
                     pd.Series([r[1] for r in rows], name="tier"))
    chi2, p, _, _ = chi2_contingency(ct)
    print(f"dose balance (tier x fold): chi2={chi2:.2f}, p={p:.3f}  "
          f"({'balanced' if p > 0.05 else 'imbalanced'})")
    print("  conditions per fold x tier:")
    print(ct.to_string().replace("\n", "\n  "))

    # --- 4. fold sizes ---
    sizes = [len(te) for _, te in folds]
    nconds = [len(set(cond[te])) for _, te in folds]
    print(f"\ntest plants per fold    : {sizes}")
    print(f"test conditions per fold: {nconds}  (mean {np.mean(nconds):.1f})")


if __name__ == "__main__":
    main()
