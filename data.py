"""Data loading, derivation, and double-stratified fold construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold,
)

from config import (
    ACTIVE_TREATMENT_BY_EXP, DATA_PATH, DOBSCV_N_TIERS, FEATURES, N_SPLITS,
    RANDOM_STATE, TARGETS_AVAILABLE,
)


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the master dataset (xlsx or csv) and reconcile column names.

    - Reads the first sheet of an xlsx, or the CSV directly.
    - Renames Temp_Sum/RH_Sum to Temp_SUM/RH_SUM (spec casing).
    - Keeps Mo_input as-is (present from the 24.05.26 dataset onward).
    """
    if path.lower().endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)

    df = df.rename(columns={"Temp_Sum": "Temp_SUM", "RH_Sum": "RH_SUM"})

    # Defensive: if older datasets are reused, inject Mo_input zeros so the
    # feature set still aligns with the spec.
    if "Mo_input" not in df.columns:
        df["Mo_input"] = 0.0

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    missing_targets = [c for c in TARGETS_AVAILABLE if c not in df.columns]
    if missing_targets:
        # Don't fail; the training loop skips targets that aren't present.
        print(f"[data] WARNING — targets missing from CSV will be skipped: "
              f"{missing_targets}")

    return df.reset_index(drop=True)


def _active_gradient(row: pd.Series) -> float:
    """Return the active fertilizer level for a row, per its Experiment_ID."""
    col = ACTIVE_TREATMENT_BY_EXP.get(int(row["Experiment_ID"]))
    return float(row[col]) if col and col in row.index else 0.0


def attach_stratification_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add `Active_Gradient` and `Strat_Key` columns in-place-friendly fashion."""
    out = df.copy()
    out["Active_Gradient"] = out.apply(_active_gradient, axis=1)
    out["Strat_Key"] = (
        out["Experiment_ID"].astype(str) + "::" + out["Active_Gradient"].astype(str)
    )

    # Lump rare strata (fewer than N_SPLITS members) so StratifiedKFold can run.
    counts = out["Strat_Key"].value_counts()
    rare = set(counts.index[counts < N_SPLITS])
    if rare:
        out.loc[out["Strat_Key"].isin(rare), "Strat_Key"] = "__rare__"
    return out


def _condition_key(df_strat: pd.DataFrame) -> pd.Series:
    """True dosing-condition key (experiment x active-gradient level), WITHOUT
    the rare-strata lumping that `Strat_Key` applies for StratifiedKFold. This is
    the indivisible group for DOB-SCV."""
    return (df_strat["Experiment_ID"].astype(str) + "::"
            + df_strat["Active_Gradient"].astype(str))


def _per_experiment_dose_tier(df_strat: pd.DataFrame,
                              n_tiers: int = DOBSCV_N_TIERS) -> pd.Series:
    """Grade each row into a dose tier (0..n_tiers-1) using the rank of its
    active-gradient level WITHIN its own experiment (low -> high). A condition's
    tier is well-defined because every row of a condition shares one gradient.

    Symmetric binning (rank fraction over n_levels-1): the minimum dose maps to
    the lowest tier and the maximum dose to the highest tier, so when n_levels is
    not divisible by n_tiers the shortfall falls on the MIDDLE tier and the Low/
    High extremes stay balanced (e.g. 8 levels -> 3/2/3, not 3/3/2)."""
    tier = pd.Series(0, index=df_strat.index, dtype=int)
    for _, sub in df_strat.groupby("Experiment_ID"):
        r = sub["Active_Gradient"].rank(method="dense")   # 1..n_levels
        n_levels = int(r.max())
        if n_levels <= 1:
            tier.loc[sub.index] = 0
        else:
            b = np.floor((r - 1) / (n_levels - 1) * n_tiers).astype(int)
            tier.loc[sub.index] = np.minimum(b, n_tiers - 1)
    return tier


def build_folds(df: pd.DataFrame, n_splits: int = N_SPLITS, scheme: str = "dobscv"):
    """Return (df_strat, folds).

    scheme:
      "dobscv"     -> Distribution-Optimally-Balanced Stratified CV [HEADLINE].
                      Whole conditions grouped and distributed round-robin across
                      folds within each per-experiment dose tier: single
                      deterministic partition, every experiment in every fold,
                      dose-balanced, leakage-free.
      "grouped"    -> StratifiedGroupKFold, groups=Strat_Key, y=dose_bin [secondary/SGK, leakage-free]
      "stratified" -> StratifiedKFold on Strat_Key                       [legacy interpolation upper bound]
      "loeo"       -> LeaveOneGroupOut, groups=Experiment_ID             [transfer estimate; folds = n experiments]

    Indices reference the rows of the returned DataFrame (positional).
    """
    df_strat = attach_stratification_keys(df)
    keys = df_strat["Strat_Key"].values

    if scheme == "dobscv":
        cond = _condition_key(df_strat)
        tier = _per_experiment_dose_tier(df_strat, n_tiers=DOBSCV_N_TIERS)
        # one representative row per condition (tier + experiment + gradient)
        info = pd.DataFrame({
            "cond": cond.values, "tier": tier.values,
            "exp": df_strat["Experiment_ID"].values,
            "grad": df_strat["Active_Gradient"].values,
        }).drop_duplicates("cond")
        # Doubly-balanced deterministic assignment. Within each (experiment x dose
        # tier) cell the conditions are dealt round-robin across folds, and the
        # starting fold is rotated by BOTH the experiment and the tier:
        #   * rotating per experiment spreads every experiment evenly across all
        #     folds, so each fold is a representative mini-replica of the full
        #     design (every experiment appears in every fold's test set), and
        #   * rotating per tier (2*tier) keeps the Low/Mid/High tiers balanced
        #     across folds instead of co-locating.
        # This balances the joint experiment x dose distribution, which is what
        # distinguishes DOB-SCV from a plain grouped/stratified split.
        exp_index = {e: i for i, e in enumerate(sorted(info["exp"].unique()))}
        fold_of_cond = {}
        for (e, t), sub in info.groupby(["exp", "tier"]):
            offset = (exp_index[e] + 2 * int(t)) % n_splits
            for pos, key in enumerate(sub.sort_values("grad")["cond"].tolist()):
                fold_of_cond[key] = (offset + pos) % n_splits
        fid = cond.map(fold_of_cond).to_numpy()
        folds = [(np.where(fid != k)[0], np.where(fid == k)[0])
                 for k in range(n_splits)]

    elif scheme == "stratified":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        folds = list(skf.split(df_strat, keys))

    elif scheme == "grouped":
        # dose tier for balancing (quintiles of the active gradient, rank-based)
        dose_bin = pd.qcut(df_strat["Active_Gradient"].rank(method="first"),
                           5, labels=False, duplicates="drop")
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                   random_state=RANDOM_STATE)
        folds = list(sgk.split(df_strat, dose_bin, groups=keys))

    elif scheme == "loeo":
        logo = LeaveOneGroupOut()
        folds = list(logo.split(df_strat, df_strat[TARGETS_AVAILABLE[0]],
                                groups=df_strat["Experiment_ID"].values))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return df_strat, folds


def fold_summary(df_strat: pd.DataFrame, folds) -> pd.DataFrame:
    """Diagnostic: count samples per Experiment_ID in each fold's test set."""
    rows = []
    for i, (_, te) in enumerate(folds):
        sub = df_strat.iloc[te]
        for exp, n in sub["Experiment_ID"].value_counts().items():
            rows.append({"Fold": i, "Experiment_ID": int(exp), "Test_N": int(n)})
    return pd.DataFrame(rows).sort_values(["Fold", "Experiment_ID"]).reset_index(drop=True)
