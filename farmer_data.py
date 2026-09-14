"""Loader and schema reconciler for the farmer (Control_Table) validation set.

The farmer xlsx uses the literal string "NaaN" for missing cells and a slightly
different column casing than the controlled DB. This module produces a clean
DataFrame whose columns line up with `config.FEATURES + TARGETS_AVAILABLE`,
plus helpers describing which features are always-present vs systematically
missing.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

from config import FEATURES, TARGETS_AVAILABLE

# Commercial-cohort table. Not distributed with the code (see data/README.md):
# place it under data/ or set FARMER_PATH.
FARMER_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data",
    "Contro_Table_DB_27.05.26_rounded.xlsx",
)
FARMER_PATH = os.environ.get("FARMER_PATH", FARMER_PATH_DEFAULT)

# As of the 27.05.26 update, the only features the grower could not record
# remain the cumulative transpiration (no balance) and the destructive shoot
# dry weight. The four microclimatic integrals (PAR_SUM / VPD_SUM / Temp_SUM
# / RH_SUM) were retrieved post-hoc and are now present.
MISSING_FEATURES = ["CT", "SDW"]
AVAILABLE_FEATURES = [f for f in FEATURES if f not in MISSING_FEATURES]


def load_farmer(path: str = FARMER_PATH) -> pd.DataFrame:
    """Return a tidy farmer DataFrame.

    - Reads the first sheet of the xlsx.
    - Coerces the literal string 'NaaN' (and 'NaN', 'nan') to real NaN.
    - Renames `Experiment_id` → `Experiment_ID`, `Sample_id` → `Sample ID #`
      to match the controlled-set casing.
    - Renames `Temp_Sum`/`RH_Sum` → `Temp_SUM`/`RH_SUM` (spec casing).
    - All FEATURES + TARGETS columns are coerced to numeric.
    """
    df = pd.read_excel(path, sheet_name=0)

    rename = {
        "Experiment_id": "Experiment_ID",
        "Sample_id": "Sample ID #",
        "Temp_Sum": "Temp_SUM",
        "RH_Sum": "RH_SUM",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Replace 'NaaN' / 'NaN' / 'nan' string sentinels with real NaN.
    df = df.replace({"NaaN": np.nan, "NaN": np.nan,
                     "nan": np.nan}).infer_objects(copy=False)

    # Coerce every column we care about to numeric.
    for c in FEATURES + TARGETS_AVAILABLE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (n_features × 2) DataFrame: per-column NaN count and pct."""
    out = pd.DataFrame({
        "Column": FEATURES,
        "NaN_count": [int(df[c].isna().sum()) if c in df.columns else len(df)
                      for c in FEATURES],
    })
    out["NaN_pct"] = (out["NaN_count"] / max(len(df), 1) * 100).round(1)
    out["Status"] = np.where(out["NaN_pct"] >= 99.0, "MISSING_ALL",
                     np.where(out["NaN_pct"] > 0.0, "PARTIAL", "PRESENT"))
    return out


def split_by_availability(features: Iterable[str] = FEATURES):
    """Return (available, missing) feature names for the farmer set."""
    return (
        [f for f in features if f not in MISSING_FEATURES],
        [f for f in features if f in MISSING_FEATURES],
    )
