"""Commercial-cohort prediction under structural feature missingness.

Strategy A — *parsimony* — refit AutoGluon 1.5 + TabPFN v3 on the controlled
dataset using only the 15 features the grower can also record
(farmer_data.AVAILABLE_FEATURES; CT and SDW excluded from training and
inference), then predict the commercial cohort directly. This is the strategy
reported in the article (Fig. 8, Section 2.5.7).
"""
from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

from farmer_data import AVAILABLE_FEATURES


# ---------------------------------------------------------------------------
#  Strategy A — Parsimony re-training on AVAILABLE_FEATURES only
# ---------------------------------------------------------------------------
def run_strategy_a(controlled: pd.DataFrame, farmer: pd.DataFrame,
                   targets: Iterable[str], parsimony_dir: str,
                   time_limit: int = 600) -> pd.DataFrame:
    """Re-fit AG 1.5 + TabPFN-v3 on the controlled-grown matrix using only
    AVAILABLE_FEATURES (everything farmers also have). Predict on the farmer
    set and return a long-format predictions DataFrame.

    AG predictors are persisted under `parsimony_dir/<target>/` so SHAP and
    further inference can reuse them without retraining.
    """
    from autogluon.tabular import TabularPredictor
    from sklearn.metrics import mean_absolute_error, r2_score, \
        root_mean_squared_error
    from tqdm import tqdm
    from config import (AG_PRESET, NUM_CORES, NUM_GPUS_AG)

    os.makedirs(parsimony_dir, exist_ok=True)
    X_train = controlled[AVAILABLE_FEATURES]
    X_farmer = farmer[AVAILABLE_FEATURES]

    records: list[dict] = []
    for tgt in tqdm(list(targets), desc="Strategy A — parsimony fit"):
        y_train = controlled[tgt]
        y_farmer = farmer[tgt]

        # ---- AutoGluon ----
        ag_path = os.path.join(parsimony_dir, tgt)
        ag = TabularPredictor(label=tgt, path=ag_path,
                              problem_type="regression", verbosity=0)
        ag.fit(
            train_data=pd.concat([X_train, y_train], axis=1),
            presets=AG_PRESET, time_limit=time_limit,
            num_cpus=NUM_CORES, num_gpus=NUM_GPUS_AG,
        )
        ag_pred = ag.predict(X_farmer).values
        for i in range(len(farmer)):
            records.append({
                "Target": tgt, "Strategy": "A", "Model": "AutoGluon",
                "row": i, "Actual": float(y_farmer.iloc[i]),
                "Predicted": float(ag_pred[i]),
            })

        # ---- TabPFN-v3 ----
        pfn = _fit_tabpfn_v3_inner(X_train, y_train)
        pfn_pred = pfn.predict(X_farmer.values)
        for i in range(len(farmer)):
            records.append({
                "Target": tgt, "Strategy": "A", "Model": "TabPFN",
                "row": i, "Actual": float(y_farmer.iloc[i]),
                "Predicted": float(pfn_pred[i]),
            })

    return pd.DataFrame(records)


def _fit_tabpfn_v3_inner(X: pd.DataFrame, y: pd.Series):
    """Fresh TabPFN-v3 fit (32-member ensemble), same kwargs as training.py."""
    from tabpfn import TabPFNRegressor
    from config import (DEVICE, NUM_CORES, RANDOM_STATE,
                        TABPFN_IGNORE_LIMITS, TABPFN_MODEL_VERSION,
                        TABPFN_N_ESTIMATORS)
    kwargs = dict(device=DEVICE, n_estimators=TABPFN_N_ESTIMATORS,
                  n_jobs=NUM_CORES,
                  ignore_pretraining_limits=TABPFN_IGNORE_LIMITS,
                  random_state=RANDOM_STATE)
    try:
        m = TabPFNRegressor.create_default_for_version(
            TABPFN_MODEL_VERSION, **kwargs)
    except TypeError:
        for bad in ("n_jobs", "ignore_pretraining_limits"):
            kwargs.pop(bad, None)
        m = TabPFNRegressor.create_default_for_version(
            TABPFN_MODEL_VERSION, **kwargs)
    m.fit(X.values, y.values)
    return m
