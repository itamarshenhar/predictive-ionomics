"""Dual-model training: AutoGluon 1.5 (best_quality) vs TabPFN v3.

Writes raw artifacts to disk after every (target, fold, step), so the loop is
fully resumable and plotting / SHAP are decoupled from training.

Artifacts under <run_dir>/:
  metrics/learning_curve.csv          - per (target, fold, step, model)
  predictions/full_data_preds.csv     - per row in test fold at step == 1.0
  architecture/autogluon_leaderboard.csv
  models/<target>/Fold_<i>_Step_<s>/  - AutoGluon model dir at step == 1.0
"""
from __future__ import annotations

import gc
import json
import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tqdm import tqdm

from config import (
    AG_PRESET, AG_TIME_LIMIT_FULL, AG_TIME_LIMIT_SUB, DEVICE,
    FEATURES, LEARNING_CURVE_FRACTIONS, MIN_SAMPLES, NUM_CORES, NUM_GPUS_AG,
    RANDOM_STATE,
    TABPFN_IGNORE_LIMITS, TABPFN_MAX_SAMPLES, TABPFN_MODEL_VERSION,
    TABPFN_N_ESTIMATORS, TABPFN_USE_CLOUD, TARGETS_AVAILABLE,
)

warnings.filterwarnings("ignore")


def verify_versions() -> dict:
    """Sanity-check AutoGluon 1.5 + TabPFN v3 at startup.

    Confirms the exact checkpoint that `create_default_for_version("v3")`
    resolves to and refuses to proceed if it isn't a v3 file.
    """
    import importlib.metadata as md
    from tabpfn import TabPFNRegressor

    ag = md.version("autogluon.tabular")
    pfn = md.version("tabpfn")
    if not ag.startswith("1.5"):
        warnings.warn(f"AutoGluon {ag} found; spec calls for 1.5.x")

    probe = TabPFNRegressor.create_default_for_version(TABPFN_MODEL_VERSION)
    ckpt = str(probe.model_path)
    if "v3" not in ckpt.lower():
        raise RuntimeError(
            f"TabPFN model checkpoint is not v3: {ckpt}. Confirm the v3 "
            "checkpoint is downloaded (run `python -c \"from tabpfn import "
            "TabPFNRegressor; TabPFNRegressor.create_default_for_version('v3')\"`)."
        )

    info = {"autogluon": ag, "tabpfn_pkg": pfn,
            "tabpfn_model_version": TABPFN_MODEL_VERSION,
            "tabpfn_checkpoint": ckpt}
    print(f"[versions] AutoGluon={ag}  TabPFN_pkg={pfn}  "
          f"TabPFN_model={TABPFN_MODEL_VERSION} "
          f"({os.path.basename(ckpt)})  device={DEVICE}")
    return info


# --- metric helper -----------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


# --- AutoGluon helpers -------------------------------------------------------
def _fit_autogluon(train_df: pd.DataFrame, target: str, model_path: str,
                   time_limit: int):
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor(
        label=target, path=model_path, problem_type="regression", verbosity=0,
    )
    predictor.fit(
        train_data=train_df,
        presets=AG_PRESET,
        time_limit=time_limit,
        num_cpus=NUM_CORES,
        num_gpus=NUM_GPUS_AG,
    )
    return predictor


def _ag_leaderboard(predictor, target: str, fold_idx: int) -> pd.DataFrame:
    """Leaderboard + ensemble weights + per-base-model hyperparameters.

    Adds columns:
      - ensemble_weight : weight of this model in the L2 weighted ensemble
                         (0 for non-base models / models not selected)
      - stack_level     : already provided by AG
      - hyperparameters : compact JSON string of the model's hyperparameters
      - model_type      : AG model class name (e.g. 'LightGBM', 'CatBoost')
    """
    import json as _json
    lb = predictor.leaderboard(silent=True).copy()
    lb["Target"] = target
    lb["Fold"] = fold_idx

    info = predictor.info()
    model_info = info.get("model_info", {})

    # Ensemble weights live in each WeightedEnsemble_*'s children: averaged
    # across bagging children when there are multiple.
    weights: dict[str, float] = {}
    for m_name, m_info in model_info.items():
        if m_info.get("model_type", "") != "WeightedEnsembleModel":
            continue
        accum: dict[str, list[float]] = {}
        for child in m_info.get("children_info", {}).values():
            for base, w in child.get("model_weights", {}).items():
                accum.setdefault(base, []).append(float(w))
        for base, ws in accum.items():
            # Largest weight wins if the same base appears in multiple ensembles
            weights[base] = max(weights.get(base, 0.0), float(np.mean(ws)))
    lb["ensemble_weight"] = lb["model"].map(weights).fillna(0.0)

    # Hyperparameters (compact JSON) + model class per row.
    hparams, mtypes = [], []
    for m_name in lb["model"]:
        mi = model_info.get(m_name, {})
        hparams.append(_json.dumps(mi.get("hyperparameters", {}), default=str))
        mtypes.append(mi.get("model_type", ""))
    lb["hyperparameters"] = hparams
    lb["model_type"] = mtypes
    return lb


# --- TabPFN helpers ----------------------------------------------------------
def _build_tabpfn():
    """Construct a TabPFNRegressor wired to max-accuracy settings.

    Routes to `tabpfn_client` (hosted) when both `TABPFN_USE_CLOUD` is set and
    the client package + `TABPFN_API_TOKEN` env var are available; otherwise
    uses local `tabpfn` 8.x with n_estimators=TABPFN_N_ESTIMATORS.
    """
    if TABPFN_USE_CLOUD:
        try:
            from tabpfn_client import TabPFNRegressor as CloudRegressor  # type: ignore
            token = os.environ.get("TABPFN_API_TOKEN")
            if not token:
                raise RuntimeError("TABPFN_USE_CLOUD set but TABPFN_API_TOKEN unset")
            try:
                from tabpfn_client import set_access_token  # type: ignore
                set_access_token(token)
            except Exception:
                pass  # newer clients pick up the env var directly
            return CloudRegressor()
        except ImportError:
            print("[TabPFN] tabpfn_client not installed; falling back to local model")

    from tabpfn import TabPFNRegressor
    # Pin to the explicit v3 checkpoint via the version-aware factory rather
    # than relying on `model_path="auto"` (which can change as new defaults
    # ship). Overrides are passed through as kwargs.
    kwargs = dict(
        device=DEVICE,
        n_estimators=TABPFN_N_ESTIMATORS,
        n_jobs=NUM_CORES,
        ignore_pretraining_limits=TABPFN_IGNORE_LIMITS,
        random_state=RANDOM_STATE,
    )
    try:
        return TabPFNRegressor.create_default_for_version(
            TABPFN_MODEL_VERSION, **kwargs,
        )
    except TypeError:
        # Older tabpfn signatures lack some kwargs — strip them and retry.
        for k in ("n_jobs", "ignore_pretraining_limits"):
            kwargs.pop(k, None)
        return TabPFNRegressor.create_default_for_version(
            TABPFN_MODEL_VERSION, **kwargs,
        )


def _fit_tabpfn(X_train: pd.DataFrame, y_train: pd.Series):
    if len(X_train) > TABPFN_MAX_SAMPLES:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), TABPFN_MAX_SAMPLES, replace=False
        )
        X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]
    model = _build_tabpfn()
    model.fit(X_train.values, y_train.values)
    return model


# --- main loop ---------------------------------------------------------------
def run_training(df_strat: pd.DataFrame, folds, run_dir: str,
                 targets: Iterable[str] = TARGETS_AVAILABLE,
                 fractions: Iterable[float] = LEARNING_CURVE_FRACTIONS,
                 keep_models: bool = True,
                 models: Iterable[str] | None = None,
                 fold_subset: Iterable[int] | None = None) -> dict:
    """Execute the dual-model sweep.

    Streams to CSV after every (target, fold, step) so a kill/restart loses at
    most one step. On resume, completed (target, fold, step) triples are skipped.

    `models` restricts which learners run — used to split the GPU-bound learner
    (TabPFN) from the CPU-bound one (AutoGluon) into separate SLURM
    arrays. `fold_subset` restricts which outer folds run, so a shard can be one
    (target, fold) cell instead of a whole target.
    """
    metrics_path = os.path.join(run_dir, "metrics", "learning_curve.csv")
    preds_path   = os.path.join(run_dir, "predictions", "full_data_preds.csv")
    arch_path    = os.path.join(run_dir, "architecture", "autogluon_leaderboard.csv")

    # Models written to learning_curve.csv.
    active_models = ["AutoGluon", "TabPFN"]
    if models is not None:
        wanted = list(models)
        unknown = [m for m in wanted if m not in active_models]
        if unknown:
            raise ValueError(f"unknown model(s) {unknown}; known: {active_models}")
        active_models = wanted
    run_ag  = "AutoGluon" in active_models
    run_pfn = "TabPFN" in active_models
    print(f"[training] models: {active_models}")

    done_keys = set()
    if os.path.exists(metrics_path):
        prev = pd.read_csv(metrics_path)
        # A (target, fold, frac, model) entry is complete iff both Set=Train and
        # Set=Test rows are present. Otherwise we redo it.
        grp = prev.groupby(
            ["Target", "Fold", "Training_Frac", "Model"]
        )["Set"].nunique()
        for (tgt, fold, frac, model), n_sets in grp.items():
            if n_sets >= 2:
                done_keys.add((tgt, int(fold), round(float(frac), 4), model))

    n_total_rows = len(df_strat)
    n_folds_run = (len(folds) if fold_subset is None
                   else len([i for i in range(len(folds)) if i in fold_subset]))
    pbar = tqdm(total=len(list(targets)) * n_folds_run * len(list(fractions)),
                desc="Sweep", dynamic_ncols=True)

    for target in targets:
        target_model_dir = os.path.join(run_dir, "models", target)
        os.makedirs(target_model_dir, exist_ok=True)

        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            if fold_subset is not None and fold_idx not in fold_subset:
                continue
            train_full = df_strat.iloc[tr_idx]
            test_df    = df_strat.iloc[te_idx]
            X_test, y_test = test_df[FEATURES], test_df[target]
            test_payload = pd.concat([X_test, y_test], axis=1)

            # Pre-compute the fold's full-training-pool mean for the x=0%
            # baseline ("zero-knowledge" dummy regressor).
            y_train_pool = train_full[target].values
            y_train_pool_mean = float(np.mean(y_train_pool))

            for frac in fractions:
                frac = float(frac)
                key_ag  = (target, fold_idx, round(frac, 4), "AutoGluon")
                key_pfn = (target, fold_idx, round(frac, 4), "TabPFN")
                keys_by_model = {m: (target, fold_idx, round(frac, 4), m)
                                 for m in active_models}
                if all(k in done_keys for k in keys_by_model.values()):
                    pbar.update(1); continue

                n = int(len(train_full) * frac)

                # x = 0 % : dummy mean predictor, identical for both models.
                if frac == 0.0:
                    base_row0 = {
                        "Target": target, "Fold": fold_idx,
                        "Training_Frac": 0.0, "Training_Pct": 0.0,
                        "Sample_N": 0, "error": "",
                    }
                    train_pred = np.full_like(y_train_pool, y_train_pool_mean,
                                              dtype=float)
                    test_pred = np.full(len(y_test), y_train_pool_mean,
                                        dtype=float)
                    m_train = _metrics(y_train_pool, train_pred)
                    m_test  = _metrics(y_test.values, test_pred)
                    for model in active_models:
                        for setname, m in (("Train", m_train), ("Test", m_test)):
                            _append_row(metrics_path, {
                                **base_row0, "Model": model, "Set": setname,
                                "baseline": "mean_predictor", **m,
                            })
                    pbar.update(1); continue

                if n < MIN_SAMPLES:
                    pbar.update(1); continue

                train_sample = train_full.sample(n=n, random_state=RANDOM_STATE)
                X_tr, y_tr = train_sample[FEATURES], train_sample[target]

                full_step = (frac >= 0.999)
                time_limit = AG_TIME_LIMIT_FULL if full_step else AG_TIME_LIMIT_SUB

                base_row = {
                    "Target": target, "Fold": fold_idx,
                    "Training_Frac": frac, "Training_Pct": frac * 100,
                    "Sample_N": n,
                }

                # ----- AutoGluon ------------------------------------------------
                if run_ag and key_ag not in done_keys:
                    ag_path = os.path.join(
                        target_model_dir, f"Fold_{fold_idx}_Step_{frac:.2f}"
                    )
                    try:
                        ag = _fit_autogluon(
                            pd.concat([X_tr, y_tr], axis=1), target, ag_path, time_limit,
                        )
                        ag_train_pred = ag.predict(X_tr).values
                        ag_test_pred  = ag.predict(X_test).values
                        _append_row(metrics_path, {
                            **base_row, "Model": "AutoGluon", "Set": "Train",
                            **_metrics(y_tr.values, ag_train_pred),
                        })
                        _append_row(metrics_path, {
                            **base_row, "Model": "AutoGluon", "Set": "Test",
                            **_metrics(y_test.values, ag_test_pred),
                        })

                        if full_step:
                            _append_df(arch_path, _ag_leaderboard(ag, target, fold_idx))
                            preds_df = pd.DataFrame({
                                "Target": target, "Fold": fold_idx, "Model": "AutoGluon",
                                "Actual": y_test.values, "Predicted": ag_test_pred,
                            })
                            _append_df(preds_path, preds_df)

                        # Keep step==1.0 model dirs (needed for SHAP), purge
                        # intermediate sweep steps when keep_models is False.
                        if not keep_models and not full_step:
                            _purge_dir(ag_path)
                        del ag
                    except Exception as e:
                        for s in ("Train", "Test"):
                            _append_row(metrics_path, {
                                **base_row, "Model": "AutoGluon", "Set": s,
                                "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                                "error": repr(e)[:200],
                            })

                # ----- TabPFN v3 ----------------------------------------------
                if run_pfn and key_pfn not in done_keys:
                    try:
                        pfn = _fit_tabpfn(X_tr, y_tr)
                        pfn_train_pred = pfn.predict(X_tr.values)
                        pfn_test_pred  = pfn.predict(X_test.values)
                        _append_row(metrics_path, {
                            **base_row, "Model": "TabPFN", "Set": "Train",
                            **_metrics(y_tr.values, pfn_train_pred),
                        })
                        _append_row(metrics_path, {
                            **base_row, "Model": "TabPFN", "Set": "Test",
                            **_metrics(y_test.values, pfn_test_pred),
                        })
                        if full_step:
                            preds_df = pd.DataFrame({
                                "Target": target, "Fold": fold_idx, "Model": "TabPFN",
                                "Actual": y_test.values, "Predicted": pfn_test_pred,
                            })
                            _append_df(preds_path, preds_df)
                        del pfn
                    except Exception as e:
                        for s in ("Train", "Test"):
                            _append_row(metrics_path, {
                                **base_row, "Model": "TabPFN", "Set": s,
                                "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                                "error": repr(e)[:200],
                            })

                # release GPU / CPU memory between iterations
                gc.collect()
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                elif DEVICE == "cuda":
                    torch.cuda.empty_cache()
                pbar.update(1)

    pbar.close()
    summary = {
        "metrics": metrics_path, "predictions": preds_path,
        "architecture": arch_path,
        "n_rows": n_total_rows,
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --- IO helpers --------------------------------------------------------------
def _append_row(path: str, row: dict) -> None:
    df = pd.DataFrame([row])
    _append_df(path, df)


def _append_df(path: str, df: pd.DataFrame) -> None:
    """Append a DataFrame to a CSV, keeping column alignment with the header.

    If the file exists, we re-order `df` to match the existing header and add
    NaN columns for any header columns it doesn't carry. New columns in `df`
    that aren't already in the file are dropped silently with a one-time
    warning — adding columns mid-run would corrupt every prior row.
    """
    if not os.path.exists(path):
        df.to_csv(path, mode="w", header=True, index=False)
        return
    with open(path, "r") as f:
        existing_cols = f.readline().strip().split(",")
    aligned = df.reindex(columns=existing_cols)
    extra = set(df.columns) - set(existing_cols)
    if extra:
        warnings.warn(f"_append_df: dropping new cols {extra} not in {path}")
    aligned.to_csv(path, mode="a", header=False, index=False)


def _purge_dir(path: str) -> None:
    import shutil
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
