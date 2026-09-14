"""Central configuration for the ionomic prediction pipeline.

All hardware, feature, target, fold, and learning-curve constants live here so
the training, plotting, and SHAP modules stay in sync.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import torch

# --- Hardware (Apple M4 Max origin; HURCS A100/A30 on cluster) ---------------
# Prefer CUDA (HURCS Dogfish=A100 / Puffin=A30), then Apple MPS, then CPU, so
# the same config.py works unchanged on the Mac and on the cluster.
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
# Honour the SLURM allocation on the cluster; fall back to local core count.
NUM_CORES = int(os.environ.get(
    "SLURM_CPUS_PER_TASK", max(1, (os.cpu_count() or 14) - 2)))
NUM_GPUS_AG = 0                                  # AutoGluon tabular is CPU-native

# --- Dataset paths -----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Path to the master xlsx (experiments 1..16). The dataset is not distributed
# with the code (see data/README.md): place it under data/ or set DATA_PATH.
DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(PROJECT_ROOT, "data", "Min_Bal_DB_24.05.26.xlsx"),
)
# Legacy alias retained for any external callers that still import DATA_CSV.
DATA_CSV = DATA_PATH

def make_run_dir(prefix: str = "Ionomic_Prediction_Run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(PROJECT_ROOT, f"{prefix}_{stamp}")
    os.makedirs(out, exist_ok=True)
    for sub in ("metrics", "predictions", "architecture", "importance",
                "models", "shap", "plots"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out

# --- Features & targets ------------------------------------------------------
# Mo_input is now a real column in the 24.05.26 dataset (no longer injected).
FEATURES = [
    "N_input", "P_input", "K_input", "Ca_input", "Mg_input", "Fe_input", "Mo_input",
    "Plant_type", "Substrate", "CT", "J_day", "SDW", "DAT",
    "PAR_SUM", "VPD_SUM", "Temp_SUM", "RH_SUM",
]

# 12 user-requested targets — all present in the 24.05.26 dataset.
# TRN_output ("total reduced N") replaces the user's earlier NH4_output, which
# is not measured separately in this study.
TARGETS_SPEC = [
    "NO3_output", "TRN_output", "P_output",  "K_output",
    "Ca_output",  "Mg_output",  "Fe_output", "Mn_output",
    "Zn_output",  "Bo_output",  "Mo_output", "Na_output",
]
TARGETS_AVAILABLE = list(TARGETS_SPEC)

# --- Stratification: active fertilizer treatment per experiment --------------
ACTIVE_TREATMENT_BY_EXP = {
    1: "N_input",  2: "N_input",  13: "N_input",
    3: "K_input",  4: "K_input",  14: "K_input",
    5: "Mg_input", 6: "Mg_input",
    7: "Fe_input", 8: "Fe_input",
    9: "P_input",  10: "P_input", 15: "P_input",
    11: "Ca_input", 12: "Ca_input",
    16: "Mo_input",
}

# --- CV & learning curve -----------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42

# Headline CV = DOB-SCV (Distribution-Optimally-Balanced Stratified CV): whole
# dosing conditions (Strat_Key) are grouped and distributed round-robin across
# folds within each dose tier -> single deterministic partition, every
# experiment in every fold, dose-balanced, leakage-free. The dose tier is graded
# PER EXPERIMENT (low/mid/high of each experiment's own active-gradient range).
DOBSCV_N_TIERS = 3

# 11 training-set fractions every 10 % (0.00 .. 1.00). At fraction 0 the
# training step short-circuits to a zero-knowledge mean-baseline (predict
# mean(y_train_pool) for both Train and Test sets) so all 4 lines have a
# computed data point at x = 0 %.
LEARNING_CURVE_FRACTIONS = np.round(np.arange(0.00, 1.0001, 0.10), 4)
MIN_SAMPLES = 30                # below this (and > 0) we skip the step
TABPFN_MAX_SAMPLES = 10_000     # TabPFN context limit

# AutoGluon: best_quality (max-accuracy preset). Time budgets cap a single fit.
# Raised 4x for the final production run (2026-07-26): at 900/240 s the
# best_quality search was being cut off well before convergence, which capped
# the headline model's quality. The split CPU/GPU arrays run all 60
# (target, fold) cells concurrently, so 4x the search costs ~2.5 h of extra
# wall-clock (~3.5 h total), not 4x the total time.
# Env-overridable to restore the fast profile: AG_TIME_LIMIT_FULL=900 ...
AG_TIME_LIMIT_FULL = int(os.environ.get("AG_TIME_LIMIT_FULL", "3600"))  # step == 1.0
AG_TIME_LIMIT_SUB = int(os.environ.get("AG_TIME_LIMIT_SUB", "900"))     # intermediate
AG_PRESET = "best_quality"

# TabPFN: highest-accuracy local config (per Prior Labs guidance n=32) using
# the explicit **v3** checkpoint. `create_default_for_version("v3", ...)` is
# called in training.py so the loaded weights are guaranteed to be v3 rather
# than whichever model `model_path="auto"` happens to resolve to.
TABPFN_MODEL_VERSION = "v3"
TABPFN_N_ESTIMATORS = 32
TABPFN_IGNORE_LIMITS = True

# TabPFN cloud switch. If TABPFN_USE_CLOUD env var is set AND the
# `tabpfn_client` package is importable AND TABPFN_API_TOKEN is set, the
# training loop will route TabPFN calls through the hosted service. The token
# is read from the environment only — never persist it to disk.
TABPFN_USE_CLOUD = bool(os.environ.get("TABPFN_USE_CLOUD"))

# --- SHAP explainability -----------------------------------------------------
# Production ("main publication") config — the clean, final SHAP dataset run on
# the HURCS A100 for all 12 targets, both models:
#     250 explained / 2048 coalitions / K=5 background / TabPFN-32.
# KernelSHAP cost is dominated by model `.predict()` calls and scales as
#     n_explain x nsamples x K x n_estimators.
SHAP_N_EXPLAIN = 250            # instances attributed per (target, model)
SHAP_N_BACKGROUND = 80          # rows sampled before k-means summarisation
SHAP_KMEANS = 5                 # k-means clusters for the background (K)
SHAP_NSAMPLES = 2048            # KernelSHAP coalitions per explained instance
                                # (well into the converged regime for 17 features)
TABPFN_SHAP_N_ESTIMATORS = 32   # TabPFN ensemble for explanation — matched to
                                # the prediction model (32) per reviewer request;
                                # an 8-estimator check gave ρ≥0.995 vs this.

# Supplementary robustness run uses a reduced explained sample: rank
# correlations of mean|SHAP| converge well below the main n_explain, so 150
# keeps the (otherwise quadratic-feeling) K=20 x 2048 settings tractable. The
# main run stays at SHAP_N_EXPLAIN above; only the robustness sweep uses this.
SHAP_ROBUSTNESS_N_EXPLAIN = 150

# --- Plot styling ------------------------------------------------------------
MODEL_PALETTE = {"AutoGluon": "#1A365D", "TabPFN": "#DD6B20"}
