"""Confirm TabPFN SHAP feature-importance rankings are stable to ensemble size.

For TRN and Ca: fit TabPFN-v3 at 8 and at 32 estimators on the same fold-0
training split, run identical KernelSHAP (same explained sample, background,
nsamples), and compare per-feature mean|SHAP| via Spearman rank correlation,
Pearson, and top-5 overlap.
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import shap
from scipy.stats import spearmanr, pearsonr

from data import load_dataset, build_folds
from config import (FEATURES, N_SPLITS, DEVICE, NUM_CORES, RANDOM_STATE,
                    TABPFN_IGNORE_LIMITS, TABPFN_MODEL_VERSION,
                    SHAP_N_EXPLAIN, SHAP_N_BACKGROUND, SHAP_KMEANS, SHAP_NSAMPLES)
from tabpfn import TabPFNRegressor

TARGETS = ["TRN_output", "Ca_output"]

def fit_tabpfn(X, y, n_est):
    kw = dict(device=DEVICE, n_estimators=n_est, n_jobs=NUM_CORES,
              ignore_pretraining_limits=TABPFN_IGNORE_LIMITS,
              random_state=RANDOM_STATE)
    try:
        m = TabPFNRegressor.create_default_for_version(TABPFN_MODEL_VERSION, **kw)
    except TypeError:
        for b in ("n_jobs", "ignore_pretraining_limits"): kw.pop(b, None)
        m = TabPFNRegressor.create_default_for_version(TABPFN_MODEL_VERSION, **kw)
    m.fit(X.values, y.values)
    return m

df = load_dataset()
CV_SCHEME = os.environ.get("CV_SCHEME", "dobscv")
print(f"cv_scheme={CV_SCHEME}  (fold-0 training split)", flush=True)
df_strat, folds = build_folds(df, n_splits=N_SPLITS, scheme=CV_SCHEME)
tr_idx = folds[0][0]
X_all = df_strat[FEATURES]
rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X_all), size=min(SHAP_N_BACKGROUND, len(X_all)), replace=False)
ex_idx = rng.choice(len(X_all), size=min(SHAP_N_EXPLAIN, len(X_all)), replace=False)
X_bg = X_all.iloc[bg_idx].reset_index(drop=True)
X_ex = X_all.iloc[ex_idx].reset_index(drop=True)

def mean_abs_shap(model):
    f = lambda A: model.predict(np.asarray(A))
    expl = shap.KernelExplainer(f, shap.kmeans(X_bg.values, min(SHAP_KMEANS, len(X_bg))))
    sv = expl.shap_values(X_ex.values, nsamples=SHAP_NSAMPLES, silent=True)
    return np.abs(sv).mean(0)

results = {}   # target -> (imp8, imp32, rho, r)
records = []
for tgt in TARGETS:
    y = df_strat.iloc[tr_idx][tgt]
    Xtr = df_strat.iloc[tr_idx][FEATURES]
    print(f"\n===== {tgt} =====", flush=True)
    t0 = time.time()
    imp8 = mean_abs_shap(fit_tabpfn(Xtr, y, 8))
    print(f"  8-estimator SHAP done ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    imp32 = mean_abs_shap(fit_tabpfn(Xtr, y, 32))
    print(f"  32-estimator SHAP done ({time.time()-t0:.0f}s)", flush=True)

    rho, _ = spearmanr(imp8, imp32)
    r, _ = pearsonr(imp8, imp32)
    results[tgt] = (imp8, imp32, rho, r)
    order8 = [FEATURES[i] for i in np.argsort(imp8)[::-1]]
    order32 = [FEATURES[i] for i in np.argsort(imp32)[::-1]]
    top5_overlap = len(set(order8[:5]) & set(order32[:5]))
    print(f"  Spearman rank corr (importance, 8 vs 32) = {rho:.3f}", flush=True)
    print(f"  Pearson  corr  (mean|SHAP|, 8 vs 32)     = {r:.3f}", flush=True)
    print(f"  Top-5 feature overlap                    = {top5_overlap}/5", flush=True)
    for fi, feat in enumerate(FEATURES):
        records.append(dict(Target=tgt, Feature=feat,
                            mean_abs_SHAP_8est=imp8[fi],
                            mean_abs_SHAP_32est=imp32[fi]))

# --- persist vectors ---
_vals_dir = os.environ.get("OUT_DIR",
                           os.path.join(os.environ.get("RUN_DIR", "."), "plots"))
os.makedirs(_vals_dir, exist_ok=True)
pd.DataFrame(records).to_csv(
    os.path.join(_vals_dir, "shap_estimator_check_values.csv"), index=False)
print("\nsaved shap_estimator_check_values.csv", flush=True)

# --- scatter figure: 8-est vs 32-est mean|SHAP| per feature ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "savefig.dpi": 600})
disp = {"TRN_output": "TRN", "Ca_output": "Ca"}
fig, axes = plt.subplots(1, len(TARGETS), figsize=(len(TARGETS) * 5.0, 5.0),
                         squeeze=False)
for ax, tgt in zip(axes[0], TARGETS):
    imp8, imp32, rho, r = results[tgt]
    hi = max(imp8.max(), imp32.max()) * 1.08
    ax.plot([0, hi], [0, hi], "--", color="grey", lw=1.0, zorder=1)
    ax.scatter(imp8, imp32, s=60, color="#1A365D", edgecolor="white",
               linewidth=0.5, zorder=3)
    # label the 3 most important features
    order = np.argsort(imp8 + imp32)[::-1][:3]
    for i in order:
        ax.annotate(FEATURES[i], (imp8[i], imp32[i]), fontsize=12,
                    xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(disp.get(tgt, tgt), fontsize=20, fontweight="bold", pad=8)
    ax.set_xlabel("mean |SHAP|  (8 estimators)", fontsize=15)
    ax.set_ylabel("mean |SHAP|  (32 estimators)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.text(0.04, 0.96,
            f"Spearman $\\rho$ = {rho:.3f}\nPearson $r$ = {r:.3f}",
            transform=ax.transAxes, fontsize=14, va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                      boxstyle="round,pad=0.2"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.tight_layout()
outdir = os.environ.get(
    "OUT_DIR",
    os.path.join(os.environ.get("RUN_DIR", "."), "plots"))
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, f"SHAP_Estimator_Stability.{ext}"),
                bbox_inches="tight")
plt.close(fig)
print("saved SHAP_Estimator_Stability.pdf/png", flush=True)
print("\nDONE", flush=True)
