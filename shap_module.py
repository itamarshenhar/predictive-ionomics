"""SHAP explainability — computed for BOTH AutoGluon 1.5 and TabPFN v3.

For each target we explain the model trained on fold 0's training split, then
explain a common sample drawn from the full dataset so the two models are
attributed on identical rows (enabling a like-for-like comparison).

Outputs per target into <run_dir>/shap/<target>/ (suffixed by model):
  shap_values_<model>.csv         (model in {AutoGluon, TabPFN})
  shap_samples_<model>.csv
  beeswarm_<model>.pdf
  dependence_<model>_<feature>.pdf       (top-3 features by mean |SHAP|)
  interaction_<model>_<feature>.pdf      (top-3, coloured by strongest interactor)
  force_<model>_<level>.pdf              (low / median / high prediction)

A run-level AutoGluon-vs-TabPFN attribution-consistency figure is written to
<run_dir>/plots/SHAP_Model_Comparison.pdf.

Backward compatibility: legacy unsuffixed `shap_values.csv` / `shap_samples.csv`
written by earlier runs are treated as the AutoGluon outputs when reading.
"""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from config import (FEATURES, TARGETS_AVAILABLE, RANDOM_STATE, SHAP_N_EXPLAIN,
                    SHAP_N_BACKGROUND, SHAP_KMEANS, SHAP_NSAMPLES,
                    TABPFN_SHAP_N_ESTIMATORS)

MODELS = ("AutoGluon", "TabPFN")

# Display-name overrides for feature labels in figures (underlying data columns
# and SHAP values are unchanged — this only relabels the axis text).
FEATURE_DISPLAY = {"PAR_SUM": "PPFD_SUM"}


def _disp_feat(name: str) -> str:
    return FEATURE_DISPLAY.get(name, name)


# --- model loaders -----------------------------------------------------------
def _load_best_predictor(run_dir: str, target: str, fold_idx: int = 0):
    """Load the AutoGluon predictor for the full-data step of a given fold."""
    from autogluon.tabular import TabularPredictor
    base = os.path.join(run_dir, "models", target, f"Fold_{fold_idx}_Step_1.00")
    if not os.path.isdir(base):
        cand = sorted(
            d for d in os.listdir(os.path.join(run_dir, "models", target))
            if d.startswith(f"Fold_{fold_idx}_")
        )
        if not cand:
            raise FileNotFoundError(f"No AutoGluon model for {target} fold {fold_idx}")
        base = os.path.join(run_dir, "models", target, cand[-1])
    return TabularPredictor.load(base, require_version_match=False)


def _fit_tabpfn_for_shap(X_train: pd.DataFrame, y_train: pd.Series,
                         n_estimators: int | None = None):
    """Fit a TabPFN-v3 for explanation at the requested SHAP ensemble size.

    `n_estimators` defaults to TABPFN_SHAP_N_ESTIMATORS (32, matched to the
    prediction model). KernelSHAP calls .predict() ~nsamples times per
    instance, so the ensemble size is the dominant cost multiplier.
    """
    from tabpfn import TabPFNRegressor
    from config import (DEVICE, NUM_CORES, TABPFN_IGNORE_LIMITS,
                        TABPFN_MODEL_VERSION)
    n_est = n_estimators if n_estimators is not None else TABPFN_SHAP_N_ESTIMATORS
    kwargs = dict(device=DEVICE, n_estimators=n_est,
                  n_jobs=NUM_CORES, ignore_pretraining_limits=TABPFN_IGNORE_LIMITS,
                  random_state=RANDOM_STATE)
    try:
        m = TabPFNRegressor.create_default_for_version(TABPFN_MODEL_VERSION, **kwargs)
    except TypeError:
        for bad in ("n_jobs", "ignore_pretraining_limits"):
            kwargs.pop(bad, None)
        m = TabPFNRegressor.create_default_for_version(TABPFN_MODEL_VERSION, **kwargs)
    m.fit(X_train.values, y_train.values)
    return m


# --- SHAP predict-fn wrappers ------------------------------------------------
class _AGWrapper:
    """Vectorised predict_fn for an AutoGluon TabularPredictor."""
    def __init__(self, predictor, features):
        self.predictor = predictor
        self.features = list(features)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.features)
        return self.predictor.predict(df).values


class _TabPFNWrapper:
    """Vectorised predict_fn for a fitted TabPFN regressor."""
    def __init__(self, model):
        self.model = model

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X))


# --- per-model SHAP worker ---------------------------------------------------
def _shap_one_model(model_name: str, predict_fn, X_bg: pd.DataFrame,
                    X_ex: pd.DataFrame, out_dir: str, target: str,
                    k: int | None = None, nsamples: int | None = None) -> dict:
    """Run KernelSHAP for a single (model, target) and emit all artifacts.

    `k` (k-means background centroids) and `nsamples` (coalitions per instance)
    default to the config globals; the cluster runner overrides them per cell to
    sweep the robustness axes.
    """
    k = k if k is not None else SHAP_KMEANS
    nsamples = nsamples if nsamples is not None else SHAP_NSAMPLES
    explainer = shap.KernelExplainer(
        predict_fn, shap.kmeans(X_bg.values, min(k, len(X_bg))))
    shap_values = explainer.shap_values(
        X_ex.values, nsamples=nsamples, silent=True)

    pd.DataFrame(shap_values, columns=FEATURES).to_csv(
        os.path.join(out_dir, f"shap_values_{model_name}.csv"), index=False)
    X_ex.to_csv(os.path.join(out_dir, f"shap_samples_{model_name}.csv"),
                index=False)

    # --- beeswarm ---
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_ex, show=False)
    plt.title(f"SHAP beeswarm — {target} ({model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"beeswarm_{model_name}.pdf"),
                bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_features = [FEATURES[i] for i in order[:3]]

    for feat in top_features:
        try:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(feat, shap_values, X_ex, show=False)
            plt.title(f"SHAP dependence — {target} ({model_name}): {feat}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"dependence_{model_name}_{feat}.pdf"),
                        bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  [warn] dependence {model_name}/{target}/{feat}: {e}")

        try:
            inter_idx = shap.approximate_interactions(feat, shap_values, X_ex)
            inter_feat = X_ex.columns[inter_idx[0]]
            plt.figure(figsize=(7, 5))
            j = X_ex.columns.get_loc(feat)
            sc = plt.scatter(X_ex[feat], shap_values[:, j],
                             c=X_ex[inter_feat], cmap="coolwarm", s=22, alpha=0.85)
            plt.colorbar(sc, label=inter_feat)
            plt.xlabel(feat); plt.ylabel(f"SHAP value for {feat}")
            plt.title(f"{target} ({model_name}): {feat} × {inter_feat}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"interaction_{model_name}_{feat}.pdf"),
                        bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  [warn] interaction {model_name}/{target}/{feat}: {e}")

    preds = predict_fn(X_ex.values)
    q_idx = {"low": int(np.argmin(preds)),
             "median": int(np.argsort(preds)[len(preds) // 2]),
             "high": int(np.argmax(preds))}
    for label, idx in q_idx.items():
        try:
            plt.figure(figsize=(14, 3))
            shap.force_plot(explainer.expected_value, shap_values[idx, :],
                            X_ex.iloc[idx, :], matplotlib=True, show=False)
            plt.title(f"{target} ({model_name}) — {label} prediction sample")
            plt.savefig(os.path.join(out_dir, f"force_{model_name}_{label}.pdf"),
                        bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  [warn] force {model_name}/{target}/{label}: {e}")

    return {"model": model_name, "top_features": top_features,
            "mean_abs": mean_abs}


def compute_shap_for_target(run_dir: str, df: pd.DataFrame, target: str,
                            fold_idx: int = 0,
                            n_background: int = SHAP_N_BACKGROUND,
                            n_explain: int = SHAP_N_EXPLAIN,
                            models: Iterable[str] = MODELS,
                            train_idx: np.ndarray | None = None,
                            k: int | None = None,
                            nsamples: int | None = None,
                            n_estimators: int | None = None,
                            seed: int = RANDOM_STATE,
                            shap_subdir: str = "shap") -> dict:
    """Build SHAP artifacts for one target, for each requested model.

    `train_idx` (positional rows of `df`) is fold_idx's training split — used
    to fit TabPFN on exactly the data the saved AutoGluon model saw. If None,
    TabPFN is fit on all rows.

    Sweep knobs (None -> config defaults), used by the cluster runner:
      `k`            k-means background centroids (K).
      `nsamples`     KernelSHAP coalitions per instance.
      `n_estimators` TabPFN explanation ensemble size.
      `seed`         RNG seed for the *explained* sample — this is the only
                     thing that changes between explained-set A and B. The
                     background sample is always drawn with RANDOM_STATE so the
                     A/B comparison isolates the explained-set effect.
      `shap_subdir`  output sub-tree under run_dir (default "shap"; the cluster
                     runner passes e.g. "shap_runs/<setting_id>").
    """
    out_dir = os.path.join(run_dir, shap_subdir, target)
    os.makedirs(out_dir, exist_ok=True)

    X_all = df[FEATURES]
    rng_bg = np.random.RandomState(RANDOM_STATE)        # fixed across A/B
    rng_ex = np.random.RandomState(seed)                # set A=RANDOM_STATE, B=other
    bg_idx = rng_bg.choice(len(X_all), size=min(n_background, len(X_all)),
                           replace=False)
    ex_idx = rng_ex.choice(len(X_all), size=min(n_explain, len(X_all)),
                           replace=False)
    X_bg = X_all.iloc[bg_idx].reset_index(drop=True)
    X_ex = X_all.iloc[ex_idx].reset_index(drop=True)

    results = {}
    for model_name in models:
        if model_name == "AutoGluon":
            predictor = _load_best_predictor(run_dir, target, fold_idx)
            predict_fn = _AGWrapper(predictor, FEATURES)
        elif model_name == "TabPFN":
            if train_idx is not None:
                X_tr = df.iloc[train_idx][FEATURES]
                y_tr = df.iloc[train_idx][target]
            else:
                X_tr, y_tr = df[FEATURES], df[target]
            predict_fn = _TabPFNWrapper(
                _fit_tabpfn_for_shap(X_tr, y_tr, n_estimators=n_estimators))
        else:
            raise ValueError(f"Unknown model {model_name}")

        results[model_name] = _shap_one_model(
            model_name, predict_fn, X_bg, X_ex, out_dir, target,
            k=k, nsamples=nsamples)

    return {"target": target, "out_dir": out_dir,
            "n_explain": len(X_ex), "models": results}


# ---------------------------------------------------------------------------
#  Publication-quality SHAP beeswarm grid
# ---------------------------------------------------------------------------
def _shap_cmap():
    """The canonical SHAP beeswarm colormap (blue=low -> red/magenta=high),
    matching shap.plots.beeswarm. Falls back to RdBu_r if unavailable."""
    try:
        from shap.plots import colors as _sc
        return _sc.red_blue
    except Exception:
        import matplotlib as mpl
        return _shap_cmap()
def _beeswarm_into_ax(ax, shap_values: np.ndarray, X: pd.DataFrame,
                      feature_order: list, top_k: int = 8,
                      cmap=None, dot_size: float = 7.0,
                      jitter_strength: float = 0.32,
                      ylabel_fs: int = 8, xtick_fs: int = 7,
                      xlabel_fs: int = 8, show_yticklabels: bool = True):
    """Render a SHAP-style beeswarm into a single matplotlib axis.

    Hand-rolled (no `shap.summary_plot` call) so we have full control over
    sizing, ordering, and the colour mapping. Vertical jitter is density-aware:
    points in a high-density horizontal bin get a wider spread, which is the
    same trick SHAP itself uses to avoid visual collisions.
    """
    import matplotlib as mpl
    cmap = cmap or _shap_cmap()

    n_features = min(top_k, len(feature_order))
    order = feature_order[:n_features]                  # already sorted by |SHAP|
    y_positions = list(range(n_features - 1, -1, -1))   # top of plot = first

    rng = np.random.RandomState(0)
    for y, feat in zip(y_positions, order):
        col = X.columns.get_loc(feat)
        sv = shap_values[:, col]
        fv = X[feat].values.astype(float)

        # Color by feature value, robustly scaled to 5th–95th percentile.
        v_lo, v_hi = np.nanpercentile(fv, [5, 95])
        if v_hi == v_lo:
            color_norm = np.full_like(fv, 0.5)
        else:
            color_norm = np.clip((fv - v_lo) / (v_hi - v_lo), 0.0, 1.0)

        # Density-aware vertical jitter so dots don't pile into a thick line.
        # Histogram counts along the SHAP-value axis act as a local density.
        counts, edges = np.histogram(sv, bins=30)
        bin_idx = np.clip(np.digitize(sv, edges) - 1, 0, len(counts) - 1)
        density = counts[bin_idx]
        max_d = density.max() if density.size else 1
        scale = np.sqrt(density / max(max_d, 1)) * jitter_strength
        jitter = (rng.rand(len(sv)) - 0.5) * 2 * scale

        ax.scatter(sv, np.full_like(sv, y, dtype=float) + jitter,
                   c=color_norm, cmap=cmap, s=dot_size,
                   alpha=0.78, edgecolors="none", rasterized=False,
                   vmin=0.0, vmax=1.0)

    ax.axvline(0, color="grey", lw=0.6, alpha=0.7)
    ax.set_yticks(y_positions)
    if show_yticklabels:
        ax.set_yticklabels([_disp_feat(f) for f in order], fontsize=ylabel_fs)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=xtick_fs)
    ax.set_xlabel("SHAP value", fontsize=xlabel_fs)
    ax.set_ylim(-0.6, n_features - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _read_shap_pair(shap_dir: str, model: str):
    """Return (shap_values ndarray, X samples DataFrame) for a model, or None.

    Prefers `shap_values_<model>.csv`; for AutoGluon, falls back to the legacy
    unsuffixed `shap_values.csv` written by earlier single-model runs.
    """
    sv_path = os.path.join(shap_dir, f"shap_values_{model}.csv")
    ex_path = os.path.join(shap_dir, f"shap_samples_{model}.csv")
    if not os.path.exists(sv_path) and model == "AutoGluon":
        legacy_sv = os.path.join(shap_dir, "shap_values.csv")
        legacy_ex = os.path.join(shap_dir, "shap_samples.csv")
        if os.path.exists(legacy_sv) and os.path.exists(legacy_ex):
            sv_path, ex_path = legacy_sv, legacy_ex
    if not (os.path.exists(sv_path) and os.path.exists(ex_path)):
        return None
    return pd.read_csv(sv_path).values, pd.read_csv(ex_path)


# ---------------------------------------------------------------------------
#  Paired macronutrient beeswarm matrix (AutoGluon | TabPFN side-by-side)
# ---------------------------------------------------------------------------
MACRO_TARGETS = ["NO3_output", "TRN_output", "P_output",
                 "K_output", "Ca_output", "Mg_output"]
def plot_macro_beeswarm_pairs(run_dir: str,
                              targets: list | None = None,
                              top_k: int = 8,
                              out_path: str | None = None) -> str:
    """Beeswarm matrix for the 6 macronutrients, AutoGluon | TabPFN per target.

    Each target occupies an adjacent pair of panels (left = AutoGluon, right =
    TabPFN). Both panels of a pair share the SAME feature ordering (by combined
    mean |SHAP|) and the SAME x-axis limits, so the two models' attributions are
    directly comparable. Layout: 3 rows × 4 columns (two targets per row); a
    single shared feature-value colourbar at the bottom.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from plotting import _disp, _setup_style

    _setup_style()
    targets = list(targets or MACRO_TARGETS)
    cmap = _shap_cmap()

    # 2 targets per row -> 4 columns; rows = ceil(n/2).
    n = len(targets)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 4, figsize=(4 * 4.0, nrows * 4.6),
                             squeeze=False)

    for idx, target in enumerate(targets):
        r, cpair = idx // 2, (idx % 2) * 2
        ax_ag, ax_pf = axes[r][cpair], axes[r][cpair + 1]

        ag = _read_shap_pair(os.path.join(run_dir, "shap", target), "AutoGluon")
        pf = _read_shap_pair(os.path.join(run_dir, "shap", target), "TabPFN")
        if ag is None or pf is None:
            for ax in (ax_ag, ax_pf):
                ax.set_axis_off()
            ax_ag.text(1.05, 0.5, f"{_disp(target)}\n(need both models)",
                       ha="center", va="center", transform=ax_ag.transAxes,
                       fontsize=16, color="grey")
            continue

        sv_ag, X_ag = ag
        sv_pf, X_pf = pf
        # Independent feature ordering per model: most -> least impactful for
        # that specific beeswarm (its own y-axis).
        order_ag = [X_ag.columns[i] for i in np.argsort(np.abs(sv_ag).mean(0))[::-1]]
        order_pf = [X_pf.columns[i] for i in np.argsort(np.abs(sv_pf).mean(0))[::-1]]
        # Shared symmetric x-limits within the pair so SHAP magnitudes are
        # comparable across the two models.
        kc_ag = [X_ag.columns.get_loc(f) for f in order_ag[:min(top_k, len(order_ag))]]
        kc_pf = [X_pf.columns.get_loc(f) for f in order_pf[:min(top_k, len(order_pf))]]
        xmax = max(np.abs(sv_ag[:, kc_ag]).max(), np.abs(sv_pf[:, kc_pf]).max())
        xlim = (-1.05 * xmax, 1.05 * xmax)

        _beeswarm_into_ax(ax_ag, sv_ag, X_ag, order_ag, top_k=top_k, cmap=cmap,
                          dot_size=9, ylabel_fs=13, xtick_fs=12, xlabel_fs=14,
                          show_yticklabels=True)
        _beeswarm_into_ax(ax_pf, sv_pf, X_pf, order_pf, top_k=top_k, cmap=cmap,
                          dot_size=9, ylabel_fs=13, xtick_fs=12, xlabel_fs=14,
                          show_yticklabels=True)
        ax_ag.set_xlim(*xlim); ax_pf.set_xlim(*xlim)
        ax_ag.set_title(f"{_disp(target)} · AutoGluon", fontsize=17,
                        fontweight="bold", pad=8)
        ax_pf.set_title(f"{_disp(target)} · TabPFN", fontsize=17,
                        fontweight="bold", pad=8)

    # Blank any unused panels.
    for j in range(n, nrows * 2):
        axes[j // 2][(j % 2) * 2].set_axis_off()
        axes[j // 2][(j % 2) * 2 + 1].set_axis_off()

    # Shared horizontal colourbar.
    fig.subplots_adjust(bottom=0.10, hspace=0.42, wspace=0.55)
    cbar_ax = fig.add_axes([0.30, 0.04, 0.42, 0.014])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Median", "High"])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Feature value (per-feature 5th–95th-percentile scaling)",
                   fontsize=13, labelpad=4)

    out_path = out_path or os.path.join(
        run_dir, "plots", "SHAP_Beeswarm_Macronutrients_paired.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
#  AutoGluon vs TabPFN attribution-consistency figure
# ---------------------------------------------------------------------------
def plot_shap_model_comparison(run_dir: str,
                               targets: list | None = None,
                               out_path: str | None = None) -> str:
    """Per-target scatter of mean |SHAP| per feature: AutoGluon vs TabPFN.

    Each point is one feature; x = AutoGluon mean |SHAP|, y = TabPFN mean
    |SHAP|. A 1:1 line and the coefficient of determination (R², the square of
    the Pearson correlation between the two attribution vectors) quantify how
    consistently the two architectures rank feature importance. This is the
    figure that substantiates the paper's claim that the high accuracy reflects
    a shared physiological signal rather than a model artifact.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from config import TARGETS_SPEC
    from plotting import _grid_shape, _setup_style, _disp

    _setup_style()
    targets = list(targets or TARGETS_SPEC)
    rows, cols = _grid_shape(len(targets))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.2),
                             squeeze=False)
    flat = axes.flatten()
    for ax in flat[len(targets):]:
        ax.set_axis_off()

    for ax, target in zip(flat, targets):
        shap_dir = os.path.join(run_dir, "shap", target)
        ag = _read_shap_pair(shap_dir, "AutoGluon")
        pf = _read_shap_pair(shap_dir, "TabPFN")
        if ag is None or pf is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{_disp(target)}\n(need both models)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=22, color="grey")
            continue

        ag_imp = np.abs(ag[0]).mean(axis=0)
        pf_imp = np.abs(pf[0]).mean(axis=0)
        feats = list(ag[1].columns)

        ax.scatter(ag_imp, pf_imp, s=60, color="#1A365D",
                   edgecolor="white", linewidth=0.5, zorder=3)
        hi = float(max(ag_imp.max(), pf_imp.max())) * 1.08
        ax.plot([0, hi], [0, hi], "--", color="grey", lw=1.0, zorder=1)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)

        # Label the 3 most important features. Default: centered above the
        # point; but relocate to the side/below when the point is near a panel
        # edge so the text never spills into the neighbouring plot.
        for i in np.argsort((ag_imp + pf_imp))[::-1][:3]:
            xf = ag_imp[i] / hi if hi else 0.0   # x position as axes fraction
            yf = pf_imp[i] / hi if hi else 0.0
            if xf > 0.72:                        # near right edge -> go left
                ha, va, off = "right", "center", (-7, 0)
            elif yf > 0.85:                      # near top edge -> go below
                ha, va, off = "center", "top", (0, -7)
            else:                                # interior -> centered above
                ha, va, off = "center", "bottom", (0, 6)
            ax.annotate(_disp_feat(feats[i]), (ag_imp[i], pf_imp[i]), fontsize=13,
                        xytext=off, textcoords="offset points",
                        ha=ha, va=va, annotation_clip=False)

        r, _ = pearsonr(ag_imp, pf_imp)
        r2 = r ** 2
        ax.text(0.04, 0.96, f"$R^2$ = {r2:.2f}", transform=ax.transAxes,
                fontsize=15, va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                          boxstyle="round,pad=0.2"))
        ax.set_title(_disp(target), fontsize=24, fontweight="bold", pad=8)
        ax.set_xlabel("AutoGluon mean |SHAP|", fontsize=18)
        ax.set_ylabel("TabPFN mean |SHAP|", fontsize=18)
        ax.tick_params(labelsize=15)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.26, hspace=0.30)  # gap between panels

    out_path = out_path or os.path.join(
        run_dir, "plots", "SHAP_Model_Comparison.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def run_shap(run_dir: str, df: pd.DataFrame,
             targets: Iterable[str] = TARGETS_AVAILABLE,
             fold_idx: int = 0, n_background: int = SHAP_N_BACKGROUND,
             n_explain: int = SHAP_N_EXPLAIN,
             models: Iterable[str] = MODELS,
             cv_scheme: str = "dobscv") -> list:
    """Compute SHAP for each target × model, then render the run-level grids.

    `df` must be the SAME DataFrame used for training (df_strat), so fold
    reconstruction yields the identical fold_idx training split that the saved
    AutoGluon models were trained on. `cv_scheme` must match the scheme the
    models were trained with, or fold_idx's training split won't line up.
    """
    from data import build_folds
    import config

    # Reconstruct the fold split to fit TabPFN on the same training rows.
    _, folds = build_folds(df, n_splits=config.N_SPLITS, scheme=cv_scheme)
    train_idx = folds[fold_idx][0] if fold_idx < len(folds) else None

    results = []
    for tgt in tqdm(list(targets), desc="SHAP per target × model"):
        try:
            results.append(compute_shap_for_target(
                run_dir, df, tgt, fold_idx=fold_idx,
                n_background=n_background, n_explain=n_explain,
                models=models, train_idx=train_idx,
            ))
        except FileNotFoundError as e:
            print(f"[skip] {tgt}: {e}")

    # Render the article figures (both models are needed for all three).
    if set(models) >= {"AutoGluon", "TabPFN"}:
        try:
            render_article_figures(run_dir)
        except Exception as e:
            print(f"[warn] SHAP figures: {e}")
    return results


MICRONUTRIENTS = ["Fe_output", "Mn_output", "Zn_output",
                  "Bo_output", "Mo_output", "Na_output"]


def render_article_figures(run_dir: str) -> None:
    """Figs. 5, 6 and 7 of the article from the per-target shap_values CSVs."""
    print("Wrote", plot_macro_beeswarm_pairs(run_dir))                 # Fig. 5
    print("Wrote", plot_macro_beeswarm_pairs(                          # Fig. 6
        run_dir, targets=MICRONUTRIENTS,
        out_path=os.path.join(run_dir, "plots",
                              "SHAP_Beeswarm_Micronutrients_paired.pdf")))
    print("Wrote", plot_shap_model_comparison(run_dir))                # Fig. 7


# ---------------------------------------------------------------------------
#  CLI: python shap_module.py --run-dir <dir>   ->  Figs. 5, 6, 7
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=(
        "Render the article's SHAP figures from already-computed per-target "
        "shap_values_<model>.csv files (no recomputation)."
    ))
    ap.add_argument("--run-dir", required=True,
                    help="Existing Ionomic_Prediction_Run_* directory.")
    args = ap.parse_args()
    render_article_figures(args.run_dir)
