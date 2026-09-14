"""Decoupled plotting from artifacts written by training.py.

Reads only the CSVs under <run_dir>/{metrics,predictions,importance,architecture}.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from config import MODEL_PALETTE, TARGETS_SPEC


# The two learners compared in the article.
HEADLINE_MODELS = ("AutoGluon", "TabPFN")


def _setup_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 400,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titleweight": "bold",
    })


def _grid_iter(targets, axes):
    """Pair each target with a grid axis; trailing axes are blanked."""
    flat = axes.flatten()
    for i, ax in enumerate(flat):
        yield (targets[i] if i < len(targets) else None), ax


def _grid_shape(n_targets: int) -> tuple:
    """Square-ish grid that holds n_targets cells.

    The pipeline ships a 4x4 default for the 12-target spec (4 blank cells).
    """
    import math
    cols = 4 if n_targets > 9 else (3 if n_targets > 4 else 2)
    rows = int(math.ceil(n_targets / cols))
    return rows, cols


# Display-name overrides for panel titles (e.g. boron Bo -> B).
_TARGET_DISPLAY = {"Bo_output": "B"}


def _disp(tgt: str) -> str:
    return _TARGET_DISPLAY.get(tgt, tgt.replace("_output", ""))


# --- 1. Actual vs Predicted -------------------------------------------------
def plot_actual_vs_predicted(run_dir: str, targets=TARGETS_SPEC) -> str:
    _setup_style()
    preds = pd.read_csv(os.path.join(run_dir, "predictions", "full_data_preds.csv"))
    preds = preds[preds["Model"].isin(HEADLINE_MODELS)]

    from matplotlib.lines import Line2D

    rows, cols = _grid_shape(len(list(targets)))
    # Larger panels give the big text room to breathe.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.2),
                             squeeze=False)
    for tgt, ax in _grid_iter(list(targets), axes):
        if tgt is None or tgt not in preds["Target"].unique():
            ax.set_axis_off()
            if tgt is not None:
                ax.text(0.5, 0.5, f"{_disp(tgt)}\n(not in data)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=22, color="grey")
            continue

        sub = preds[preds["Target"] == tgt]
        sns.scatterplot(
            data=sub, x="Actual", y="Predicted", hue="Model",
            palette=MODEL_PALETTE, alpha=0.55, s=22, ax=ax, edgecolor="none",
            legend=False,                      # one shared legend instead
        )
        lo = float(min(sub["Actual"].min(), sub["Predicted"].min()))
        hi = float(max(sub["Actual"].max(), sub["Predicted"].max()))
        rng = hi - lo
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1, alpha=0.6)
        ax.set_xlim(lo - 0.03 * rng, hi + 0.03 * rng)
        ax.set_ylim(lo - 0.03 * rng, hi + 0.05 * rng)
        ax.set_title(_disp(tgt), fontsize=24, pad=8)

        lines = []
        for m_name in ("AutoGluon", "TabPFN"):
            d = sub[sub["Model"] == m_name]
            if d.empty:
                continue
            r2 = r2_score(d["Actual"], d["Predicted"])
            rmse = root_mean_squared_error(d["Actual"], d["Predicted"])
            mae = mean_absolute_error(d["Actual"], d["Predicted"])
            lines.append(f"{m_name[:2]}: $R^2$={r2:.2f}  RMSE={rmse:.2g}  MAE={mae:.2g}")
        # Inset sits inside the panel (upper-left), no surrounding outline;
        # a faint white background keeps it legible over the scatter.
        ax.text(0.035, 0.975, "\n".join(lines), transform=ax.transAxes,
                fontsize=14, va="top", linespacing=1.4,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                          boxstyle="round,pad=0.2"))
        ax.set_xlabel("Actual", fontsize=20)
        ax.set_ylabel("Predicted", fontsize=20)
        ax.tick_params(axis="both", labelsize=15)

    # Single shared legend at the bottom — frees the lower-right of every panel.
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=12,
               markerfacecolor=MODEL_PALETTE["AutoGluon"], markeredgecolor="none",
               label="AutoGluon"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=12,
               markerfacecolor=MODEL_PALETTE["TabPFN"], markeredgecolor="none",
               label="TabPFN"),
        Line2D([0], [0], linestyle="--", color="black", alpha=0.6, label="1:1 line"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=20, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.03, 1, 1.0))
    out = os.path.join(run_dir, "plots", "Actual_vs_Predicted_grid.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# --- 2. Learning curves (4 lines: AG-Train, AG-Test, TabPFN-Train, TabPFN-Test)
# Backwards-compat: older metrics CSVs without a "Set" column are treated as
# Set=Test (the legacy behavior), giving 2 lines.
def _paired_ttest_train_vs_test(raw: pd.DataFrame, model: str,
                                training_pct: float):
    """Paired t-test on the 5-fold R² (Train vs Test) for one (model, pct).

    Returns (p_value, n_pairs) or (None, n_pairs) if too few paired folds.
    """
    from scipy import stats as _stats
    rm = raw[(raw["Model"] == model) & (raw["Training_Pct"] == training_pct)]
    tr = rm[rm["Set"] == "Train"].set_index("Fold")["R2"]
    te = rm[rm["Set"] == "Test"].set_index("Fold")["R2"]
    common = tr.index.intersection(te.index)
    tr_v = tr.loc[common].dropna(); te_v = te.loc[common].dropna()
    common_clean = tr_v.index.intersection(te_v.index)
    if len(common_clean) < 3:
        return None, len(common_clean)
    res = _stats.ttest_rel(tr_v.loc[common_clean].values,
                           te_v.loc[common_clean].values)
    return float(res.pvalue), int(len(common_clean))


def _significance_marker(p: float) -> str:
    if p <= 0.001: return "***"
    if p <= 0.01:  return "**"
    if p <= 0.05:  return "*"
    return ""


def _lc_series_style(model: str, setname: str) -> dict:
    """Return matplotlib kwargs for a (model, Train/Test) series.

    Train vs. Test is encoded in TWO redundant visual channels so the legend
    is unambiguous even if one channel rasterises poorly:
      * line style    : Train = dashed, Test = solid
      * marker filling: Train = open (white face), Test = filled (line colour)
    """
    color = MODEL_PALETTE.get(model, "grey")
    is_train = (setname == "Train")
    return dict(
        color=color,
        linestyle="--" if is_train else "-",
        marker="o",
        markersize=5,
        markeredgecolor=color,
        markerfacecolor="white" if is_train else color,
        linewidth=1.6,
    )


def plot_learning_curves(run_dir: str, targets=TARGETS_SPEC) -> str:
    _setup_style()
    lc = pd.read_csv(os.path.join(run_dir, "metrics", "learning_curve.csv"))
    # Keep the headline learning-curve grid two-model (see plot_actual_vs_predicted).
    lc = lc[lc["Model"].isin(HEADLINE_MODELS)]
    if "Set" not in lc.columns:
        lc["Set"] = "Test"
    lc["Series"] = lc["Model"] + " — " + lc["Set"]

    from matplotlib.lines import Line2D

    rows, cols = _grid_shape(len(list(targets)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.2),
                             squeeze=False)
    # Stable legend order: model groups together, Test before Train.
    series_order = ["AutoGluon — Test", "AutoGluon — Train",
                    "TabPFN — Test",    "TabPFN — Train"]

    for tgt, ax in _grid_iter(list(targets), axes):
        if tgt is None or tgt not in lc["Target"].unique():
            ax.set_axis_off()
            if tgt is not None:
                ax.text(0.5, 0.5, f"{_disp(tgt)}\n(not in data)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=22, color="grey")
            continue

        sub = lc[lc["Target"] == tgt]
        agg = (sub.groupby(["Series", "Model", "Set", "Training_Pct"],
                           as_index=False)["R2"]
                  .agg(mean="mean", std="std"))

        for series_name in series_order:
            g = agg[agg["Series"] == series_name].sort_values("Training_Pct")
            if g.empty:
                continue
            style = _lc_series_style(g["Model"].iloc[0], g["Set"].iloc[0])
            ax.plot(g["Training_Pct"], g["mean"], label=series_name, **style)
            ax.fill_between(g["Training_Pct"],
                            g["mean"] - g["std"].fillna(0),
                            g["mean"] + g["std"].fillna(0),
                            color=style["color"], alpha=0.08, linewidth=0)

        # --- Paired t-test asterisks (Train vs Test, per model) -----------
        # Stagger AutoGluon and TabPFN vertically so co-significant asterisks
        # don't overlap when both models are over-fitting at the same x.
        AST_OFFSET = {"AutoGluon": 0.035, "TabPFN": 0.075}
        train_pcts = sorted(p for p in sub["Training_Pct"].unique() if p > 0)
        for model_name in ("AutoGluon", "TabPFN"):
            color = MODEL_PALETTE.get(model_name, "grey")
            for pct in train_pcts:
                p_val, n = _paired_ttest_train_vs_test(sub, model_name, pct)
                if p_val is None:
                    continue
                marker = _significance_marker(p_val)
                if not marker:
                    continue
                # Anchor above the higher of train/test mean for this model.
                mm = agg[(agg["Model"] == model_name) &
                         (agg["Training_Pct"] == pct)]["mean"]
                if mm.empty:
                    continue
                y_pos = float(mm.max()) + AST_OFFSET[model_name]
                ax.text(pct, y_pos, marker, ha="center", va="bottom",
                        color=color, fontsize=20, fontweight="bold",
                        clip_on=False)

        ax.set_xlim(0, 100); ax.set_ylim(-0.1, 1.20)
        ax.set_xticks(range(0, 101, 20))
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_title(_disp(tgt), fontsize=24, pad=8)
        ax.set_ylabel("$R^2$", fontsize=20)
        ax.set_xlabel("Training data (%)", fontsize=20)
        ax.tick_params(axis="both", labelsize=15)

    # Single shared legend at the bottom (4 series), plus a concise note on the
    # Train/Test encoding and the significance asterisks.
    legend_handles = []
    for series_name in series_order:
        model, setname = series_name.split(" — ")
        st = _lc_series_style(model, setname)
        legend_handles.append(Line2D([0], [0], **st, label=series_name))
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.005),
               handlelength=3.0, columnspacing=2.0)

    fig.tight_layout(rect=(0, 0.03, 1, 1.0))
    out = os.path.join(run_dir, "plots", "Learning_Curves_grid.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# --- 3. Per-target permutation importance bars ------------------------------
def generate_all_plots(run_dir: str) -> dict:
    """Render the two training-stage article figures (12-target grids)."""
    paths = {
        "actual_vs_predicted": plot_actual_vs_predicted(run_dir, TARGETS_SPEC),   # Fig. 3
        "learning_curves":     plot_learning_curves(run_dir, TARGETS_SPEC),       # Fig. 4
    }
    return paths
