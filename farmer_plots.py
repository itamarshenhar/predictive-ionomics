"""Fig. 8 of the article — commercial-cohort validation, one strategy per figure.

Controlled-CV out-of-fold predictions form a grey hexbin background per model;
the commercial-cohort predictions for the chosen strategy are overlaid, coloured
by model, with each model's multiplicative bias as an inset.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TARGETS_SPEC
from plotting import _grid_shape, _setup_style


def fig2_single_strategy(
    cv_preds: pd.DataFrame, farmer_preds: pd.DataFrame, strategy: str,
    out_path: str, targets: Iterable[str] = TARGETS_SPEC,
) -> str:
    """Fig 2 (per-strategy) — Actual vs Predicted for ONE strategy.

    Controlled CV cloud as a grey hexbin background; farmer points for the
    chosen strategy coloured by model (AutoGluon vs TabPFN), with the
    multiplicative bias (mean predicted / mean observed) of each model as an
    inset. No r / R2 insets: with the commercial cohort grown under a single
    fertigation recipe the models are not being asked to discriminate between
    plants, so a regression-fit statistic would be uninformative.
    """
    from matplotlib.lines import Line2D
    import matplotlib as mpl
    from plotting import _disp
    _setup_style()
    targets = list(targets)
    rows, cols = _grid_shape(len(targets))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.2),
                             squeeze=False)
    flat = axes.flatten()
    for ax in flat[len(targets):]:
        ax.set_axis_off()

    model_color = {"AutoGluon": "#1A365D", "TabPFN": "#DD6B20"}
    # Controlled-CV reference clouds, one constant grey per model: lighter grey
    # for AutoGluon, darker grey for TabPFN. Single-colour maps keep every
    # occupied hexbin at the same shade regardless of point density.
    cloud_cmap = {
        "AutoGluon": mpl.colors.ListedColormap(["#bdbdbd"]),  # lighter grey
        "TabPFN":    mpl.colors.ListedColormap(["#4d4d4d"]),  # darker grey
    }
    fp = farmer_preds[farmer_preds["Strategy"] == strategy]

    for tgt, ax in zip(targets, flat):
        cv_t = cv_preds[cv_preds["Target"] == tgt]
        f_t = fp[fp["Target"] == tgt]

        if cv_t.empty and f_t.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{_disp(tgt)}\n(no data)", ha="center",
                    va="center", transform=ax.transAxes, color="grey",
                    fontsize=22)
            continue

        # Draw AutoGluon (lighter) first, TabPFN (darker) on top.
        for cloud_model in ("AutoGluon", "TabPFN"):
            cm = cv_t[cv_t["Model"] == cloud_model] if "Model" in cv_t else cv_t
            if cm.empty:
                continue
            ax.hexbin(cm["Actual"], cm["Predicted"],
                      gridsize=25, cmap=cloud_cmap[cloud_model], mincnt=1,
                      alpha=0.55, linewidths=0.2,
                      edgecolors=cloud_cmap[cloud_model].colors[0])

        bias_lines = []          # (text, colour) per model, drawn as an inset
        for mdl, color in model_color.items():
            sub = f_t[f_t["Model"] == mdl]
            if sub.empty:
                continue
            # Multiplicative bias: mean predicted / mean observed. Reported
            # instead of R^2 because the commercial cohort holds all seven
            # fertigation inputs constant, so the models are not being asked to
            # discriminate between plants — only to place the central estimate.
            obs_mean = sub["Actual"].mean()
            if obs_mean and np.isfinite(obs_mean) and obs_mean != 0:
                bias_lines.append((f"{mdl[:2]} \u00d7{sub['Predicted'].mean()/obs_mean:.2f}",
                                   color))
            ax.errorbar(sub["Actual"], sub["Predicted"], yerr=None,
                        fmt="o", ms=7, lw=0, elinewidth=0.6, capsize=1.5,
                        color=color, alpha=0.85)

        vals = pd.concat([cv_t.get("Actual"), f_t.get("Actual"),
                          cv_t.get("Predicted"), f_t.get("Predicted")],
                         axis=0).dropna()
        if not vals.empty:
            lo, hi = float(vals.min()), float(vals.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.7)

        # Bias inset (one coloured line per model), borderless, upper-left.
        for i, (txt, col) in enumerate(bias_lines):
            ax.text(0.035, 0.975 - i * 0.075, txt, transform=ax.transAxes,
                    fontsize=15, va="top", color=col, fontweight="medium")

        ax.set_title(_disp(tgt), fontsize=24, pad=8)
        ax.set_xlabel("Actual", fontsize=20)
        ax.set_ylabel("Predicted", fontsize=20)
        ax.tick_params(axis="both", labelsize=15)

    # Single shared legend at the bottom (frees the panels of in-axes legends).
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=12,
               markerfacecolor=model_color["AutoGluon"], markeredgecolor="none",
               label="AutoGluon"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=12,
               markerfacecolor=model_color["TabPFN"], markeredgecolor="none",
               label="TabPFN"),
        Line2D([0], [0], linestyle="--", color="black", alpha=0.7, label="1:1 line"),
        Line2D([0], [0], marker="h", linestyle="none", markersize=12,
               markerfacecolor="#bdbdbd", markeredgecolor="none",
               label="AutoGluon CV"),
        Line2D([0], [0], marker="h", linestyle="none", markersize=12,
               markerfacecolor="#4d4d4d", markeredgecolor="none",
               label="TabPFN CV"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=18, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
