"""Fig S6 — DOB-SCV cross-validation schematic (symmetric 3-2-3 dose tiers).

Panel A: per-experiment dose composition (Low/Mid/High plants), symmetric binning.
Panel B: the ACTUAL fold assignment from the production partition
         (data.build_folds(scheme="dobscv") — global round-robin within each
         dose tier), colored by dose tier when a condition is held out for test;
         beige = train. Not a schematic: every colored cell is the real test fold
         of that dosing condition.

Data are recomputed from the production tiering (data._per_experiment_dose_tier,
symmetric ÷(n-1)) and the production folds.

    python make_dobscv_figure.py   ->  fig_S6_dobscv.{pdf,png}
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

from data import (_condition_key, _per_experiment_dose_tier, build_folds,
                  load_dataset)

# --- font: prefer a clean sans-serif like the original ----------------------
for _name in ("Arial", "Helvetica"):
    try:
        p = fm.findfont(_name, fallback_to_default=False)
        fm.fontManager.addfont(p)
        FONT = _name
        break
    except Exception:
        FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "pdf.fonttype": 42, "svg.fonttype": "none"})

BLUE, ORANGE, GREY, BEIGE = "#4C72B0", "#DD8452", "#8E9BAA", "#EDEAD7"
TIER_COL = {0: BLUE, 1: ORANGE, 2: GREY}
INK = "#1a1a1a"
MIN = {1: "N", 2: "N", 3: "K", 4: "K", 5: "Mg", 6: "Mg", 7: "Fe", 8: "Fe",
       9: "P", 10: "P", 11: "Ca", 12: "Ca", 13: "N", 14: "K", 15: "P", 16: "Mo"}
NFOLDS = 5


def get_data():
    # ACTUAL production partition (global round-robin within each dose tier).
    df, folds = build_folds(load_dataset(), scheme="dobscv")
    df["cond"] = _condition_key(df)
    df["tier"] = _per_experiment_dose_tier(df).values
    fold_id = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k
    df["fold"] = fold_id
    exps = sorted(df["Experiment_ID"].unique())
    # Panel A: plants per (exp, tier)
    compA = (df.groupby(["Experiment_ID", "tier"]).size().unstack("tier")
             .reindex(exps).reindex(columns=[0, 1, 2]).fillna(0).astype(int))
    # Panel B: per exp, conditions ordered low->high dose -> (tier, real test fold)
    perexp = {}
    for e in exps:
        sub = df[df.Experiment_ID == e].drop_duplicates("cond").copy()
        sub["rank"] = sub["Active_Gradient"].rank(method="dense").astype(int)
        sub = sub.sort_values("rank")
        perexp[e] = list(zip(sub["tier"].tolist(), sub["fold"].tolist()))
    return exps, compA, perexp


def panel_A(ax, exps, compA):
    x = np.arange(len(exps))
    low, mid, high = compA[0].values, compA[1].values, compA[2].values
    for vals, base, col in [(low, np.zeros_like(low), BLUE),
                            (mid, low, ORANGE),
                            (high, low + mid, GREY)]:
        ax.bar(x, vals, bottom=base, color=col, width=0.72,
               edgecolor="white", linewidth=0.6)
        for xi, v, b in zip(x, vals, base):
            if v > 0:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{e}\n{MIN[e]}" for e in exps], fontsize=9)
    ax.set_ylabel("Plants per experiment", fontsize=12)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    # legend (top)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY)]
    ax.legend(handles, ["Low dose", "Mid dose", "High dose"], loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=11,
              handlelength=1.1, columnspacing=2.0)
    ax.text(-0.055, 1.06, "A", transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top")


def panel_B(ax, exps, perexp):
    gap = 1.2                       # gap between experiment blocks
    x0 = 0.0
    starts = {}
    for e in exps:
        starts[e] = x0
        x0 += len(perexp[e]) + gap
    total_w = x0 - gap

    for e in exps:
        sx = starts[e]
        cells = perexp[e]
        ncol = len(cells)
        # beige background for the whole block (5 rows x ncol)
        for c in range(ncol):
            for f in range(NFOLDS):
                y = NFOLDS - 1 - f      # fold 0 (Fold 1) at top
                ax.add_patch(Rectangle((sx + c, y), 1, 1, facecolor=BEIGE,
                                       edgecolor="white", linewidth=0.5))
        # colored test cells
        for c, (tier, fold) in enumerate(cells):
            y = NFOLDS - 1 - fold
            ax.add_patch(Rectangle((sx + c, y), 1, 1, facecolor=TIER_COL[tier],
                                   edgecolor="white", linewidth=0.5))
        # experiment header
        ax.text(sx + ncol / 2, NFOLDS + 0.25, f"E{e}·{MIN[e]}", ha="center",
                va="bottom", fontsize=9)
        # separator line after the block (except last)
        if e != exps[-1]:
            xline = sx + ncol + gap / 2
            ax.plot([xline, xline], [0, NFOLDS], color="#333333", lw=0.8)

    ax.set_xlim(-6.5, total_w + 0.3)
    ax.set_ylim(-0.4, NFOLDS + 1.1)
    for f in range(NFOLDS):
        ax.text(-1.0, NFOLDS - 1 - f + 0.5, f"Fold {f + 1}", ha="right",
                va="center", fontsize=10)
    ax.axis("off")
    ax.text(-0.055, 1.02, "B", transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top")
    # legend (bottom)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY, BEIGE)]
    labels = ["Test — Low dose", "Test — Mid dose", "Test — High dose", "Train"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False, fontsize=10.5, handlelength=1.1,
              columnspacing=2.0)


def main():
    exps, compA, perexp = get_data()
    tot = compA.sum()
    print(f"Panel A totals: Low={tot[0]} Mid={tot[1]} High={tot[2]} "
          f"(plants={tot.sum()})")

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.22)
    panel_A(fig.add_subplot(gs[0]), exps, compA)
    panel_B(fig.add_subplot(gs[1]), exps, perexp)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.95, bottom=0.06)
    fig.savefig("fig_S6_dobscv.pdf", bbox_inches="tight")
    fig.savefig("fig_S6_dobscv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_S6_dobscv.pdf/.png")


if __name__ == "__main__":
    main()
