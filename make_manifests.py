"""Generate SLURM-array job manifests for the SHAP cluster run.

Each manifest is a CSV with one row per (model, target, setting) cell. The
SLURM array index (SLURM_ARRAY_TASK_ID) selects the row; run_shap_cell.py reads
that row and computes exactly that one SHAP cell. Splitting by model lets the
GPU array (TabPFN) and the CPU array (AutoGluon) be sized and submitted
independently.

Writes into ./manifests/:
  main_tabpfn.csv   12 rows  (TabPFN  x 12 targets)   -> A100 GPU array
  main_ag.csv       12 rows  (AutoGluon x 12 targets) -> Glacier CPU array
  rob_tabpfn.csv    28 rows  (TabPFN  x 4 targets x 7 settings) -> A100 GPU array
  rob_ag.csv        28 rows  (AutoGluon x 4 targets x 7 settings) -> CPU array

Main config (per the publication spec):   250 / 2048 / K=5 / TabPFN-32.
Robustness targets:                        TRN, Ca, Fe, Mo.
Robustness axes (one-at-a-time from a 150 / 2048 / K=5 / fold0 / setA baseline):
  coalitions  1024 vs 2048
  K           5 vs 10 vs 20
  explained   set A vs set B
  fold        fold 1 vs fold 2 (baseline anchors fold 0)
"""
from __future__ import annotations

import csv
import os

import config

OUT_DIR = os.path.join(config.PROJECT_ROOT, "manifests")

# Column order is the contract with run_shap_cell.py.
FIELDS = ["row_id", "phase", "setting_id", "model", "target", "fold",
          "n_explain", "n_background", "nsamples", "k", "n_estimators",
          "sample_set", "seed"]

MAIN_TARGETS = config.TARGETS_AVAILABLE                     # 12
ROB_TARGETS = ["TRN_output", "Ca_output", "Fe_output", "Mo_output"]

# Set-B explained-sample seed (set A == RANDOM_STATE). Any seed != RANDOM_STATE
# yields an independent random explained set; this one is fixed for repeatability.
SEED_SET_B = 20260618

NEST = config.TABPFN_SHAP_N_ESTIMATORS                      # 32
NBG = config.SHAP_N_BACKGROUND                              # 80
NE_MAIN = config.SHAP_N_EXPLAIN                             # 250
NE_ROB = config.SHAP_ROBUSTNESS_N_EXPLAIN                   # 150


def _row(rid, phase, setting_id, model, target, *, fold=0, n_explain,
         nsamples, k, sample_set="A", seed=config.RANDOM_STATE):
    return {
        "row_id": rid, "phase": phase, "setting_id": setting_id,
        "model": model, "target": target, "fold": fold,
        "n_explain": n_explain, "n_background": NBG, "nsamples": nsamples,
        "k": k, "n_estimators": NEST, "sample_set": sample_set, "seed": seed,
    }


def _main_rows(model):
    return [
        _row(i, "main", "main", model, t,
             n_explain=NE_MAIN, nsamples=2048, k=5)
        for i, t in enumerate(MAIN_TARGETS)
    ]


def _rob_rows(model):
    """7 settings per target, one-axis-at-a-time from the rob baseline."""
    rows, rid = [], 0
    for t in ROB_TARGETS:
        cells = [
            # baseline: 150 / 2048 / K5 / fold0 / setA  (common reference)
            ("rob_base",     dict(fold=0, n_explain=NE_ROB, nsamples=2048, k=5)),
            # coalition axis
            ("rob_coal1024", dict(fold=0, n_explain=NE_ROB, nsamples=1024, k=5)),
            # background-K axis
            ("rob_k10",      dict(fold=0, n_explain=NE_ROB, nsamples=2048, k=10)),
            ("rob_k20",      dict(fold=0, n_explain=NE_ROB, nsamples=2048, k=20)),
            # explained-set axis
            ("rob_setB",     dict(fold=0, n_explain=NE_ROB, nsamples=2048, k=5,
                                  sample_set="B", seed=SEED_SET_B)),
            # fold axis
            ("rob_fold1",    dict(fold=1, n_explain=NE_ROB, nsamples=2048, k=5)),
            ("rob_fold2",    dict(fold=2, n_explain=NE_ROB, nsamples=2048, k=5)),
        ]
        for setting_id, kw in cells:
            rows.append(_row(rid, "robustness", setting_id, model, t, **kw))
            rid += 1
    return rows


def _smoke_rows():
    """2 tiny rows (both models, one target) for a fast end-to-end check.
    Trivial settings -> runs in minutes but exercises the full code path:
    data load, fold split, AutoGluon model load, TabPFN fit, KernelSHAP,
    CSV write, and the resume marker. Output isolated under shap_runs/smoke/."""
    t = "TRN_output"
    common = dict(fold=0, n_explain=8, nsamples=32, k=2)
    rows = [_row(0, "smoke", "smoke", "TabPFN", t, **common),
            _row(1, "smoke", "smoke", "AutoGluon", t, **common)]
    # tiny background + ensemble so the smoke test is seconds, not minutes
    for r in rows:
        r["n_background"] = 10
        r["n_estimators"] = 2
    return rows


def _write(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:18s} {len(rows):3d} rows  (array 0-{len(rows) - 1})")
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing manifests to {OUT_DIR}/")
    _write("main_tabpfn.csv", _main_rows("TabPFN"))
    _write("main_ag.csv", _main_rows("AutoGluon"))
    _write("rob_tabpfn.csv", _rob_rows("TabPFN"))
    _write("rob_ag.csv", _rob_rows("AutoGluon"))
    _write("smoke.csv", _smoke_rows())
    print("Done. Use the array range printed above in each .sbatch (--array).")


if __name__ == "__main__":
    main()
