# Predictive ionomics pipeline

Code for **"Designed nutrient-gradient phenotyping enables predictive ionomics of
tropical leafy greens"** (Shenhar *et al.*). It trains and compares two tabular
learners — AutoGluon 1.5 (stacked AutoML) and TabPFN v3 (tabular foundation model) —
to predict 12 leaf-mineral concentrations from fertigation, plant, physiological and
microclimate features, under a cross-validation design that never splits a dosing
condition between training and test. It also produces the Kernel-SHAP attribution
analysis with its robustness checks, and the external validation on a commercial cohort.

**Scope.** The repository contains exactly the code that produced the article's
figures and tables listed below — nothing else. Analyses that were run during the
project but are not reported (reference baselines, alternative imputation strategies
for the commercial cohort, calibration) have been removed from the published code.
The article's numbers were produced on the Hebrew University HURCS cluster (Slurm);
every stage is a Slurm job.

---

## What each article figure / table comes from

All paths are inside the run directory the pipeline creates.

| article | file | produced by |
|---|---|---|
| Fig. 3 cross-validated prediction | `plots/Actual_vs_Predicted_grid.pdf` | `plotting.plot_actual_vs_predicted` via `slurm/plots.sbatch` |
| Fig. 4 learning curves (+ paired t-tests) | `plots/Learning_Curves_grid.pdf` | `plotting.plot_learning_curves` via `slurm/plots.sbatch` |
| Fig. 5 SHAP, N + macronutrients | `plots/SHAP_Beeswarm_Macronutrients_paired.pdf` | `shap_module.render_article_figures` via `slurm/make_figures.sbatch` |
| Fig. 6 SHAP, micronutrients | `plots/SHAP_Beeswarm_Micronutrients_paired.pdf` | same |
| Fig. 7 cross-model attribution agreement | `plots/SHAP_Model_Comparison.pdf` | same |
| Fig. 8 commercial validation (bias insets) | `plots/fig2_strategy_A.pdf` | `farmer_plots.fig2_single_strategy` via `slurm/farmer.sbatch` |
| Fig. S6 DOB-SCV partition | `fig_S6_dobscv.pdf` | `make_dobscv_figure.py` |
| Table S3–S4 AutoGluon configuration / architecture | `architecture/autogluon_leaderboard.csv` | `training.py` |
| Table S5 TabPFN configuration | `config.py` + the kwargs in `training.py` | — |
| Table S6 SHAP robustness | `shap_runs/robustness_rank_correlations.csv` | `analyze_robustness.py` via `slurm/submit_robustness.sh` |
| §2.5.2 leakage / coverage / dose-balance statistics | stdout | `verify_cv.py` |
| §2.5.6 8- vs 32-estimator stability | `plots/shap_estimator_check_values.csv` | `shap_estimator_check.py` via `slurm/submit_estimator_check.sh` |

Figs. 1, 2 and S1–S5 and Tables 1, 2, S1, S2 describe the experimental platform and
are not produced by this code.

---

## Repository layout

```
config.py              constants: features, targets, CV settings, model budgets, SHAP settings
data.py                dataset loading; build_folds() — DOB-SCV (headline), StratifiedGroupKFold, stratified, LOEO
training.py            per-(target, fold) training of AutoGluon + TabPFN; learning curves; leaderboard
plotting.py            Fig. 3 and Fig. 4
shap_module.py         Kernel SHAP for both learners; Figs. 5, 6, 7
run.py                 single-process driver (train | plot | shap) for development
verify_cv.py           leakage / experiment-coverage / dose-balance table for a CV scheme

run_train_cell.py      one (target[, fold]) shard — what each Slurm array task executes
run_shap_cell.py       one SHAP cell (target × model × setting)
merge_train_shards.py  merges shard CSVs into the run dir, relocates models, QC gate
make_manifests.py      SHAP cell manifests (main + robustness)
analyze_robustness.py  Table S6
shap_estimator_check.py  8- vs 32-member TabPFN SHAP stability
make_dobscv_figure.py  Fig. S6

farmer_data.py         commercial-cohort loading; MISSING_FEATURES = [CT, SDW]
farmer_strategies.py   Strategy A: retrain both learners on the 15 available features
validate_farmer.py     runs Strategy A, writes farmer_predictions.csv + Fig. 8
farmer_plots.py        Fig. 8
regen_farmer_fig2.py   re-draw Fig. 8 from saved predictions without recomputing

slurm/                 sbatch files + submit_*.sh orchestrators (see Reproduction)
requirements-cluster.txt   pinned environment (Python 3.11, CUDA 12.4 build of torch)
data/README.md         dataset schema — the data itself is not distributed
```

---

## Cross-validation design

`Strat_Key = Experiment_ID × active-fertilizer gradient` identifies a **dosing
condition** (125 conditions, ~10 plants each). The headline scheme, **DOB-SCV**
(`data.build_folds(scheme="dobscv")`), treats each condition as indivisible and assigns
conditions round-robin within every (experiment × dose-tier) cell, so each of the 5 folds
holds out whole conditions while still containing every experiment and every dose tier.
It is deterministic (no seed). `python verify_cv.py` prints the leakage (0 %), experiment
coverage (16/16 per fold) and dose-balance (χ²) table for each scheme.

`scheme="grouped"` (StratifiedGroupKFold, seed 42) is the alternative leakage-free scheme
the article reports as a sensitivity check; `"stratified"` reproduces the leaky design the
study replaced.

---

## Reproduction on a Slurm cluster

1. Copy the repository and the two datasets (see `data/README.md`) to the cluster.
2. `cp slurm/env.sh.example slurm/env.sh` and fill in the lab account, project path and
   dataset paths. `env.sh` is gitignored.
3. On a login node: `bash slurm/setup_env.sh` (venv from `requirements-cluster.txt`), then
   `bash slurm/prefetch_tabpfn.sh` (downloads the TabPFN v3 weights).

Then, from the project directory:

```
bash slurm/submit_fast.sh both        # training: DOB-SCV then StratifiedGroupKFold
                                      #   60 CPU tasks (AutoGluon, 32 cores each)
                                      #   12 GPU tasks (TabPFN, L40S, 8 cores each)
                                      #   -> merge -> Figs. 3-4 -> Fig. 8, chained by dependency
bash slurm/submit_shap.sh both        # Kernel SHAP, both learners -> Figs. 5-7
RUN_DIR=<dir> bash slurm/submit_robustness.sh      # Table S6 (needs fold 0-2 models)
bash slurm/submit_estimator_check.sh               # 8 vs 32 estimators
python verify_cv.py                                # §2.5.2 partition statistics
python make_dobscv_figure.py                       # Fig. S6
```

`submit_fast.sh` records the run directory it created in `slurm/CURRENT_RUN_DIR_<scheme>`;
later stages read it from there or from `RUN_DIR`. Every sbatch refuses to run without
`RUN_DIR` set, so a bare `sbatch` cannot silently overwrite a previous run.

**Cost.** The article's analyses (both CV schemes, SHAP, external validation) used
≈13,900 CPU core-hours and 14.6 GPU-hours on HURCS — about US$115 and ~2 days
wall-clock. AutoGluon is >95 % of it. One training scheme via `submit_fast.sh` is ~3.5 h
wall-clock.

**Local runs.** `run.py` runs everything in one process for development
(`python run.py --stage train --cv-scheme dobscv --targets TRN_output`), but the full
sweep is impractical off-cluster, and TabPFN SHAP on Apple MPS measured ~48× slower
than on an L40S.

---

## Things to know before trusting an output

- **Identical filenames across CV schemes.** A DOB-SCV run and a StratifiedGroupKFold run
  each write `plots/SHAP_Model_Comparison.pdf` etc. Only the `*_dobscv_*` run is the
  article. Check the run-directory name before using a figure.
- **Determinism.** DOB-SCV folds and the training-subsample draw (`RANDOM_STATE=42`) are
  deterministic; AutoGluon's internal training is not seeded, so its metrics reproduce to
  within fold-to-fold noise, not to the digit. TabPFN is seeded.
- **`PAR_SUM`** in the code is **`PPFD_SUM`** in the article.
- **Budgets.** `AG_TIME_LIMIT_FULL=3600` s per full-size fit, `AG_TIME_LIMIT_SUB=900` s at
  learning-curve fractions. Overridable by environment variable for smoke tests; the
  article's numbers used these values.

---

## Provenance

The article's results were produced by a cluster snapshot of this project on
2026-07-26/27 (training, SHAP, external validation) with the robustness and
estimator-stability checks run afterwards against the archived models. The published
repository is that code with the unreported analyses removed:

- **Byte-identical to the as-run code:** `data.py`, `farmer_data.py`, `run.py`,
  `run_shap_cell.py`, `make_manifests.py`, `analyze_robustness.py`,
  `shap_estimator_check.py`, `verify_cv.py`, `make_dobscv_figure.py`, and the Slurm
  files for training-GPU, merge, plots, SHAP, robustness and the estimator check.
- **Deletions only** (unreported branches removed; the reported logic is untouched):
  `training.py` (reference-baseline branch, permutation importance), `config.py`
  (baseline / Optuna constants), `plotting.py` (three non-article plots),
  `shap_module.py` (per-model beeswarm grids), `merge_train_shards.py`,
  `run_train_cell.py`, `slurm/train_cpu.sbatch`, `slurm/submit_fast.sh` (calibration
  step), `slurm/make_figures.sbatch`.
- **Reduced to Strategy A only:** `farmer_strategies.py`, `farmer_plots.py`,
  `validate_farmer.py`, `regen_farmer_fig2.py`, `slurm/farmer.sbatch`.

Figs. 3–8 rendered by this code were verified pixel-identical to those rendered by the
as-run code from the same run directory.

---

## Environment

`requirements-cluster.txt` pins the versions quoted in the article: Python 3.11,
AutoGluon 1.5.0, tabpfn 8.0.3, scikit-learn 1.7.2, shap 0.51.0, PyTorch 2.6.0+cu124.
TabPFN v3 weights are downloaded on first use; run `slurm/prefetch_tabpfn.sh` on a login
node with internet access before submitting jobs.

## License and attribution

This code is released under the [MIT License](LICENSE). The libraries it depends on
keep their own licences: AutoGluon (Apache 2.0), scikit-learn and PyTorch
(BSD-3-Clause), shap (MIT). TabPFN is distributed by Prior Labs under a modified
Apache 2.0 licence with an attribution clause — **Built with PriorLabs-TabPFN** — and its
model weights are downloaded separately on first use under those terms.

## Citation

See [CITATION.cff](CITATION.cff).
