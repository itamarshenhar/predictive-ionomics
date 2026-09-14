# Data

The two datasets are **not distributed with the code**. They are available from the
corresponding author upon reasonable request (see the article's Data availability
statement). Place them in this directory, or point `DATA_PATH` / `FARMER_PATH` at them.

| file | rows | used by |
|---|---|---|
| `Min_Bal_DB_24.05.26.xlsx` | 1,163 plants, 16 experiments, 125 dosing conditions | everything |
| `Contro_Table_DB_27.05.26_rounded.xlsx` | 99 plants from a grower-operated commercial site | `validate_farmer.py` |

## Expected columns

Both files are read from the first sheet with `pandas.read_excel`.

**Identifiers / grouping** — `Experiment_ID`, plus one fertigation-treatment column per
experiment named in `config.ACTIVE_TREATMENT_BY_EXP`. `data.build_folds` combines them
into `Strat_Key = Experiment_ID × active-fertilizer gradient`, the dosing-condition
group that cross-validation never splits.

**Input features** (`config.FEATURES`, 17):

```
N_input P_input K_input Ca_input Mg_input Fe_input Mo_input   # fertigation inputs
Plant_type Substrate                                           # categorical
CT SDW                                                         # cumulative transpiration, shoot dry weight
J_day DAT                                                      # Julian day, days after transplanting
PAR_SUM VPD_SUM Temp_SUM RH_SUM                                # cumulative microclimate integrals
```

`PAR_SUM` is displayed as `PPFD_SUM` in the article (`shap_module.FEATURE_DISPLAY`).

**Targets** (`config.TARGETS`, 12) — leaf mineral concentrations, mg g⁻¹ DW, columns
`<element>_output`: `NO3 TRN P K Ca Mg Fe Mn Zn Bo Mo Na`.

**Commercial cohort** — same columns except `CT` and `SDW`, which are absent for every
plant (`farmer_data.MISSING_FEATURES`); the remaining 15 features are complete. All
seven fertigation inputs and `Substrate` are constant across the cohort (one recipe),
which is why the article reports multiplicative bias rather than R² for it.
