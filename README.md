# DIVE

Automated machine learning for tabular data, driven from the command line.

Point it at a CSV, name the column you want to predict, and it profiles the
data, engineers features, trains a zoo of models, tunes the leaders, stacks
them, and writes a self-contained HTML report — plus the crosschecks that catch
the mistakes which quietly invalidate a model.

```bash
dive validate --data data.csv --target churned      # is this data trainable?
dive train    --data data.csv --target churned --output ./out
dive predict  --model ./out/model.pkl --data new.csv --output scored.csv
```

---

## Install

```bash
git clone https://github.com/Aman-i1/DIVE.git
cd DIVE
pip install -e .
```

That puts an `dive` command on your PATH — identical on Windows, macOS, and
Linux. Verify with `dive --help`.

### Optional extras

The tool runs on scikit-learn alone. Extras widen the model zoo and enable
tuning; anything missing is skipped with a note saying what was lost and how to
get it.

```bash
pip install -e ".[full]"        # everything below
pip install -e ".[boosters]"    # xgboost, lightgbm, catboost
pip install -e ".[tuning]"      # optuna
pip install -e ".[explain]"     # category_encoders, shap
```

Run `dive deps` to see what is active.

---

## Google Colab

No local setup. Paste into a cell:

```python
!git clone https://github.com/Aman-i1/DIVE.git
%cd DIVE
!pip install -q -e ".[full]"

!dive train --data examples/sample.csv --target diagnosis \
    --mode fast --output /content/out

from IPython.display import HTML
HTML(open('/content/out/report.html').read())
```

The report embeds its plots as base64, so it renders inline and can be
downloaded as a single file. A complete walkthrough lives in
[`examples/colab_quickstart.ipynb`](examples/colab_quickstart.ipynb).

---

## Commands

| Command | Purpose |
|---|---|
| `dive train` | Train the zoo; write model, leaderboard, plots, and report |
| `dive predict` | Score new rows with a saved model, with a schema check first |
| `dive validate` | Run the crosschecks with no training |
| `dive explain` | Plain-English pipeline account + standalone reproduction code |
| `dive report` | Render the full HTML report for a saved model |
| `dive docs` | Open or serve the HTML documentation |
| `dive deps` | Show which optional packages are installed |

`--help` and `--version` work everywhere. Exit codes: `0` success, `1` user or
data error, `2` unexpected internal error.

### train

```bash
dive train --data sales.csv --target revenue \
    --mode balanced --time-budget 900 --output ./out
```

| Mode | Models | Tuning | Stacking | Use when |
|---|---|---|---|---|
| `fast` | 3 small | no | no | First look, very large data, CI smoke tests |
| `balanced` | full zoo | yes | yes | The default |
| `competition` | full zoo + AdaBoost | yes | yes | Maximum accuracy; adds feature selection |

Common flags: `--time-budget` (seconds), `--test-size`, `--cv-folds`,
`--random-state`, `--time-series`, `--config`, `--no-plots`, `--no-report`,
`--skip-validation`.

Any flag can live in a YAML file instead:

```yaml
# settings.yaml
data: sales.csv
target: revenue
mode: balanced
time_budget: 900
output: ./out
```

```bash
dive train --config settings.yaml
```

Explicit flags override the file. An unknown key is an error, not a silent
no-op.

### predict

```bash
dive predict --model ./out/model.pkl --data new.csv \
    --output scored.csv --proba
```

The incoming schema is compared against training before anything is scored. A
missing feature column fails immediately rather than misaligning columns and
returning confident nonsense. Extra columns are ignored; columns that were
dropped during training may be absent.

### validate

```bash
dive validate --data data.csv --target churned
```

Trains nothing. Returns exit code 1 on any FAIL, so it works as a CI gate; add
`--strict` to fail on warnings too.

---

## What the crosschecks catch

| Check | Fails when |
|---|---|
| `target_health` | Near-constant target, or a class with too few rows to cross-validate |
| `target_leakage` | A single feature predicts the target at ≥0.98 association |
| `duplicate_rows` | ≥1% of holdout rows also appear in the training split |
| `train_holdout_drift` | A numeric feature is distributed differently across the split (warn) |
| `missing_data` | A column is effectively empty |
| `cv_stability` | Fold-to-fold score spread exceeds 10% of the mean (warn) |
| `predict_schema` | Incoming data is missing a required feature column |

Leakage is the one worth internalising: a column that encodes the answer
produces a model that looks excellent in testing and fails in production.
`validate` catches it in seconds.

Full explanations of each verdict: `dive docs`.

---

## Output files

Everything lands in `--output` (default `./dive_output`):

```
out/
├── model.pkl           # fitted model + feature engineer
├── leaderboard.csv     # every model and its scores, best first
├── report.html         # self-contained report, plots inlined
├── validation.json     # machine-readable crosscheck verdicts
├── metadata.json       # settings, schema, timings, skipped models
└── plots/
    ├── leaderboard.png
    ├── comparison.png
    ├── diagnostics.png
    └── feature_importance.png
```

File names are stable, so scripts and CI jobs can depend on them.

---

## Python API

```python
import pandas as pd
from dive import Dive

model = Dive(target="diagnosis", mode="balanced", time_budget=600)
model.fit(pd.read_csv("data.csv"))

print(model.leaderboard())
model.save("model.pkl")

loaded = Dive.load("model.pkl")
predictions = loaded.predict(new_frame)
```

The fitted feature engineer is pickled with the model, so `predict` never
refits — inference applies exactly the transformations learned at training time.

---

## What it does to your data

1. **Profiles** — problem type, imbalance, missing values, constant / ID-like /
   high-cardinality / datetime columns.
2. **Engineers features** — drops IDs and constants, expands dates into calendar
   parts, clips outliers to the 1st–99th percentile, groups rare categories,
   adds frequency and target encodings, downcasts to float32.
3. **Splits** — stratified for classification, chronological with
   `--time-series`.
4. **Preprocesses** — median/most-frequent imputation, standard scaling,
   one-hot encoding, variance filtering, optional PCA.
5. **Trains** — linear models, random forest, extra trees, histogram gradient
   boosting, MLP, KNN, AdaBoost, plus XGBoost / LightGBM / CatBoost when
   installed. GPU is detected automatically.
6. **Tunes** — Optuna with pruning, inside the time budget.
7. **Stacks** — out-of-fold predictions from the top models feed a meta-learner.

`dive explain` reports exactly which of these ran for your dataset.

---

## Development

```bash
pip install -e ".[full,dev]"
pytest
```

CI runs the suite on Ubuntu, Windows, and macOS across Python 3.9 and 3.12,
plus a scikit-learn-only job that verifies the tool still works with no
optional dependencies installed.

---

## License

MIT — see [LICENSE](LICENSE).
