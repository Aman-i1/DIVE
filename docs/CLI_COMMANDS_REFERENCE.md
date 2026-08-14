# DIVE CLI: Comprehensive Command Reference & Manual

This reference manual documents every single command, subcommand, parameter, option, and workflow available in the **DIVE** (Data Intelligence, Validation & Ensembling) CLI.

---

## Command Index & Overview

| Command | Category | Purpose | Typical Run Time |
|---|---|---|---|
| [`dive autopilot`](#1-dive-autopilot) | Orchestration | Full 20-step Senior ML Review, Reliability & AutoML workflow | 5m - 30m |
| [`dive auto`](#2-dive-auto) | Orchestration | Declarative or autonomous multi-model industrial pipeline execution | 1m - 15m |
| [`dive train`](#3-dive-train) | Training | Multi-model zoo search, tuning, calibrated stacking, and reporting | 30s - 30m |
| [`dive predict`](#4-dive-predict) | Inference | Batch inference or interactive terminal prediction machine | < 5s |
| [`dive contract`](#5-dive-contract) | Reliability | Formal Prediction Contract inference and JSON specification | < 2s |
| [`dive review`](#6-dive-review) | Reliability | Senior ML Practitioner automated review and deployment gating | < 10s |
| [`dive gate`](#7-dive-gate) | Production | Deployment gatekeeper auditing drift, leakage, and schema safety | < 5s |
| [`dive doctor`](#8-dive-doctor) | Diagnostics | Pre-flight audit of dataset readiness and 0-100 score | < 5s |
| [`dive validate`](#9-dive-validate) | Validation | Pre-training data health, leakage, and validation strategy check | < 5s |
| [`dive serve`](#10-dive-serve) | Production | Production REST API server with interactive `/help` endpoint | Daemon |
| [`dive drift`](#11-dive-drift) | Observability | PSI and Kolmogorov-Smirnov drift analysis on production data | < 5s |
| [`dive explain`](#12-dive-explain) | Explainability | Feature importance, permutation attributions, and SHAP plots | < 10s |
| [`dive report`](#13-dive-report) | Reporting | Interactive web HTML and LaTeX research paper PDF generation | < 10s |
| [`dive reproduce`](#14-dive-reproduce) | Reproducibility | Content-addressable SHA-256 bundle and standalone `reproduce.py` | < 5s |
| [`dive audit`](#15-dive-audit) | Compliance | Cryptographically verifiable compliance and governance certificates | < 5s |
| [`dive export`](#16-dive-export) | Export | Export models to ONNX, PMML, TorchScript, or Rust C-FFI / WASM | < 5s |
| [`dive info`](#17-dive-info) | Inspection | Deep dataset inspection (types, missingness, memory, target) | < 2s |
| [`dive benchmark`](#18-dive-benchmark) | Benchmarking | Hardware latency, RAM consumption, and prediction throughput | < 5s |
| [`dive deps`](#19-dive-deps) | Environment | Hardware and library environment audit (CUDA, LightGBM, etc.) | < 1s |
| [`dive docs`](#20-dive-docs) | Documentation | Offline documentation server and browser viewer | Daemon |
| [`dive upgrade`](#21-dive-upgrade) | Maintenance | Automated package upgrade and migration helper | < 10s |

---

## 1. `dive autopilot`

### Overview
`dive autopilot` coordinates the end-to-end **20-Step Senior ML Review + Reliability + Production-Gating Platform**. It executes:
1. Formal Prediction Contract inference (entity, timestamp, prediction horizon, allowed information cutoff).
2. Comprehensive Data Quality audit and statistical Inferred Relational Rules discovery (`[INFERRED RULE]`).
3. Cross-partition contamination detection (exact duplicates, entity CV leakage, label conflicts).
4. Adversarial Validation (evaluating train/test covariate shift via discriminator AUC).
5. Model search across LightGBM, XGBoost, CatBoost, HistGradientBoosting, RandomForest, ExtraTrees, and MLP.
6. Model Stress Testing (shuffled-target permutation sanity tests, seed bootstrap stability, and feature reliance).
7. Weak Population Failure Segmentation discovery.
8. Senior ML Review synthesis (`PASS`, `PASS_WITH_WARNINGS`, or `BLOCKED` verdicts).
9. One-click reproducibility bundle export.

### Syntax
```bash
dive autopilot [OPTIONS] DATA_PATH
```

### Options & Arguments
- `DATA_PATH` *(Required)*: Path to the input tabular dataset (`.csv`, `.tsv`, `.parquet`, `.feather`, `.jsonl`, `.xlsx`, `.orc`, `.h5`, `.dta`).
- `--target, -t TEXT` *(Required)*: Name of the target column to predict.
- `--mode, -m [fast|balanced|competition]` *(Default: `balanced`)*: Search depth and model capacity.
  - `fast`: Small model subset for rapid smoke testing (30s - 2m).
  - `balanced`: Full model zoo with hyperparameter search (5m - 10m).
  - `competition`: Extensive multi-fidelity ASHA search + Calibrated Stacking (10m - 30m).
- `--budget, -b TEXT` *(Default: `10m`)*: Total compute time budget (e.g. `60s`, `5m`, `30m`, `2h`).
- `--entity, -e TEXT` *(Optional)*: Column identifying the grouping entity (e.g. `customer_id`, `user_id`).
- `--time-column, --time TEXT` *(Optional)*: Column containing event timestamps for temporal splitting.
- `--horizon TEXT` *(Optional)*: Prediction horizon (e.g. `30d`, `7d`, `24h`).
- `--output, -o PATH` *(Default: `./dive_autopilot_out`)*: Directory to save all review reports, models, and artifacts.

### Example
```bash
dive autopilot customer_churn.csv --target churn --entity customer_id --time-column signup_date --mode balanced --budget 10m --output ./churn_autopilot
```

---

## 2. `dive auto`

### Overview
`dive auto` is the declarative and autonomous AutoML command. It can run purely from CLI arguments or execute from a declarative YAML / JSON configuration file.

### Syntax
```bash
dive auto [OPTIONS] [DATA_PATH]
```

### Options & Arguments
- `DATA_PATH` *(Optional if `--config` provided)*: Path to dataset.
- `--config, -c PATH` *(Optional)*: Path to a YAML or JSON configuration file (e.g. `dive_config.yaml`).
- `--target, -t TEXT`: Target column name.
- `--mode, -m [fast|balanced|competition]` *(Default: `balanced`)*: Optimization depth.
- `--budget, -b TEXT` *(Default: `10m`)*: Execution time budget (e.g. `600s`, `10m`).
- `--output, -o PATH` *(Default: `./dive_output`)*: Artifact output directory.

### Example with YAML Config
Create `dive_config.yaml`:
```yaml
data:
  path: "telecom.csv"
  target: "churn"
  entity_column: "customer_id"
study:
  mode: "balanced"
  time_budget: 300
  metric: "roc_auc"
output:
  directory: "./telecom_run"
```
Run:
```bash
dive auto --config dive_config.yaml
```

---

## 3. `dive train`

### Overview
`dive train` performs tabular model training across the candidate model zoo, cross-validation, hyperparameter tuning, probability calibration (Platt scaling / Isotonic regression), multi-layer stacking, and automated report generation.

### Syntax
```bash
dive train [OPTIONS] DATA_PATH
```

### Options & Arguments
- `DATA_PATH` *(Required)*: Path to dataset.
- `--target, -t TEXT` *(Required)*: Column to predict.
- `--mode, -m [fast|balanced|competition]` *(Default: `balanced`)*: Training profile.
- `--time-budget, -b INTEGER` *(Default: `1800`)*: Time budget in seconds.
- `--group-column, -g TEXT` *(Optional)*: Column for group-aware cross validation.
- `--time-column, --time TEXT` *(Optional)*: Column for temporal time-series splitting.
- `--test-size FLOAT` *(Default: `0.2`)*: Fraction of dataset reserved for holdout validation (e.g. `0.2` = 20%).
- `--random-state INTEGER` *(Default: `42`)*: Reproducible random seed.
- `--output, -o PATH` *(Default: `./dive_output`)*: Output directory for `model.pkl`, `leaderboard.csv`, `report.html`, and `report.pdf`.

### Example
```bash
dive train dataset.csv --target is_fraud --mode competition --time-budget 600 --output ./fraud_model
```

---

## 4. `dive predict`

### Overview
Scores unlabelled or new records using a trained DIVE model. Supports batch CSV/Parquet processing as well as an interactive single-row terminal "prediction machine" with probability confidence distributions.

### Syntax
```bash
dive predict [OPTIONS] [MODEL_PATH] [DATA_PATH]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to `model.pkl` or any standalone predictor in `models/*.pkl`.
- `--data, -d PATH` *(Optional)*: Path to dataset to score. If omitted, launches the interactive terminal prediction machine.
- `--output, -o PATH` *(Default: `predictions.csv`)*: Path to write scored predictions.
- `--proba, -p` *(Flag)*: Include class probability distributions (e.g. `prob_0`, `prob_1`).
- `--include-input, -i` *(Flag)*: Prepend original input columns to the output CSV.

### Examples
```bash
# 1. Batch scoring with probabilities
dive predict --model ./fraud_model/model.pkl --data new_transactions.csv --output ./scores.csv --proba

# 2. Interactive terminal prediction machine
dive predict --model ./fraud_model/model.pkl
```

---

## 5. `dive contract`

### Overview
Infers or verifies the **Formal Prediction Contract** for a dataset. Declares the target, problem type, entity grouping, prediction timestamp, prediction horizon, and allowed information cutoff. Flags `UNKNOWN` when context cannot be inferred statistically.

### Syntax
```bash
dive contract [OPTIONS] DATA_PATH
```

### Options & Arguments
- `DATA_PATH` *(Required)*: Input dataset.
- `--target, -t TEXT` *(Required)*: Target column name.
- `--entity, -e TEXT` *(Optional)*: Entity / grouping identifier column.
- `--time-column, --time TEXT` *(Optional)*: Event timestamp column.
- `--horizon TEXT` *(Optional)*: Prediction horizon (e.g. `30d`).
- `--output, -o PATH` *(Default: `./prediction_contract.json`)*: Output JSON path.

### Example
```bash
dive contract transactions.csv --target is_fraud --entity account_id --output ./fraud_contract.json
```

---

## 6. `dive review`

### Overview
Executes an automated **Senior ML Practitioner Review** auditing data quality, relational constraints, leakage, covariate shift, permutation sanity, and slice weaknesses, generating a consolidated matrix with explicit required action items.

### Syntax
```bash
dive review [OPTIONS] DATA_PATH
```

### Options & Arguments
- `DATA_PATH` *(Required)*: Input dataset.
- `--target, -t TEXT` *(Required)*: Target column name.
- `--entity, -e TEXT` *(Optional)*: Entity column name.
- `--time-column, --time TEXT` *(Optional)*: Time column name.
- `--output, -o PATH` *(Default: `./senior_review.json`)*: Output JSON path.

### Example
```bash
dive review customer_data.csv --target churn --entity cust_id --output ./churn_review.json
```

---

## 7. `dive gate`

### Overview
The **Production Deployment Gatekeeper**. Evaluates a candidate model against production batches or reference datasets, asserting schema compatibility, target leakage safety, and distribution drift thresholds before allowing CI/CD deployment.

### Syntax
```bash
dive gate [OPTIONS] MODEL_PATH
```

### Options & Arguments
- `MODEL_PATH` *(Required)*: Path to trained `model.pkl`.
- `--data, -d PATH` *(Required)*: Incoming production dataset batch to verify.
- `--ref, -r PATH` *(Optional)*: Baseline training dataset for comparative drift checking.
- `--strict` *(Flag)*: Fail with exit code 1 if any warnings are detected.

### Example
```bash
dive gate ./model.pkl --data ./current_batch.csv --ref ./training_data.csv --strict
```

---

## 8. `dive doctor`

### Overview
Runs a diagnostic pre-flight audit on dataset readiness, missingness patterns, duplicate row leakage, high cardinality, temporal ordering, and calculates the **DIVE Production Readiness Score (0 - 100)**.

### Syntax
```bash
dive doctor [OPTIONS] DATA_PATH
```

### Options & Arguments
- `DATA_PATH` *(Required)*: Dataset to inspect.
- `--target, -t TEXT` *(Required)*: Target column.
- `--group-column, -g TEXT` *(Optional)*: Grouping column.
- `--time-column, --time TEXT` *(Optional)*: Time column.

### Example
```bash
dive doctor loan_applications.csv --target default --group-column applicant_id
```

---

## 9. `dive validate`

### Overview
Performs comprehensive pre-training data integrity validation, checking target health, column data types, missing value percentages, and multi-collinearity.

### Syntax
```bash
dive validate [OPTIONS] DATA_PATH
```

### Options & Arguments
- `DATA_PATH` *(Required)*: Dataset to validate.
- `--target, -t TEXT` *(Required)*: Target column name.
- `--group-column, -g TEXT` *(Optional)*: Entity group column.
- `--time-column, --time TEXT` *(Optional)*: Temporal column.

### Example
```bash
dive validate credit_risk.csv --target default
```

---

## 10. `dive serve`

### Overview
Spawns a production HTTP REST API server exposing endpoints for real-time scoring, class probabilities, input schema, health probes, server latency metrics, and interactive documentation via `/help`.

### Syntax
```bash
dive serve [OPTIONS]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to `model.pkl` or standalone predictor pickle.
- `--host TEXT` *(Default: `127.0.0.1`)*: Network interface to bind.
- `--port INTEGER` *(Default: `8000`)*: Port number to listen on.

### REST Endpoints
- `GET /help` or `GET /`: Interactive API user guide, curl examples, and JSON payload templates.
- `GET /health`: Health and liveness probe.
- `GET /metadata`: Training metadata and model performance metrics.
- `GET /schema`: Formal expected feature names, types, and ranges.
- `POST /predict`: Generate point predictions for `{"data": [ { ... } ]}`.
- `POST /predict_proba`: Generate calibrated probabilities for classification tasks.
- `GET /metrics`: Request volume, error rate, and p50/p95/p99 latency metrics.

### Example
```bash
dive serve --model ./higgs_max_quality/model.pkl --port 8000
```

---

## 11. `dive drift`

### Overview
Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) two-sample statistical tests between a baseline reference dataset and a current production batch.

### Syntax
```bash
dive drift [OPTIONS]
```

### Options & Arguments
- `--ref, -r PATH` *(Required)*: Path to reference baseline dataset (e.g. training set).
- `--curr, -c PATH` *(Required)*: Path to current production dataset.
- `--target, -t TEXT` *(Optional)*: Target column name.
- `--threshold FLOAT` *(Default: `0.2`)*: PSI threshold triggering significant drift alerts.
- `--output, -o PATH` *(Optional)*: Output JSON path for drift metrics.

### Example
```bash
dive drift --ref train_jan.csv --curr prod_feb.csv --threshold 0.25 --output ./drift_report.json
```

---

## 12. `dive explain`

### Overview
Computes and visualizes global and local model explanations: Permutation Feature Importance, SHAP summary distributions, and partial dependence plots.

### Syntax
```bash
dive explain [OPTIONS]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to trained `model.pkl`.
- `--data, -d PATH` *(Required)*: Dataset to explain.
- `--output, -o PATH` *(Default: `./explanations`)*: Output directory for importance plots and CSVs.

### Example
```bash
dive explain --model ./model.pkl --data ./test_sample.csv --output ./shap_plots
```

---

## 13. `dive report`

### Overview
Generates a standalone, self-contained interactive HTML diagnostic dashboard and a publication-quality LaTeX research paper PDF report from an existing training run.

### Syntax
```bash
dive report [OPTIONS]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to trained `model.pkl`.
- `--output, -o PATH` *(Default: `./report`)*: Directory to save `report.html` and `report.pdf`.

### Example
```bash
dive report --model ./dive_output/model.pkl --output ./executive_report
```

---

## 14. `dive reproduce`

### Overview
Generates an isolated, zero-dependency `reproduce.py` script, environment lockfile, and content-addressable SHA-256 artifacts enabling 100% bitwise exact reproduction of a model run.

### Syntax
```bash
dive reproduce [OPTIONS]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to `model.pkl`.
- `--output, -o PATH` *(Default: `./reproduce_bundle`)*: Bundle directory.

### Example
```bash
dive reproduce --model ./dive_output/model.pkl --output ./reproduction_pkg
```

---

## 15. `dive audit`

### Overview
Audits model lineage, data transformations, security posture, and validation checks, generating a signed SHA-256 cryptographically verifiable compliance certificate.

### Syntax
```bash
dive audit [OPTIONS]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to `model.pkl`.
- `--output, -o PATH` *(Default: `./audit_certificate.json`)*: Output certificate path.

### Example
```bash
dive audit --model ./model.pkl --output ./audit_cert.json
```

---

## 16. `dive export`

### Overview
Exports trained DIVE models to portable formats for embedded, high-throughput, or non-Python deployment environments (ONNX, PMML, TorchScript, WebAssembly, Rust C-FFI).

### Syntax
```bash
dive export [OPTIONS]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to `model.pkl`.
- `--format, -f [onnx|pmml|torchscript|cffi|wasm]` *(Default: `onnx`)*: Target export format.
- `--output, -o PATH` *(Required)*: Output file path.

### Example
```bash
dive export --model ./model.pkl --format onnx --output ./model.onnx
```

---

## 17. `dive info`

### Overview
Performs deep structural and statistical inspection of any tabular dataset, reporting row/column counts, memory footprint, fine-grained semantic column types, missingness, and target label distributions.

### Syntax
```bash
dive info [OPTIONS] DATA_PATH
```

### Options & Arguments
- `DATA_PATH` *(Required)*: Dataset to inspect.
- `--target, -t TEXT` *(Optional)*: Target column for class distribution analysis.

### Example
```bash
dive info transactions.parquet --target is_fraud
```

---

## 18. `dive benchmark`

### Overview
Runs latency, CPU/GPU utilization, RAM allocation, and prediction throughput benchmarks on a trained model pipeline.

### Syntax
```bash
dive benchmark [OPTIONS]
```

### Options & Arguments
- `--model, -m PATH` *(Required)*: Path to `model.pkl`.
- `--data, -d PATH` *(Required)*: Dataset to benchmark on.
- `--batch-sizes TEXT` *(Default: `1,10,100,1000`)*: Comma-separated batch sizes to test.

### Example
```bash
dive benchmark --model ./model.pkl --data ./test.csv --batch-sizes 1,16,64,256
```

---

## 19. `dive deps`

### Overview
Inspects the local runtime environment and reports optional hardware acceleration (NVIDIA CUDA GPU) and library statuses (`xgboost`, `lightgbm`, `catboost`, `torch`, `shap`, `fastapi`, `uvicorn`).

### Syntax
```bash
dive deps
```

### Example
```bash
dive deps
```

---

## 20. `dive docs`

### Overview
Launches a local offline documentation server in your default web browser containing the complete user manual, guides, and API specifications.

### Syntax
```bash
dive docs [OPTIONS]
```

### Options & Arguments
- `--port INTEGER` *(Default: `8080`)*: Port to serve documentation on.
- `--no-browser` *(Flag)*: Do not automatically launch the web browser.

### Example
```bash
dive docs --port 8080
```

---

## 21. `dive upgrade`

### Overview
Checks for newer versions of DIVE on PyPI and executes an automated upgrade migration.

### Syntax
```bash
dive upgrade
```
