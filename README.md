# DIVE — Tabular ML Reliability + AutoML + MLOps Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aman-i1/DIVE/blob/main/examples/colab_quickstart.ipynb)

**DIVE** is a production-oriented **Tabular ML Reliability, AutoML, and MLOps platform**.

Instead of blindly fitting models on unverified data, DIVE acts like a senior ML engineer sitting beside you: it audits dataset health, detects target leakage, identifies entity contamination and temporal dependencies to select safe validation strategies, optimizes model zoos under memory constraints, calibrates probabilities, analyzes failure segments, tracks immutable experiments, serves production REST APIs, and monitors data/prediction drift.

```bash
dive doctor data.csv --target churned              # ML-readiness diagnostic audit & readiness score
dive train  data.csv --target churned --output ./out # Resource-aware AutoML + Ensembling
dive serve  --model ./out/model.pkl                 # Deploy FastAPI REST prediction server
dive drift  --ref train.csv --curr prod.csv        # Production drift detection & retraining advice
```

---

## 🏗 Architecture & Flow

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Raw Data    │ ──> │  ML Doctor   │ ──> │ Validation Advisor  │ ──> │ Feature Engineering │
└──────────────┘     └──────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                                 │
                                                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ REST Serving │ <── │ Model        │ <── │ Explainability &    │ <── │ Resource-Aware      │
│ & Drift Mon  │     │ Registry     │     │ Failure Analysis    │     │ AutoML Zoo & Stacking│
└──────────────┘     └──────────────┘     └─────────────────────┘     └─────────────────────┘
```

---

## 🚀 Quickstart

### Installation

```bash
pip install dive-ml
# or from source:
git clone https://github.com/Aman-i1/DIVE.git
cd DIVE
pip install -e .
```

### CLI Workflow

```bash
# 1. Run ML Doctor diagnostic audit & check Production Readiness Score
dive doctor sales.csv --target churn --group-column customer_id

# 2. Train AutoML model zoo with memory limit & time budget
dive train --data sales.csv --target churn --time-budget 600 --output ./dive_out

# 3. Register model into local registry & evaluate promotion gates
dive models register ./dive_out/model.pkl --name churn_model --stage candidate
dive models promote churn_model v1 production

# 4. Deploy production FastAPI REST API server
dive serve --model ./dive_out/model.pkl --port 8000

# 5. Monitor drift against production batch data
dive drift --ref sales.csv --curr production_week1.csv

# 6. Export reproducibility manifest
dive reproduce EXP-000001
```

### Python API Workflow

```python
from dive import (
    Dive,
    DiveDoctor,
    ValidationAdvisor,
    ModelAdvisor,
    ExperimentTracker,
    ModelRegistry,
    DriftDetector,
    load_predictor,
)

# 1. Run Diagnostic ML Doctor
doctor = DiveDoctor(target="churned", group_column="customer_id")
report = doctor.analyze(df)
print(report)

# 2. Fit AutoML Pipeline
dive = Dive(target="churned", mode="balanced")
dive.fit(df)

# 3. Predict on new rows with exact fitted preprocessing
predictor = load_predictor("dive_output/model.pkl")
preds = predictor.predict(new_df)
```

---

## ⚡ CLI Commands Reference

| Command | Purpose |
|---|---|
| `dive doctor` | Full 17-point ML-readiness audit & Production Readiness Score |
| `dive train` | Resource-aware AutoML training, tuning, stacking & reporting |
| `dive predict` | Score new rows with schema validation & fitted preprocessing |
| `dive validate` | Standalone crosscheck suite (leakage, duplicates, target health) |
| `dive serve` | Serve model as a FastAPI REST prediction server |
| `dive experiments` | List, inspect, and compare tracked experiment runs |
| `dive models` | Local model registry catalog & automated promotion gates |
| `dive drift` | Detect PSI, KS, and JS divergence data & prediction drift |
| `dive reproduce` | Export reproducibility manifest bundles (`experiment.json`, `environment.json`) |
| `dive benchmark` | Run DIVE dataset scaling & performance benchmark suite |
| `dive upgrade` | Auto-upgrade DIVE to latest GitHub release & update dependencies |
| `dive explain` | Pipeline account, local feature contributions & counterfactuals |

| `dive report` | Render standalone HTML diagnostic report |
| `dive docs` | Open or serve local HTML documentation |
| `dive deps` | Show optional dependency status |

---

## 🤔 Why DIVE Exists

Traditional AutoML tools focus exclusively on benchmark accuracy — often training models on datasets contaminated by:
1. **Target Leakage**: Features that encode the outcome after the prediction event.
2. **Unsafe Cross-Validation**: Random K-Fold splits on repeated entity data that leak customer/patient identities across folds.
3. **Distribution Shift**: Silent performance collapse when production data shifts away from training distributions.
4. **Black-Box Decisions**: Obscure model choices without explainable justifications.

**DIVE** solves this by enforcing ML Reliability as a first-class citizen: every automatic decision is explainable, every dataset is audited, validation strategy is matched to entity/temporal structure, and models are served with schema enforcement and drift monitoring.

---

## ⚠️ Limitations

- **Tabular Focus**: Designed specifically for tabular data (numeric, categorical, datetime, text metadata). Not designed for raw image or audio tensor data.
- **In-Memory Scale**: Optimized for single-node machines (up to millions of rows). Large-scale distributed processing (Spark/Ray) is intentionally not coupled to keep DIVE lightweight.
- **Statistical Assumptions**: Leakage and drift metrics rely on empirical distributions (PSI, KS, MI) which require sufficient sample sizes (>= 10 rows).
