# DIVE: Industrial-Grade Autonomous AutoML & ML Reliability Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aman-i1/DIVE/blob/main/examples/colab_quickstart.ipynb)

**DIVE** is an industrial-grade **Autonomous AutoML, Reliability, and MLOps Platform** designed for mission-critical tabular machine learning.

A user provides a dataset and an objective (`dive auto data.csv --target churn` or `dive.create_study()`). DIVE autonomously audits dataset health, detects temporal/entity leakage risks, selects safe validation strategies, engineers leakage-safe features, schedules multi-fidelity trials (ASHA), performs calibrated stacking, quantifies uncertainty with conformal prediction, generates cryptographic audit certificates, serves high-throughput batch and REST endpoints, and monitors production drift.

```bash
dive auto   data.csv --target churn --budget 10m --output ./out # Full 20-engine autonomous execution
dive doctor data.csv --target churn                             # Dataset diagnostic audit & readiness score
dive train  data.csv --target churn --mode balanced             # Model zoo search & calibrated stacking
dive gate   model.pkl --data prod_batch.csv --ref train.csv     # Statistical deployment gatekeeper
dive drift  --ref train.csv --curr prod.csv                     # PSI / KS drift & retraining urgency
```

---

## Core Architecture & Execution Flow

```
[Raw Dataset]
      |
      v
[Data Intelligence & Schema Profiler] ──> [Validation Intelligence & Leakage Auditor]
                                                         |
                                                         v
[Meta-Learning & Warm-Start Priors] <──── [Leakage-Safe Feature Engineering Engine]
      |
      v
[Autonomous ASHA Search Scheduler] ─────> [Calibrated Stacking & Convex Blending]
                                                         |
                                                         v
[Cryptographic Audit & Lineage DAG] <──── [Trust & Conformal Uncertainty Engine]
      |
      v
[Production REST / Streaming Batch Serving & Real-Time Drift Observability]
```

---

## 20 Core Domain Engines

1. **DIVE Study Orchestrator (`dive.orchestrator`, `dive.study`)**: Autonomous lifecycle coordination, hardware resource budgeting, and explainable decision logging.
2. **Data Intelligence Engine (`dive.data_intelligence`)**: Fine-grained semantic typing (numeric, categorical, datetime, high-cardinality, sparse) and dataset layout detection (IID, Grouped, Temporal, Panel).
3. **Validation Intelligence Engine (`dive.validation_engine`)**: Automatic detection of target leakage, entity contamination, and temporal ordering with safe strategy selection (`StratifiedGroupKFold`, `GroupKFold`, `TimeSeriesSplit`, `StratifiedKFold`, `KFold`).
4. **Leakage-Safe Feature Engineering (`dive.temporal_features`)**: Point-in-time safe lag features, shifted rolling statistics, expanding aggregates, and entity-level features.
5. **Feature Intelligence & Selection (`dive.feature_selection`)**: Mutual information ranking, collinearity filtering, and variance thresholding.
6. **Model Intelligence & Capability Registry (`dive.capability_registry`)**: Estimator capability metadata matching models to dataset shapes and hardware constraints.
7. **Autonomous Search Engine (`dive.search_scheduler`)**: Asynchronous Successive Halving (ASHA) multi-fidelity trial scheduling with multi-objective optimization (metric + calibration + latency + memory).
8. **Meta-Learning & Fingerprinting Engine (`dive.meta_learning`)**: Statistical moments, target entropy, and landmark baseline extraction for search warm-starting.
9. **Ensemble Intelligence (`dive.ensemble_diversity`)**: Pairwise error correlation, Yule's Q-statistics, and greedy forward selection with diversity penalty.
10. **Calibrated Multi-Layer Stacking (`dive.stacking_calibrated`)**: Constrained convex blend optimization and meta-estimators on out-of-fold calibrated probabilities.
11. **Trust & Uncertainty Engine (`dive.uncertainty`, `dive.trust`)**: Distribution-free conformal prediction intervals, epistemic vs. aleatoric uncertainty decomposition, perturbation robustness testing, and subgroup disparity audits.
12. **Out-of-Distribution Detector (`dive.ood_detector`)**: Isolation Forest density combined with PCA-Mahalanobis distance scoring.
13. **Experiment Engine & Lineage DAG (`dive.lineage`)**: Cryptographic provenance tracking from raw dataset to compliance certificate with Mermaid diagram export.
14. **Content-Addressable Artifact Store (`dive.artifact_store`)**: SHA-256 content-addressable model and dataset storage with deduplication.
15. **One-Click Reproducibility (`dive.reproducibility`)**: Standalone reproduction bundle generation with `reproduce.py`, `metadata.json`, and `model.pkl`.
16. **High-Performance Serving (`dive.batch_inference`, `dive.serving`)**: Streaming chunked batch processor for large datasets and FastAPI REST API deployment.
17. **Dynamic Inference Router (`dive.inference_router`)**: Dynamic confidence-based routing between lightweight models and calibrated ensembles.
18. **Production Observability (`dive.observability`)**: Continuous feature PSI, Kolmogorov-Smirnov drift tests, prediction drift, and automated retraining urgency scoring.
19. **Champion/Challenger Promotion Gate (`dive.champion_challenger`)**: Paired Wilcoxon hypothesis testing to statistically verify model superiority before replacement.
20. **Security & Auditing (`dive.security`, `dive.audit`)**: Safe deserialization inspection, path traversal guards, and cryptographic compliance certificates.

---

## Multi-Language Ecosystem

DIVE is engineered with high-performance polyglot components:
- **Rust Core Engine (`crates/dive-core/`)**: Zero-overhead high-throughput data processing and statistical computation.
- **C-ABI Shared Library (`libdive`)**: Native shared library bindings for high-performance integration.
- **Go Binary CLI (`cmd/dive-go/`)**: Single static binary CLI tool with zero external runtime dependencies.
- **WebAssembly Engine (`wasm/`)**: Client-side in-browser validation and inference engine compiled from Rust.
- **Python Package (`dive/`)**: Unified Python SDK and Click CLI with extensive scientific ecosystem support.

---

## Quickstart

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
# 1. Run full autonomous AutoML study (orchestrating all 20 engines)
dive auto sales.csv --target churn --budget 10m --output ./dive_out

# 2. Diagnostic audit & check Production Readiness Score
dive doctor sales.csv --target churn --group-column customer_id

# 3. Model validation & deployment gatekeeper
dive gate ./dive_out/model.pkl --data prod_batch.csv --ref sales.csv --strict

# 4. Deploy production FastAPI REST API server
dive serve --model ./dive_out/model.pkl --port 8000

# 5. Monitor drift against production batch data
dive drift --ref sales.csv --curr production_week1.csv
```

### CLI Command Index

> For complete flag-by-flag documentation, real-world examples, and failure modes, see the [**Complete CLI Reference Manual**](docs/CLI_COMMANDS_REFERENCE.md).

| Command | Category | Purpose | Typical Command |
|---|---|---|---|
| `dive autopilot` | Orchestration | Full 20-step Senior ML Review, Reliability & AutoML | `dive autopilot data.csv --target churn --budget 10m` |
| `dive auto` | Orchestration | Declarative YAML or autonomous study execution | `dive auto data.csv --target churn --mode balanced` |
| `dive train` | Training | Model zoo search, tuning, calibrated stacking & reports | `dive train data.csv --target churn --output ./out` |
| `dive predict` | Inference | High-throughput batch or terminal prediction machine | `dive predict --model model.pkl --data test.csv --proba` |
| `dive contract` | Reliability | Formal Prediction Contract inference & specification | `dive contract data.csv --target churn --entity cust_id` |
| `dive review` | Reliability | Senior ML Practitioner automated review & audit matrix | `dive review data.csv --target churn --output review.json` |
| `dive gate` | Production | Pre-deployment verification gatekeeper for CI/CD | `dive gate model.pkl --data batch.csv --strict` |
| `dive doctor` | Diagnostics | Pre-flight audit & 0-100 Production Readiness Score | `dive doctor data.csv --target churn` |
| `dive serve` | Production | REST API server with interactive `/help` endpoint | `dive serve --model model.pkl --port 8000` |
| `dive drift` | Observability | PSI / Kolmogorov-Smirnov continuous drift monitoring | `dive drift --ref train.csv --curr prod.csv` |
| `dive explain` | Explainability | Permutation Importance, SHAP & partial dependence | `dive explain --model model.pkl --data test.csv` |
| `dive report` | Reporting | Standalone interactive HTML & LaTeX research PDF | `dive report --model model.pkl --output ./report` |
| `dive reproduce` | Reproducibility | Content-addressable SHA-256 bundle & `reproduce.py` | `dive reproduce --model model.pkl` |
| `dive audit` | Compliance | Signed SHA-256 cryptographic compliance certificate | `dive audit --model model.pkl --output cert.json` |
| `dive export` | Export | Export to ONNX, PMML, TorchScript, C-FFI, WASM | `dive export --model model.pkl --format onnx` |
| `dive info` | Inspection | Deep structural & statistical dataset profiler | `dive info data.csv --target churn` |
| `dive benchmark` | Benchmarking | Predictor latency, RAM, and throughput benchmarks | `dive benchmark --model model.pkl --data test.csv` |
| `dive deps` | Environment | Hardware acceleration (CUDA) and library audit | `dive deps` |
| `dive docs` | Documentation | Local offline documentation server & browser viewer | `dive docs --port 8080` |

---

## License

MIT License. Developed by Aman-i1.
