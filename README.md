# DIVE: Industrial-Grade Autonomous AutoML & ML Reliability Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aman-i1/DIVE/blob/main/examples/colab_quickstart.ipynb)

**DIVE** (Data Intelligence, Validation & Ensembling) is an industrial-grade **Autonomous Machine Learning, Reliability, and MLOps Platform** engineered for mission-critical enterprise workloads across **three core capability domains**:

```
DIVE Ecosystem:
├── DIVE ML  (Tabular & Time Series AutoML)    --> dive ml / dive train / dive auto
├── DIVE NLP (Natural Language Processing)     --> dive nlp (AutoNLP, LSA, Zero-Shot)
└── DIVE DL  (Deep Learning Across 5 Modes)    --> dive dl (Tabular, Text, Image, Audio, Video)
```

A user provides a dataset and an objective (`dive auto data.csv --target churn`, `dive nlp train text.csv`, or `dive dl train ./images`). DIVE autonomously profiles data health, mitigates temporal/entity leakage, selects safe validation strategies, engineers point-in-time features, schedules multi-fidelity trials (ASHA), performs calibrated stacking, extracts dense semantic and acoustic features, quantifies conformal uncertainty, serves high-throughput endpoints, and monitors real-time drift.

```bash
# Tabular AutoML & Reliability
dive auto   data.csv --target churn --budget 10m --output ./out # 20-engine autonomous execution
dive train  sales.csv --target revenue --time-column date       # Point-in-time safe lag features
dive gate   model.pkl --data prod_batch.csv --ref train.csv     # Pre-deployment CI/CD gatekeeper

# Natural Language Processing
dive nlp train reviews.csv --text-col review --target-col score # AutoNLP multi-representation search
dive nlp zero-shot "Invoice #1092 approved" --labels "finance,sports,gaming" # Zero-shot classification

# Deep Learning (Vision & Audio)
dive dl train ./data/images -m image -t classification         # Edge-gradient spatial vision modeling
dive dl train ./data/audio  -m audio -t classification         # 160-dim log-mel & MFCC cepstral modeling
```

---

## Three Core Capability Domains

### 1. Tabular Machine Learning (`dive ml`)
- **Memory Downcasting Optimizer**: Automatically downcasts `float64` to `float32` and `int64` to `int32/16/8`, reducing RAM consumption by 50–70% while preserving strict binary columnar schemas (`.parquet`, `.feather`).
- **Point-in-Time Safe Temporal Engine**: Automatically shifts lag features (`.shift(lag)`) and rolling window statistics (`.shift(1).rolling(w).mean()`) to eliminate temporal lookahead leakage. Group-aware rolling aggregations prevent cross-entity contamination.
- **Pre-Flight Validation & Readiness Score**: 5-point data health audit scoring datasets 0–100 before training.
- **Autonomous ASHA Scheduler**: Multi-fidelity trial search optimizing accuracy, log-loss, calibration error, and inference latency.
- **Calibrated Stacking**: Convex blend optimization on out-of-fold calibrated probabilities.
- **Production Gatekeeper & Drift**: Paired statistical deployment gating (`dive gate`) and population stability analysis (`dive drift`).

### 2. Natural Language Processing (`dive nlp`)
- **AutoNLP Autonomous Search**: Evaluates candidate representations (TF-IDF, Character n-grams, Word+Char unions, Okapi BM25, and Latent Semantic Analysis) against diverse estimators.
- **Continuous Topic Embeddings (`LSARepresentation`)**: High-speed dense semantic vectors via `TruncatedSVD` on n-gram matrices (>10,000 docs/sec CPU speed, zero external downloads).
- **Zero-Shot Classification (`ZeroShotClassifier`)**: Dual-granularity word + subword representations, domain anchor lexicon expansion (`finance`, `sports`, `medical`, `tech`, `spam`, etc.), and adaptive temperature calibration for sharp probability differentiation without training data.
- **Vocabulary Drift Monitoring**: Detects token distribution shift, length anomalies, and Out-of-Vocabulary (OOV) surges in production.
- **High-Throughput Serving**: Standalone FastAPI REST API server with interactive Swagger UI.

### 3. Multi-Modal Deep Learning (`dive dl`)
- **Five Data Modalities**: First-class support for Tabular, Text, Image, Audio, and Video.
- **Dual-Backend Execution Guarantee**:
  - *PyTorch Backend*: AdamW, Cosine Annealing learning rate schedules, gradient clipping, mixed precision (AMP), and in-memory best-epoch checkpoint recovery.
  - *Scikit-Learn Fallback*: Zero-failure CPU execution via MLP without requiring GPU drivers or multi-gigabyte neural weight downloads.
- **Acoustic Engineering (`AudioAdapter`)**: 64-band log-mel filterbanks, 13 Mel-Frequency Cepstral Coefficients (MFCCs) via DCT-II, spectral contrast (peak minus valley energy), and peak sample amplitude (160 total features).
- **Vision Engineering (`ImageAdapter`)**: Dual tier: pretrained ResNet-18 backbone (when torchvision is available) or downsampled spatial grid + 48-bin RGB histograms + spatial gradient magnitude statistics (Sobel proxy for edge contours) + contrast and color balance (1080 features).
- **Flexible Media Routing**: Direct single-file scoring (`dive dl predict model.pkl --data image.png`) or batch directory scoring.

---

## Quickstart

## Installation & Multi-OS Setup

> For detailed OS-specific setup (Linux, macOS Apple Silicon/Intel, Windows WSL2/PowerShell, and Docker), see the [**Complete Multi-OS Installation Guide**](docs/INSTALLATION.md).

### 1. Quick Install directly from GitHub (All OS)
```bash
# Core Tabular AutoML & Reliability
pip install git+https://github.com/Aman-i1/DIVE.git

# Core + NLP & FastAPI Serving Extras
pip install "dive-ml[nlp,serving] @ git+https://github.com/Aman-i1/DIVE.git"

# Full Suite (All 5 Modalities + Boosters + Deep Learning)
pip install "dive-ml[full] @ git+https://github.com/Aman-i1/DIVE.git"
```

### 2. Install from Local Clone
```bash
git clone https://github.com/Aman-i1/DIVE.git
cd DIVE

# On Linux / macOS:
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[nlp,serving]"

# On Windows (PowerShell):
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[nlp,serving]"
```

Verify your setup:
```bash
dive --version
dive deps
```


### Tabular Workflow
```bash
# Train on tabular data with time series features
dive train sales.csv --target revenue --time-column date --group-column store_id

# Score new batch
dive predict --model ./dive_output/model.pkl --data test_sales.csv --output predictions.csv

# Audit deployment readiness
dive doctor sales.csv --target revenue
```

### NLP Workflow
```bash
# Train AutoNLP model
dive nlp train reviews.csv --text-col review --target-col sentiment -o sentiment_model.pkl

# Zero-Shot text classification
dive nlp zero-shot "Quarterly revenue surged by 22% this fiscal year" --labels "finance,technology,sports"

# Serve REST API
dive nlp serve sentiment_model.pkl --port 8000
```

### Deep Learning Workflow
```bash
# Train on folder-per-class image collection
dive dl train ./images/ -m image -t classification --epochs 10 -o vision_model.pkl

# Train on audio collection
dive dl train ./audio/  -m audio -t classification --epochs 10 -o audio_model.pkl

# Score a single image or sound file
dive dl predict vision_model.pkl --input ./test_photo.png --proba
dive dl predict audio_model.pkl  --input ./test_sound.wav --proba
```

---

## CLI Command Index

> For complete flag-by-flag documentation, real-world examples, and failure modes, see the [**Complete CLI Reference Manual**](docs/CLI_COMMANDS_REFERENCE.md).

| Command Group | Command | Purpose | Typical Command |
| :--- | :--- | :--- | :--- |
| **Tabular ML** | `dive autopilot` | 20-step Senior ML Review + Reliability + AutoML | `dive autopilot data.csv --target churn` |
| | `dive auto` | Autonomous AutoML study execution | `dive auto data.csv --target churn` |
| | `dive train` | Model zoo training, tuning, stacking & reporting | `dive train data.csv --target churn --time-column date` |
| | `dive predict` | Batch scoring with probability distributions | `dive predict --model model.pkl --data test.csv` |
| | `dive doctor` | Pre-flight dataset health & 0-100 readiness audit | `dive doctor data.csv --target churn` |
| | `dive gate` | Pre-deployment gatekeeper for CI/CD pipelines | `dive gate model.pkl --data batch.csv --strict` |
| | `dive drift` | Continuous PSI & Kolmogorov-Smirnov drift monitoring | `dive drift --ref train.csv --curr prod.csv` |
| | `dive review` | Senior ML Practitioner automated review report | `dive review data.csv --target churn` |
| | `dive explain` | Feature attributions, SHAP, and pipeline summary | `dive explain --model model.pkl --data test.csv` |
| | `dive report` | Standalone interactive HTML & LaTeX research PDF | `dive report --model model.pkl --output ./report` |
| **NLP** | `dive nlp train` | Autonomous multi-representation AutoNLP search | `dive nlp train data.csv -x text -y label` |
| | `dive nlp zero-shot` | Zero-shot text classification with anchor expansion | `dive nlp zero-shot "text" --labels "a,b,c"` |
| | `dive nlp predict` | High-throughput text inference engine | `dive nlp predict model.pkl -t "text" --proba` |
| | `dive nlp profile` | Document length, character, and vocabulary profiler | `dive nlp profile data.csv -x text` |
| | `dive nlp monitor` | Real-time vocabulary shift and OOV rate monitor | `dive nlp monitor base.csv curr.csv -x text` |
| | `dive nlp serve` | Deploy production FastAPI REST API server | `dive nlp serve model.pkl --port 8000` |
| **Deep Learning** | `dive dl train` | Modality-agnostic neural training (5 modalities) | `dive dl train ./images -m image -t classification` |
| | `dive dl predict` | Single media or batch inference engine | `dive dl predict model.pkl --input sample.png` |
| | `dive dl doctor` | Deep learning environment & RAM projection audit | `dive dl doctor -m audio` |
| | `dive dl auto` | Autonomous neural hyperparameter search (AutoDL) | `dive dl auto ./images -m image --trials 5` |
| **Governance** | `dive models` | Local model registry (register, list, promote) | `dive models register churn_model --model model.pkl` |
| | `dive experiments` | Experiment tracking, comparison, and metrics | `dive experiments list` |

---

## Comprehensive Guides & Documentation

- [**Complete CLI Reference Manual**](docs/CLI_COMMANDS_REFERENCE.md)
- [**NLP Quickstart Guide**](docs/nlp_quickstart.md)
- [**NLP Architecture & Engines**](docs/nlp_architecture.md)
- [**Deep Learning Quickstart Guide**](docs/dl_quickstart.md)

---

## License

MIT License. Developed by Aman-i1.
