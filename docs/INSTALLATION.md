# DIVE Comprehensive Multi-OS Installation Guide

This guide provides detailed, step-by-step instructions to install and configure **DIVE** across **Linux**, **macOS**, **Windows**, **Google Colab / Cloud Notebooks**, and **Docker**.

---

## System Requirements & Prerequisites

- **Python**: Python 3.9, 3.10, 3.11, 3.12, or 3.13.
- **Git**: Installed and available on system `PATH`.
- **RAM**: Minimum 2 GB RAM (4 GB+ recommended).
- **Disk Space**: ~150 MB for core installation (up to 1.5 GB if installing full deep learning and GPU packages).
- **Hardware Acceleration (Optional)**:
  - NVIDIA GPU with CUDA 11.8 / 12.x for accelerated training.
  - Apple Silicon M1/M2/M3/M4 (MPS acceleration automatically detected).

---

## 1. Quick Install directly from GitHub (All Platforms)

If you just want to use DIVE directly without cloning:

```bash
# 1. Base Installation (Tabular ML & Core Engines)
pip install git+https://github.com/Aman-i1/DIVE.git

# 2. Base + NLP & Model Serving Extras
pip install "dive-ml[nlp,serving] @ git+https://github.com/Aman-i1/DIVE.git"

# 3. Full Complete Installation (Tabular, NLP, Deep Learning & GPU Boosters)
pip install "dive-ml[full] @ git+https://github.com/Aman-i1/DIVE.git"
```

Verify installation:
```bash
dive --version
dive deps
```

---

## 2. Installation by Operating System

### A. Linux (Ubuntu, Debian, Fedora, CentOS, Arch)

#### 1. Install System Dependencies
On Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git build-essential libgomp1
```
On Fedora / RHEL:
```bash
sudo dnf install -y python3 python3-pip git gcc gcc-c++ libgomp
```
On Arch Linux:
```bash
sudo pacman -S python python-pip git base-devel openmp
```

#### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Clone and Install
```bash
git clone https://github.com/Aman-i1/DIVE.git
cd DIVE
pip install --upgrade pip setuptools wheel
pip install -e ".[nlp,serving]"
```

---

### B. macOS (Apple Silicon M-Series & Intel)

#### 1. Install Prerequisites via Homebrew
Open Terminal and ensure Homebrew is installed:
```bash
# Install OpenMP (required for LightGBM and XGBoost multi-threading on macOS)
brew install libomp git python@3.11
```

#### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Clone and Install
```bash
git clone https://github.com/Aman-i1/DIVE.git
cd DIVE
pip install --upgrade pip
pip install -e ".[nlp,serving]"
```

*Note for Apple Silicon (M1/M2/M3/M4)*: PyTorch MPS acceleration is automatically detected by `dive dl` when PyTorch is installed:
```bash
pip install torch torchvision
```

---

### C. Windows 10 & 11 (PowerShell / Command Prompt)

#### 1. Install Python & Git
- Download and install **Python 3.10 - 3.13** from [python.org](https://www.python.org/downloads/).
  - **IMPORTANT**: Check the box **"Add python.exe to PATH"** during setup.
- Download and install **Git for Windows** from [git-scm.com](https://git-scm.com/).

#### 2. Enable Long Paths (Recommended for Deep Learning checkpoints)
In PowerShell as Administrator:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### 3. Create and Activate Virtual Environment
Open PowerShell or Command Prompt:
```powershell
python -m venv .venv
# On PowerShell:
.\.venv\Scripts\Activate.ps1
# (If execution policy restricts scripts, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# On Command Prompt (cmd.exe):
.\.venv\Scripts\activate.bat
```

#### 4. Clone and Install
```powershell
git clone https://github.com/Aman-i1/DIVE.git
cd DIVE
python -m pip install --upgrade pip
pip install -e .
```

To install optional NLP & Serving capabilities:
```powershell
pip install -e ".[nlp,serving]"
```

---

### D. Windows Subsystem for Linux (WSL2)

If you prefer running in a full Linux environment on Windows with seamless NVIDIA GPU pass-through:

1. Open PowerShell and install Ubuntu:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Open Ubuntu terminal in WSL2 and follow the [Linux Instructions](#a-linux-ubuntu-debian-fedora-centos-arch) above.
3. NVIDIA CUDA is automatically passed through from Windows drivers without needing separate driver installs inside WSL.

---

### E. Google Colab & Kaggle Notebooks

Run the following cell at the top of your Colab or Kaggle notebook:

```python
# Install DIVE directly into Colab runtime
!git clone -q https://github.com/Aman-i1/DIVE.git
%cd DIVE
!pip install -q -e ".[nlp,serving]"

# Verify environment
!dive --version
!dive deps
```

You can also run our official pre-configured Colab notebook directly:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aman-i1/DIVE/blob/main/examples/colab_quickstart.ipynb)

---

### F. Docker / Containerized Setup

DIVE can be deployed in lightweight or GPU-accelerated Docker containers.

#### Minimal CPU Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[nlp,serving]"

EXPOSE 8000
ENTRYPOINT ["dive"]
CMD ["--help"]
```

Build and run:
```bash
docker build -t dive-ml .
docker run --rm -it dive-ml --version
docker run --rm -p 8000:8000 -v $(pwd)/data:/data dive-ml serve --model /data/model.pkl --port 8000
```

---

## 3. Package Extras & Optional Dependencies

| Extra Name | Command | Unlocks |
| :--- | :--- | :--- |
| **`nlp`** | `pip install -e ".[nlp]"` | SentenceTransformers, transformers, tokenizers, subword extraction. |
| **`serving`** | `pip install -e ".[serving]"` | FastAPI, Uvicorn, REST API endpoints, streaming batch server. |
| **`boosters`** | `pip install -e ".[boosters]"` | LightGBM, XGBoost, CatBoost gradient boosted decision trees. |
| **`tuning`** | `pip install -e ".[tuning]"` | Optuna hyperparameter optimization and ASHA multi-fidelity search. |
| **`explain`** | `pip install -e ".[explain]"` | SHAP, permutation importance, and partial dependence plots. |
| **`full`** | `pip install -e ".[full]"` | Everything above: All engines, boosters, NLP, and serving tools. |

---

## 4. Post-Installation Health Verification

Run the built-in dependency and environment audit:
```bash
dive deps
```

Run dataset doctor and smoke test:
```bash
dive doctor examples/sample.csv --target diagnosis
```

If both commands run and return green `[PASS]` statuses, your installation is complete and ready for production!
