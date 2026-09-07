# DIVE Deep Learning (`dive dl`) Quickstart Guide

**DIVE DL** is the modality-agnostic deep learning extension for DIVE, providing end-to-end neural training and inference across **five data modalities**:

```
DIVE Deep Learning Modalities:
├── Tabular   (Structured numerical & categorical CSV/Parquet)
├── Text      (Raw text documents, reviews, or transcripts)
├── Image     (Image files & directories: JPG, PNG, WEBP)
├── Audio     (Acoustic files & directories: WAV, FLAC, OGG)
└── Video     (Video clips & frame sequences: MP4, AVI, MKV)
```

---

## 1. Dual-Backend Architecture: Zero-Failure Guarantee

DIVE DL operates on an industrial dual-backend contract:
1. **PyTorch Backend (`torch-mlp`)**: Used whenever PyTorch is present. Trains dense neural architectures with AdamW, Cosine Annealing learning rate schedules, gradient clipping, mixed precision (AMP on CUDA), and in-memory best-epoch checkpoint recovery.
2. **Scikit-Learn Backend (`sklearn-mlp`)**: Used when PyTorch is not installed (e.g. lightweight CPU laptop or constrained server). Guaranteed to complete training and produce a deployable `.pkl` predictor artifact without external multi-gigabyte downloads.

---

## 2. Advanced Feature Extraction Across Modalities

Every modality adapter transforms raw inputs into a dense `float32` representation:

### Audio Modality (`AudioAdapter`)
- **Log-Mel Spectrogram**: 64 triangular mel filterbanks covering audible frequencies.
- **Mel-Frequency Cepstral Coefficients (MFCCs)**: Discrete Cosine Transform (DCT-II) extracting 13 cepstral means and 13 standard deviations.
- **Global Spectral Statistics**: Duration, RMS energy, zero-crossing rate, spectral centroid, spectral contrast (peak minus valley energy), and peak sample amplitude (160 total features).

### Vision Modality (`ImageAdapter`)
- **Pretrained Backbone Tier**: Frozen ResNet-18 with 512 semantic features (when `torch` + `torchvision` installed).
- **Pixel & Edge Statistics Tier**: Downsampled grayscale grid, 48-bin RGB histograms, spatial gradient magnitude statistics (Sobel/Laplacian proxy for edge contours), contrast, dynamic range, and per-channel illumination moments (1080 total features).

### Video Modality (`VideoAdapter`)
- Uniform frame sampling across clip durations, featurized through the image pipeline and pooled via joint mean and max spatial pooling.

---

## 3. Python API Quickstart

### Image Classification
```python
from dive.dl import get_adapter, DLTrainer, DLPredictor, DLTask, Modality

# 1. Featurize image directory
adapter = get_adapter(Modality.IMAGE)
batch = adapter.load("./dataset/images", data_config=None)
X = adapter.fit_transform(batch.inputs)
y = batch.targets

# 2. Train neural model
trainer = DLTrainer(task=DLTask.CLASSIFICATION, epochs=10)
trainer.fit(X, y)

# 3. Save predictor artifact
predictor = DLPredictor(adapter=adapter, trainer=trainer, modality=Modality.IMAGE)
predictor.save("image_model.pkl")

# 4. Predict on a new image
pred = predictor.predict(["./test_image.png"])
print("Predicted Class:", pred[0])
```

### Audio Classification
```python
from dive.dl import get_adapter, DLTrainer, DLTask, Modality

adapter = get_adapter(Modality.AUDIO)
batch = adapter.load("./dataset/audio", data_config=None)
X = adapter.fit_transform(batch.inputs)
y = batch.targets

trainer = DLTrainer(task=DLTask.CLASSIFICATION, epochs=10)
trainer.fit(X, y)
```

---

## 4. CLI Reference

### 1. Pre-Flight Diagnostic Audit
```bash
# Check device, RAM projections, and installed acceleration packages
dive dl doctor -m image
dive dl doctor -m audio
```

### 2. Dataset Inspection & Sizing
```bash
# Inspect dataset sample counts, dimensions, and memory estimates
dive dl info ./data/images -m image
dive dl info ./data/audio -m audio
```

### 3. Model Training
```bash
# Train on folder-per-class image collection
dive dl train ./data/images -m image -t classification --epochs 10 -o image_model.pkl

# Train on audio collection
dive dl train ./data/audio -m audio -t classification --epochs 10 -o audio_model.pkl

# Force CPU fallback
dive dl train ./data/tabular.csv -m tabular -t classification --no-torch -o tab_model.pkl
```

### 4. Inference & Batch Scoring
```bash
# Score a single image or audio clip
dive dl predict image_model.pkl --input ./new_sample.png --proba
dive dl predict audio_model.pkl --input ./test_audio.wav --proba

# Batch score a folder or manifest
dive dl predict image_model.pkl --data ./unlabelled_images/ -o predictions.csv
```

### 5. Latency & Throughput Benchmarking
```bash
dive dl benchmark image_model.pkl --samples 50
```

### 6. Autonomous Multi-Trial Search (AutoDL)
```bash
dive dl auto ./data/images -m image --trials 5 -o champion_dl.pkl
```
