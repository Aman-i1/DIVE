# DIVE NLP Quickstart Guide

Welcome to **DIVE NLP**, the enterprise-grade Natural Language Processing extension for DIVE.

DIVE is structured into two clean capability domains:
```
DIVE
├── DIVE ML  (Tabular & Structured Data AutoML)  --> from dive import ml / from dive import ...
└── DIVE NLP (Natural Language Processing)       --> from dive import nlp
```

---

## 1. 30-Second Quickstart (AutoNLP)

Autonomously search representations, classical estimators, dense embeddings, and transformer architectures:

```python
import pandas as pd
from dive import nlp

# 1. Load your dataset
df = pd.DataFrame({
    "review": [
        "Incredible build quality and rapid delivery, loved it!",
        "Poor customer support, defective unit and broken on arrival.",
        "Delightful experience, exceeded expectations!",
        "Terrible and completely useless, refund refused."
    ],
    "sentiment": ["positive", "negative", "positive", "negative"]
})

# 2. Autonomous model exploration & selection
predictor, leaderboard = nlp.fit_nlp(
    data=df,
    text_column="review",
    target_column="sentiment",
    max_trials=5,
    optimize_for="balanced"  # 'balanced', 'accuracy', or 'latency'
)

# 3. Inspect the leaderboard
print(leaderboard.render())

# 4. Predict on new incoming texts
predictions = predictor.predict(["Loved the build quality!"])
print("Prediction:", predictions)

# 5. Predict class probability distributions
if predictor.has_proba:
    probabilities = predictor.predict_proba(["Loved the build quality!"])
    print("Class Probabilities:", probabilities)
```

---

## 2. Text Profiling & Contamination Auditing

Profile document lengths, vocabulary diversity, and label conflicts before modeling:

```python
from dive import nlp

# Load and profile dataset
dataset = nlp.NLPDataset.from_dataframe(df, text_column="review", target_column="sentiment")
profiler = nlp.NLPProfiler(dataset)
report = profiler.profile()

# Print formatted ASCII diagnostics
print(report.render())
```

---

## 3. Modular Text Preprocessing

Normalize Unicode, strip HTML/URLs, remove noise, and configure stop words:

```python
from dive import nlp

# Clean normalized pipeline
preprocessor = nlp.NLPPreprocessor(
    config=nlp.NLPPreprocessingConfig(
        lowercase=True,
        strip_html=True,
        strip_urls=True,
        strip_emojis=True,
        max_char_length=512
    )
)
clean_texts = preprocessor.transform(["Check out https://example.com 🚀<b>Great!</b>"])

# Or non-destructive mode for Transformers
raw_preprocessor = nlp.NLPPreprocessor.raw()
```

---

## 4. Feature Representations Zoo

DIVE NLP provides a rich family of text representations:

| Representation | Class | Signal Captured | Best For |
| :--- | :--- | :--- | :--- |
| **Word TF-IDF** | `TFIDFRepresentation` | Word-level term frequency | Fast general baseline |
| **Char N-Grams** | `CharNGramRepresentation` | Subwords, prefixes, typos | Noisy OCR/social text |
| **Word+Char Union**| `WordCharUnionRepresentation`| Joint sparse union matrix | Maximum classical accuracy |
| **Okapi BM25** | `BM25Representation` | Probabilistic relevance scoring | Keyword matching & search |
| **Dense Embeddings**| `EmbeddingRepresentation` | Semantic sentence embeddings | Zero-shot & transfer learning |

```python
from dive import nlp

rep = nlp.build_representation(representation_type="word_char_union")
X = rep.fit_transform(df["review"])
```

---

## 5. Transformer Fine-Tuning (BERT, RoBERTa, DistilBERT)

Fine-tune deep pretrained transformer architectures:

```python
from dive import nlp

# Train DistilBERT classifier
predictor, report = nlp.train_transformer(
    data=df,
    text_column="review",
    target_column="sentiment",
    model_name="distilbert",  # 'distilbert', 'bert', 'roberta', 'deberta'
    epochs=3,
    batch_size=16
)
```

---

## 6. Production Optimization (Caching & Micro-Batching)

Harden predictors for ultra-low latency and OOM prevention:

```python
from dive import nlp

# Wrap predictor with in-memory LRU caching and micro-batching
opt_predictor = nlp.optimize_nlp_predictor(
    predictor=predictor,
    enable_cache=True,
    cache_capacity=10000,
    batch_size=64
)

# Repeated queries hit the LRU cache instantly (sub-millisecond)
preds = opt_predictor.predict(["Exceptional build quality!"])
print(opt_predictor.stats())
```

---

## 7. Production REST API Serving

Launch a production FastAPI model server exposing `/nlp/predict` and `/nlp/predict_proba`:

```python
from dive import nlp

# Launch server
nlp.serve_nlp_model(predictor, host="127.0.0.1", port=8000)
```

**Query with cURL:**
```bash
curl -X POST http://127.0.0.1:8000/nlp/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Loved this product!", "Defective and broken."]}'
```

---

## 8. Real-Time Distribution Drift Monitoring

Detect token length drift, vocabulary shifts, and Out-of-Vocabulary (OOV) rate increases:

```python
from dive import nlp

monitor = nlp.NLPDriftMonitor(
    reference_texts=training_texts,
    reference_predictions=training_preds,
    oov_threshold=0.15
)

# Check incoming production batch
drift_report = monitor.check_drift(
    current_texts=production_stream_texts,
    current_predictions=production_preds
)

# Print terminal diagnostic report
print(drift_report.render())
```

---

## 9. Zero-Shot Text Classification (Zero Training Required)

Classify raw text documents into arbitrary candidate categories without any training data or labels:

```python
from dive.nlp import ZeroShotClassifier

classifier = ZeroShotClassifier(temperature=0.15)
results = classifier.predict(
    "Your monthly checking account invoice and tax statement are ready.",
    candidate_labels=["finance", "sports", "gaming", "medicine"]
)

print("Predicted:", results[0]["predicted_label"])
print("Confidence:", f"{results[0]['confidence'] * 100:.1f}%")
print("All Probabilities:", results[0]["probabilities"])
```

Via CLI:
```bash
dive nlp zero-shot "Your quarterly investment statement is available" --labels "finance,sports,gaming,medicine"
```

---

## 10. Continuous Semantic Topic Embeddings (LSA)

Compute unit-normalized dense semantic embeddings locally on CPU at >10,000 docs/sec without neural network downloads:

```python
from dive.nlp.features import LSARepresentation

rep = LSARepresentation(n_components=64, ngram_range=(1, 2))
X_dense = rep.fit_transform(["Quarterly profits increased by 14% year over year."])
print("Dense Embedding Shape:", X_dense.shape)  # (1, 64)
```

---

## 11. Complete NLP CLI Reference

```bash
# 1. Inspect dataset schema and candidates
dive nlp info reviews.csv

# 2. Profile dataset token distributions
dive nlp profile reviews.csv --text-col review --target-col sentiment

# 3. Autonomous model training & champion selection
dive nlp train reviews.csv --text-col review --target-col sentiment --trials 5 --output model.pkl

# 4. Predict on new batches or terminal strings
dive nlp predict model.pkl -t "Outstanding product quality!" --proba

# 5. Zero-Shot classification
dive nlp zero-shot "The team won the championship match" --labels "sports,finance,gaming"

# 6. Benchmark latency percentiles
dive nlp benchmark model.pkl --samples 100

# 7. Serve production REST API
dive nlp serve model.pkl --port 8000

# 8. Monitor production vocabulary drift
dive nlp monitor baseline.csv production_batch.csv --text-col review
```

