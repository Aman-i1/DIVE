# DIVE NLP Architectural Blueprint & Protocol Specification

## 1. Architectural Philosophy

DIVE NLP is built around four fundamental design tenets:

1. **Domain Isolation**:
   - `dive.ml` and `dive.nlp` are completely decoupled capability domains.
   - Root imports (`from dive import DivePredictor`, `from dive import auto_fit`) preserve 100% backward compatibility for all existing tabular ML workloads.
2. **Protocol-Driven Interfaces**:
   - Every major component conforms to `@runtime_checkable` Python `typing.Protocol` interfaces in `dive.nlp.interfaces`.
3. **Graceful Dependency Degradation**:
   - Core NLP functionality (classical models, profiling, preprocessing, tokenization, BM25, union representations) runs on standard dependencies without requiring PyTorch or CUDA.
   - Deep learning components (Dense Embeddings, Hugging Face Transformers, ONNX) probe availability via `dive.utils.optional` with zero-crash fallbacks.
4. **Production-Ready Artifacts**:
   - Every trained pipeline is self-contained as an `NLPPredictor` containing preprocessing, representations, estimators, schema metadata, and telemetry.

---

## 2. Core Protocol Hierarchy

```
NLPDatasetProtocol
    ├── texts: Sequence[str]
    ├── labels: Optional[Sequence[Any]]
    ├── has_labels: bool
    └── split(test_size, stratify, random_state)

NLPPreprocessorProtocol
    ├── fit(texts) -> Self
    ├── transform(texts) -> List[str]
    └── fit_transform(texts) -> List[str]

NLPRepresentationProtocol
    ├── fit(texts, y) -> Self
    ├── transform(texts) -> Union[np.ndarray, scipy.sparse.spmatrix]
    └── fit_transform(texts, y) -> Union[np.ndarray, scipy.sparse.spmatrix]

NLPEstimatorProtocol
    ├── fit(X, y) -> Self
    └── predict(X) -> np.ndarray

NLPPipelineProtocol
    ├── fit(texts, y) -> Self
    └── predict(texts) -> np.ndarray

NLPPredictorProtocol
    ├── model_name: str
    ├── has_proba: bool
    ├── class_names: Optional[List[str]]
    ├── predict(data) -> np.ndarray
    ├── predict_proba(data) -> np.ndarray
    └── describe_input() -> Dict[str, Any]
```

---

## 3. Component Directory Layout

```
dive/nlp/
├── __init__.py                # Consolidated public exports
├── config.py                  # NLPConfig & sub-configuration dataclasses
├── exceptions.py              # NLPError exception hierarchy
├── interfaces.py              # Runtime checkable Protocols
├── registry.py                # NLPRegistry model catalog
├── pipeline.py                # NLPPipeline composable execution graph
├── data/
│   ├── dataset.py             # NLPDataset, NLPSample, multi-format loaders
│   └── splitter.py            # Stratified & deterministic dataset splitting
├── profiler/
│   ├── report.py              # NLPProfiler, NLPProfileReport
│   └── heuristics.py          # Column auto-detection heuristics
├── preprocessing/
│   ├── normalizer.py          # TextNormalizer Unicode & regex rules
│   └── preprocessor.py        # NLPPreprocessor & raw mode
├── features/
│   ├── tfidf.py               # TFIDFRepresentation, CountRepresentation
│   ├── ngrams.py              # CharNGramRepresentation, WordCharUnionRepresentation
│   └── bm25.py                # BM25Representation Okapi BM25
├── embeddings/
│   ├── cache.py               # Two-tier memory + disk SHA-256 EmbeddingCache
│   ├── representation.py      # EmbeddingRepresentation with device detection
│   ├── similarity.py          # Semantic cosine similarity
│   └── benchmark.py           # TF-IDF vs Embedding comparative benchmark
├── models/
│   └── baselines.py           # LogisticRegression, LinearSVC, MultinomialNB, Ridge
├── transformers/
│   ├── config.py              # TransformerConfig & model aliases
│   ├── estimator.py           # TransformerClassifier & TransformerRegressor
│   └── training.py            # train_transformer fine-tuning routine
├── automl/
│   ├── trial.py               # NLPTrial execution metadata
│   ├── leaderboard.py         # NLPLeaderboard ranking & ASCII formatting
│   └── engine.py              # AutoNLP multi-objective trial exploration
├── optimization/
│   ├── cache.py               # PredictionCache thread-safe LRU caching
│   ├── batching.py            # BatchInferenceEngine micro-batching
│   ├── onnx.py                # export_nlp_to_onnx & ONNXNLPPredictor
│   └── predictor.py           # OptimizedNLPPredictor production wrapper
├── serving/
│   └── app.py                 # FastAPI REST API endpoints & fallback server
└── monitoring/
    └── drift.py               # NLPDriftMonitor & NLPDriftReport
```
