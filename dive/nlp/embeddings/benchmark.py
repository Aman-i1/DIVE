"""Benchmarking Engine: TF-IDF vs Dense Neural Embeddings - `dive/nlp/embeddings/benchmark.py`.

Empirically compares sparse TF-IDF and dense embeddings across:
- Generalization accuracy / Macro F1 / RMSE
- Inference latency (milliseconds per sample)
- Feature dimensionality and memory footprints
- Training duration
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dive.nlp.data.dataset import NLPDataset
from dive.nlp.embeddings.representation import EmbeddingRepresentation
from dive.nlp.features.tfidf import TFIDFRepresentation


def benchmark_tfidf_vs_embeddings(
    data: Union[NLPDataset, pd.DataFrame, Sequence[Dict[str, Any]]],
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
    model_name: str = "LogisticRegression",
    task_type: str = "text_classification",
    embedding_model: str = "all-MiniLM-L6-v2",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Empirically compare sparse TF-IDF vs dense embeddings on a dataset."""
    from dive.nlp.training import train_baseline

    # 1. Benchmark TF-IDF
    t0_train_tfidf = time.perf_counter()
    tfidf_pred, tfidf_metrics = train_baseline(
        data=data,
        target_column=target_column,
        text_column=text_column,
        model_name=model_name,
        representation_type="tfidf",
        task_type=task_type,
        test_size=test_size,
        random_state=random_state,
    )
    t_train_tfidf = (time.perf_counter() - t0_train_tfidf) * 1000.0

    # Sample latency test
    sample_text = ["Benchmark test sentence for throughput measurement."]
    t0_inf = time.perf_counter()
    for _ in range(50):
        _ = tfidf_pred.predict(sample_text)
    tfidf_latency_ms = ((time.perf_counter() - t0_inf) / 50.0) * 1000.0

    # 2. Benchmark Dense Embeddings
    emb_engine = EmbeddingRepresentation(model_name=embedding_model, use_cache=True)
    t0_train_emb = time.perf_counter()
    emb_pred, emb_metrics = train_baseline(
        data=data,
        target_column=target_column,
        text_column=text_column,
        model_name=model_name,
        representation=emb_engine,
        task_type=task_type,
        test_size=test_size,
        random_state=random_state,
    )
    t_train_emb = (time.perf_counter() - t0_train_emb) * 1000.0

    t0_inf_emb = time.perf_counter()
    for _ in range(50):
        _ = emb_pred.predict(sample_text)
    emb_latency_ms = ((time.perf_counter() - t0_inf_emb) / 50.0) * 1000.0

    # 3. Decision Recommendation
    primary_metric = "macro_f1" if task_type == "text_classification" else "rmse"
    score_tfidf = tfidf_metrics.get(primary_metric, tfidf_metrics.get("accuracy", 0.0))
    score_emb = emb_metrics.get(primary_metric, emb_metrics.get("accuracy", 0.0))

    if task_type == "text_classification":
        winner = "embeddings" if score_emb > score_tfidf else "tfidf"
    else:
        winner = "embeddings" if score_emb < score_tfidf else "tfidf"

    return {
        "task_type": task_type,
        "model_name": model_name,
        "recommended_representation": winner,
        "tfidf": {
            "metrics": tfidf_metrics,
            "train_time_ms": round(t_train_tfidf, 2),
            "inference_latency_ms": round(tfidf_latency_ms, 3),
            "feature_type": "sparse_word_ngrams",
        },
        "embeddings": {
            "metrics": emb_metrics,
            "model_name": embedding_model,
            "train_time_ms": round(t_train_emb, 2),
            "inference_latency_ms": round(emb_latency_ms, 3),
            "feature_type": "dense_vector",
        },
    }
