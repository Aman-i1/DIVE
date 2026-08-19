"""Phase 10 Tests - DIVE NLP Production Optimization (LRU Caching, Micro-Batching, ONNX).

Verifies:
1. PredictionCache thread-safe LRU eviction, hit/miss tracking, and cache clearing.
2. BatchInferenceEngine adaptive micro-batching for large inference payloads.
3. OptimizedNLPPredictor prediction and probability parity with base predictors.
4. Telemetry tracking (requests, average latency, cache hit rate).
5. ONNX model export and ONNXNLPPredictor execution.
6. NLPPredictorProtocol compliance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from dive.nlp import (
    BatchInferenceEngine,
    ONNXNLPPredictor,
    OptimizedNLPPredictor,
    PredictionCache,
    export_nlp_to_onnx,
    optimize_nlp_predictor,
    train_baseline,
)
from dive.nlp.interfaces import NLPPredictorProtocol


@pytest.fixture
def sentiment_corpus() -> pd.DataFrame:
    """Fixture providing sample review corpus."""
    return pd.DataFrame(
        {
            "text": [
                "Exceptional build quality and super fast shipping, loved it!",
                "Amazing performance, totally exceeded my expectations, great product!",
                "Delightful customer support, great onboarding and seamless setup.",
                "Awesome durability, loved everything about this device, top quality.",
                "Great experience with fast delivery and superb quality overall.",
                "Worst purchase ever made, broken on arrival, terrible item.",
                "Terrible customer service, completely unhelpful and rude, awful.",
                "Defective unit, useless customer support, refund was refused, terrible.",
                "Awful product, broke within one hour, completely useless and worst.",
                "Horrible product, worst purchase, completely useless and broken.",
            ],
            "label": [
                "positive",
                "positive",
                "positive",
                "positive",
                "positive",
                "negative",
                "negative",
                "negative",
                "negative",
                "negative",
            ],
        }
    )


def test_prediction_cache_lru_and_stats() -> None:
    """Verify PredictionCache LRU eviction, hit rates, and statistics."""
    cache = PredictionCache(capacity=3)

    cache.set("query 1", "pos", np.array([0.1, 0.9]))
    cache.set("query 2", "neg", np.array([0.8, 0.2]))
    cache.set("query 3", "pos", np.array([0.2, 0.8]))
    assert len(cache) == 3

    # Query 1 hit
    hit1 = cache.get("query 1")
    assert hit1 is not None
    assert hit1[0] == "pos"

    # Insert 4th item -> query 2 should be evicted (query 1 was accessed, so 2 is oldest)
    cache.set("query 4", "pos", np.array([0.3, 0.7]))
    assert cache.get("query 2") is None
    assert cache.get("query 1") is not None

    stats = cache.stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert 0.0 < stats["hit_rate"] <= 1.0

    cache.clear()
    assert len(cache) == 0


def test_batch_inference_engine() -> None:
    """Verify adaptive micro-batching partitions large queries correctly."""
    batcher = BatchInferenceEngine(default_batch_size=3)

    texts = [f"Sample query document {i}" for i in range(10)]

    def mock_predict(batch: List[str]) -> np.ndarray:
        return np.array([f"pred_{s}" for s in batch])

    results = batcher.run_batched(mock_predict, texts, batch_size=3)
    assert len(results) == 10
    assert results[0] == "pred_Sample query document 0"
    assert results[9] == "pred_Sample query document 9"


def test_optimized_nlp_predictor_parity(sentiment_corpus: pd.DataFrame) -> None:
    """Verify OptimizedNLPPredictor produces exact identical results with telemetry speedup."""
    base_predictor, _ = train_baseline(
        data=sentiment_corpus,
        text_column="text",
        target_column="label",
        model_name="LogisticRegression",
    )

    opt_predictor = optimize_nlp_predictor(
        predictor=base_predictor,
        enable_cache=True,
        cache_capacity=100,
        batch_size=4,
    )
    assert isinstance(opt_predictor, NLPPredictorProtocol)

    test_queries = [
        "Exceptional build quality, loved it!",
        "Terrible and broke on arrival.",
        "Exceptional build quality, loved it!",  # Repeated query for cache hit
    ]

    # Parity check
    base_preds = base_predictor.predict(test_queries)
    opt_preds = opt_predictor.predict(test_queries)
    np.testing.assert_array_equal(base_preds, opt_preds)

    base_probas = base_predictor.predict_proba(test_queries)
    opt_probas = opt_predictor.predict_proba(test_queries)
    np.testing.assert_allclose(base_probas, opt_probas, rtol=1e-5)

    # Telemetry
    stats = opt_predictor.stats()
    assert stats["total_requests"] >= 2
    assert "avg_latency_ms" in stats


def test_onnx_export_and_predictor(sentiment_corpus: pd.DataFrame, tmp_path: Path) -> None:
    """Verify ONNX export and ONNXNLPPredictor execution."""
    base_predictor, _ = train_baseline(
        data=sentiment_corpus,
        text_column="text",
        target_column="label",
        model_name="LogisticRegression",
    )

    onnx_path = tmp_path / "model.onnx"
    export_nlp_to_onnx(base_predictor, onnx_path)
    assert onnx_path.exists()

    onnx_predictor = ONNXNLPPredictor(base_predictor=base_predictor, onnx_model_path=onnx_path)
    assert isinstance(onnx_predictor, NLPPredictorProtocol)

    preds = onnx_predictor.predict(["Exceptional quality!"])
    assert len(preds) == 1
    assert preds[0] in ("positive", "negative")
