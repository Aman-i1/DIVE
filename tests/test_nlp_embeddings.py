"""Phase 7 Tests - DIVE NLP Dense Vector Embeddings, Caching & Benchmarking.

Verifies:
1. EmbeddingCache two-tier (memory + disk) storage, hashing, and batch retrieval.
2. EmbeddingRepresentation dense vector encoding, dimension consistency, and L2 normalization.
3. Cosine semantic similarity matrix computation.
4. NLPPipeline and train_baseline() integration with dense embeddings.
5. benchmark_tfidf_vs_embeddings() empirical comparative analysis.
6. NLPRepresentationProtocol compliance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from dive.nlp.embeddings import (
    EmbeddingCache,
    EmbeddingRepresentation,
    benchmark_tfidf_vs_embeddings,
    compute_semantic_similarity,
)
from dive.nlp.interfaces import NLPRepresentationProtocol
from dive.nlp.pipeline import NLPPipeline
from dive.nlp.models.baselines import build_baseline_model
from dive.nlp.training import train_baseline


@pytest.fixture
def sentiment_data() -> pd.DataFrame:
    """Fixture with 12 distinct sentiment texts."""
    return pd.DataFrame(
        {
            "text": [
                "Incredible top tier build quality, loved the product!",
                "Fast shipping, highly recommend this awesome store.",
                "Delightful customer support, great experience overall.",
                "Superb durability and performance under load, great value.",
                "Loved everything about this, exceeded all my expectations.",
                "Great purchase, awesome quality and sleek modern design.",
                "Worst purchase ever made, broken on arrival, terrible.",
                "Terrible customer service, completely useless and rude.",
                "Defective unit, awful shipping, broke immediately.",
                "Horrible experience, refund was refused, worst store.",
                "Awful product, broke within one hour of use, terrible.",
                "Waste of money, useless device, completely broken.",
            ],
            "label": [
                "positive",
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
                "negative",
            ],
        }
    )


def test_embedding_cache_memory_and_disk(tmp_path: Path) -> None:
    """Verify caching of vectors in RAM and persistent on disk."""
    cache = EmbeddingCache(cache_dir=tmp_path / "embeddings")

    text1 = "Natural language processing with dense embeddings."
    text2 = "Machine learning model optimization."
    vec1 = np.random.randn(384).astype(np.float32)
    vec2 = np.random.randn(384).astype(np.float32)

    # 1. Store
    cache.set(text1, vec1)
    cache.set(text2, vec2)
    assert len(cache) == 2

    # 2. Retrieve
    retrieved = cache.get(text1)
    assert retrieved is not None
    np.testing.assert_allclose(retrieved, vec1)

    # 3. Batch retrieval
    texts = [text1, "Unseen text", text2]
    hits, missing = cache.batch_get(texts)
    assert 0 in hits
    assert 2 in hits
    assert missing == [1]

    # 4. Disk persistence check
    new_cache = EmbeddingCache(cache_dir=tmp_path / "embeddings")
    disk_vec = new_cache.get(text1)
    assert disk_vec is not None
    np.testing.assert_allclose(disk_vec, vec1)


def test_embedding_representation_lifecycle() -> None:
    """Verify EmbeddingRepresentation fitting, transform, and normalization."""
    rep = EmbeddingRepresentation(dimension=128, normalize_embeddings=True)
    assert isinstance(rep, NLPRepresentationProtocol)

    texts = [
        "First document for neural embedding test.",
        "Second document discussing machine learning pipelines.",
    ]

    embs = rep.fit_transform(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 128)

    # Verify L2 normalization
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)

    # Feature properties
    assert rep.n_features_ == 128
    assert len(rep.feature_names_) == 128


def test_compute_semantic_similarity() -> None:
    """Verify cosine semantic similarity matrix computation."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])  # Identical
    vec3 = np.array([0.0, 1.0, 0.0])  # Orthogonal

    sim_same = compute_semantic_similarity(vec1, vec2)
    assert np.isclose(float(sim_same[0, 0]), 1.0)

    sim_ortho = compute_semantic_similarity(vec1, vec3)
    assert np.isclose(float(sim_ortho[0, 0]), 0.0)


def test_pipeline_and_training_with_embeddings(sentiment_data: pd.DataFrame) -> None:
    """Verify NLPPipeline and train_baseline() with dense neural embeddings."""
    rep = EmbeddingRepresentation(dimension=128, use_cache=True)
    estimator = build_baseline_model("LogisticRegression")

    pipeline = NLPPipeline(
        estimator=estimator,
        representation=rep,
        task_type="text_classification",
        model_name="EmbeddingPipeline",
    )
    pipeline.fit(sentiment_data["text"], sentiment_data["label"])

    preds = pipeline.predict(["Loved the product and fast shipping!"])
    assert len(preds) == 1
    assert preds[0] in ("positive", "negative")

    probas = pipeline.predict_proba(["Loved the product!"])
    assert probas.shape == (1, 2)
    assert np.isclose(np.sum(probas), 1.0)

    # train_baseline convenience integration
    predictor, metrics = train_baseline(
        data=sentiment_data,
        text_column="text",
        target_column="label",
        model_name="LogisticRegression",
        representation_type="embedding",
        test_size=0.25,
        random_state=42,
    )
    assert predictor.model_name == "LogisticRegression"
    assert "accuracy" in metrics
    assert "macro_f1" in metrics


def test_benchmark_tfidf_vs_embeddings(sentiment_data: pd.DataFrame) -> None:
    """Verify empirical comparative benchmarking between TF-IDF and embeddings."""
    report = benchmark_tfidf_vs_embeddings(
        data=sentiment_data,
        text_column="text",
        target_column="label",
        model_name="LogisticRegression",
        task_type="text_classification",
        test_size=0.25,
        random_state=42,
    )

    assert "recommended_representation" in report
    assert report["recommended_representation"] in ("tfidf", "embeddings")
    assert "tfidf" in report
    assert "embeddings" in report

    # Verify metric and latency tracking
    assert "inference_latency_ms" in report["tfidf"]
    assert "inference_latency_ms" in report["embeddings"]
    assert "train_time_ms" in report["tfidf"]
    assert "train_time_ms" in report["embeddings"]
