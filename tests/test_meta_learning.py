"""Unit & Integration tests for Dataset Fingerprinting & Meta-Learning Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dive.meta_learning import DatasetFingerprint, MetaLearningEngine, MetaWarmStartPriors


def test_meta_learning_fingerprint() -> None:
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    })

    engine = MetaLearningEngine()
    X = df.drop(columns=["target"])
    y = df["target"]

    fingerprint = engine.compute_fingerprint(X, y, problem_type="classification")

    assert fingerprint.n_samples == 10
    assert fingerprint.n_features == 3
    assert fingerprint.n_numeric == 2
    assert fingerprint.n_categorical == 1
    assert fingerprint.dataset_hash != ""
    assert isinstance(fingerprint.landmark_linear_score, float)
    assert isinstance(fingerprint.landmark_tree_stump_score, float)


def test_meta_learning_warm_start_priors() -> None:
    engine = MetaLearningEngine()
    fp = DatasetFingerprint(
        dataset_hash="abc12345",
        n_samples=5000,
        n_features=20,
        sample_to_feature_ratio=250.0,
        n_numeric=15,
        n_categorical=5,
        sparsity=0.0,
        mean_skewness=0.1,
        mean_kurtosis=0.0,
        target_entropy=0.95,
        landmark_linear_score=0.72,
        landmark_tree_stump_score=0.85,
    )

    priors = engine.warm_start_recommendations(fp, problem_type="classification")

    assert len(priors.recommended_model_families) > 0
    assert "LightGBM" in priors.recommended_model_families or "HistGradientBoosting" in priors.recommended_model_families
    assert priors.suggested_initial_learning_rate > 0.0
    assert priors.suggested_max_depth >= 3
