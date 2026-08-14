"""Unit & Integration tests for Calibrated Stacking, Convex Blending & Dynamic Routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from dive.inference_router import DynamicInferenceRouter, RoutedPredictionResult
from dive.stacking_calibrated import CalibratedStackingEnsemble, EnsembleWeightsResult


def test_convex_blending_weights() -> None:
    # 20 samples binary classification
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Model 1 has high accuracy
    p1 = np.where(y_true == 1, 0.9, 0.1)
    # Model 2 is noisy
    p2 = np.random.uniform(0.3, 0.7, size=20)

    ensemble = CalibratedStackingEnsemble(problem_type="classification", method="blend")
    res = ensemble.fit_blend({"m1": p1, "m2": p2}, y_true)

    assert isinstance(res, EnsembleWeightsResult)
    assert len(res.weights) == 2
    # Weights must sum to 1.0 (convexity)
    assert np.sum(res.weights) == pytest.approx(1.0, abs=1e-3)
    # Model 1 should receive higher weight than noisy Model 2
    assert res.weights[0] > res.weights[1]

    # Predict
    test_preds = ensemble.predict({"m1": p1, "m2": p2})
    assert len(test_preds) == 20
    test_probs = ensemble.predict_proba({"m1": p1, "m2": p2})
    assert test_probs.shape == (20, 2)


def test_dynamic_inference_router() -> None:
    X_test = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })

    # Fast estimator with sharp confidence for some samples, uncertain for others
    class MockFastEstimator:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([0, 0, 1, 1, 0, 1])

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            # First 3 samples confident, last 3 uncertain (near 0.5)
            return np.array([
                [0.95, 0.05],
                [0.92, 0.08],
                [0.05, 0.95],
                [0.55, 0.45],  # uncertain -> route to ensemble
                [0.52, 0.48],  # uncertain -> route to ensemble
                [0.48, 0.52],  # uncertain -> route to ensemble
            ])

    class MockEnsembleEstimator:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.ones(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    router = DynamicInferenceRouter(
        fast_estimator=MockFastEstimator(),
        ensemble_estimator=MockEnsembleEstimator(),
        confidence_threshold=0.85,
        problem_type="classification",
    )

    result = router.predict(X_test)
    assert isinstance(result, RoutedPredictionResult)
    assert len(result.predictions) == 6
    assert result.routing_decisions[:3] == ["FAST_PATH", "FAST_PATH", "FAST_PATH"]
    assert result.routing_decisions[3:] == ["ENSEMBLE_PATH", "ENSEMBLE_PATH", "ENSEMBLE_PATH"]
    assert result.pct_fast_path == 50.0
    assert result.pct_ensemble_path == 50.0
