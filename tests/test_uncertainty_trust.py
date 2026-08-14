"""Unit & Integration tests for Conformal Prediction, OOD Detection & Trust Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from dive.ood_detector import OODDetector, OODResult
from dive.trust import TrustEngine, TrustReport
from dive.uncertainty import ConformalPredictor, UncertaintyDecomposition


def test_conformal_regression_intervals() -> None:
    # Generate linear synthetic data
    np.random.seed(42)
    y_true_cal = np.linspace(10, 50, 20) + np.random.normal(0, 1.0, 20)
    y_pred_cal = np.linspace(10, 50, 20)

    conformal = ConformalPredictor(problem_type="regression", confidence_level=0.90)
    conformal.calibrate(y_true_cal, y_pred_cal)

    assert conformal.is_calibrated_ is True
    assert conformal.quantile_q_ is not None
    assert conformal.quantile_q_ > 0.0

    # Predict intervals on new test points
    y_test_preds = np.array([25.0, 35.0, 45.0])
    res = conformal.predict_interval(y_test_preds)

    assert len(res.lower_bounds) == 3
    assert len(res.upper_bounds) == 3
    assert np.all(res.lower_bounds < res.predictions)
    assert np.all(res.upper_bounds > res.predictions)


def test_conformal_classification_sets() -> None:
    y_true_cal = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    # Calibrated probabilities for binary classification
    y_pred_cal = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.85, 0.15],
        [0.15, 0.85],
        [0.75, 0.25],
        [0.25, 0.75],
    ])

    conformal = ConformalPredictor(problem_type="classification", confidence_level=0.90)
    conformal.calibrate(y_true_cal, y_pred_cal)

    assert conformal.is_calibrated_ is True
    assert conformal.quantile_q_ is not None

    test_probs = np.array([[0.95, 0.05], [0.51, 0.49]])
    res = conformal.predict_set(test_probs)

    assert len(res.prediction_sets) == 2
    assert isinstance(res.prediction_sets[0], list)


def test_uncertainty_decomposition() -> None:
    # Ensemble of 5 models on 4 samples
    ensemble_preds = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [10.2, 19.8, 30.5, 39.5],
        [9.8, 20.1, 29.5, 40.5],
        [10.1, 19.9, 30.2, 39.8],
        [9.9, 20.2, 29.8, 40.2],
    ])

    decomp = ConformalPredictor.decompose_uncertainty(ensemble_preds, problem_type="regression")

    assert len(decomp.epistemic_uncertainty) == 4
    assert len(decomp.aleatoric_uncertainty) == 4
    assert len(decomp.total_uncertainty) == 4
    assert np.all(decomp.total_uncertainty > 0)


def test_ood_detector() -> None:
    np.random.seed(42)
    # In-distribution training data around mean=0, std=1
    X_train = pd.DataFrame(np.random.normal(0, 1, size=(50, 4)), columns=["f1", "f2", "f3", "f4"])

    detector = OODDetector()
    detector.fit(X_train)

    # In-distribution test point
    X_in = pd.DataFrame(np.random.normal(0, 1, size=(5, 4)), columns=["f1", "f2", "f3", "f4"])
    res_in = detector.score(X_in)
    assert res_in.mean_ood_score < 0.8

    # Extreme anomaly / OOD test points (mean=100)
    X_out = pd.DataFrame(np.ones((5, 4)) * 100.0, columns=["f1", "f2", "f3", "f4"])
    res_out = detector.score(X_out)
    assert res_out.mean_ood_score > res_in.mean_ood_score
    assert any(s in ("OOD", "LOW_CONFIDENCE") for s in res_out.status_labels)


def test_trust_engine_audit() -> None:
    X_test = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "category": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
    })
    y_test = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    y_proba = np.array([
        [0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.95, 0.05],
        [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.05, 0.95],
        [0.8, 0.2], [0.75, 0.25], [0.25, 0.75], [0.3, 0.7],
    ])

    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_test, y_test)

    trust_engine = TrustEngine(problem_type="classification")
    report = trust_engine.audit(dummy_model, X_test, y_test, y_pred, y_proba)

    assert isinstance(report, TrustReport)
    assert 0.0 <= report.trust_score <= 100.0
    assert report.trust_grade in ("A+", "A", "B", "C", "F")
    assert len(report.render()) > 0
