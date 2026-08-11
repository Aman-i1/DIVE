"""Tests for ModelFailureAnalyzer and ProbabilityCalibrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dive.calibration import ProbabilityCalibrator
from dive.failure_analysis import ModelFailureAnalyzer


def test_model_failure_analyzer() -> None:
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    analyzer = ModelFailureAnalyzer(problem_type="classification")
    res = analyzer.analyze(y_true, y_pred)

    assert res.problem_type == "classification"
    assert "Accuracy" in res.overall_metrics
    assert res.confusion_matrix_data is not None


def test_probability_calibrator() -> None:
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.4, 0.3, 0.6, 0.5, 0.7, 0.8, 0.9, 0.95])

    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(y_true, y_proba)
    cal_p = calibrator.calibrate(y_proba)

    assert len(cal_p) == len(y_proba)
    report = calibrator.evaluate(y_true, y_proba)
    assert report.brier_after <= report.brier_before + 0.05
