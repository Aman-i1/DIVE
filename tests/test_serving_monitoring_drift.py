"""Tests for Serving, Monitoring, and Drift Detection."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from dive.drift import DriftDetector
from dive.monitoring import ModelMonitor


def test_drift_detector() -> None:
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "num": np.random.normal(0, 1, 100),
        "cat": np.random.choice(["A", "B"], 100),
    })
    # Shift current production distribution
    curr_df = pd.DataFrame({
        "num": np.random.normal(5, 1, 100),  # Drifting
        "cat": np.random.choice(["A", "B"], 100),
    })

    detector = DriftDetector()
    report = detector.analyze_drift(ref_df, curr_df)

    assert report.n_features_analyzed == 2
    assert report.n_drifting_features >= 1
    num_report = [fr for fr in report.feature_reports if fr.feature == "num"][0]
    assert num_report.drift_status == "SIGNIFICANT_DRIFT"


def test_model_monitor() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        monitor = ModelMonitor(log_dir=tmp_dir)
        monitor.log_inference(prediction_val=1, latency_ms=12.5)
        monitor.log_inference(prediction_val=0, latency_ms=15.0)
        monitor.flush()

        summary = monitor.get_summary()
        assert summary.total_requests == 2
        assert summary.avg_latency_ms > 0
        assert "1" in summary.prediction_distribution


def test_get_help_documentation() -> None:
    from dive.predictor import DivePredictor
    from dive.serving import get_help_documentation

    predictor = DivePredictor(
        model_name="MockModel",
        estimator=None,
        feature_engineer=None,
        feature_columns=["f1", "f2"],
        label_encoder=None,
        target="target",
        problem_type="classification",
        input_schema={"required_columns": ["f1", "f2"], "example_row": {"f1": 1.0, "f2": 2.0}},
    )

    doc = get_help_documentation(predictor, host="127.0.0.1", port=8000)
    assert doc["service"] == "DIVE Production REST Model Server"
    assert doc["model_name"] == "MockModel"
    assert "GET /help" in doc["endpoints"]
    assert "POST /predict" in doc["endpoints"]
    assert "curl_examples" in doc
    assert "python_usage_example" in doc
