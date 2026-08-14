"""Unit & Integration tests for Contamination, Adversarial Validation, Stress Testing & Segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from dive.adversarial_validation import AdversarialValidationReport, AdversarialValidator
from dive.contamination import ContaminationDetector, ContaminationReport
from dive.failure_segments import FailureSegmentAnalyzer, FailureSegmentsReport
from dive.model_stress import ModelStressTester, StressTestReport


def test_contamination_detection() -> None:
    train_df = pd.DataFrame({
        "cust_id": ["C1", "C2", "C3", "C4"],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1],
    })
    # Val df shares C2 with train (entity contamination) and identical row for C3
    val_df = pd.DataFrame({
        "cust_id": ["C2", "C3", "C5"],
        "f1": [2.0, 3.0, 5.0],
        "target": [1, 0, 1],
    })

    detector = ContaminationDetector()
    report = detector.audit_splits(train_df, val_df, target_column="target", entity_column="cust_id")

    assert isinstance(report, ContaminationReport)
    assert report.entity_overlap_count == 2  # C2 and C3
    assert report.contamination_risk in ("HIGH", "MEDIUM")


def test_adversarial_validation() -> None:
    np.random.seed(42)
    # Train is Gaussian(0, 1)
    train_df = pd.DataFrame({"f1": np.random.normal(0, 1, 100), "target": [0]*100})
    # Target distribution is heavily shifted Gaussian(5, 1)
    target_df = pd.DataFrame({"f1": np.random.normal(5, 1, 100), "target": [0]*100})

    validator = AdversarialValidator()
    report = validator.evaluate_shift(train_df, target_df, target_column="target")

    assert isinstance(report, AdversarialValidationReport)
    # Should easily distinguish distributions (AUC > 0.80)
    assert report.adversarial_auc > 0.80
    assert report.shift_status == "SEVERE_COVARIATE_SHIFT"
    assert len(report.top_drift_features) > 0


def test_model_stress_and_target_permutation() -> None:
    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    tester = ModelStressTester(problem_type="binary_classification")
    report = tester.run_stress_suite(model, X, y, nominal_score=0.50)

    assert isinstance(report, StressTestReport)
    assert report.permutation_sanity_status == "PASS"


def test_failure_segment_discovery() -> None:
    # 50 rows, region 'North' has high accuracy, 'South' has 0% accuracy
    X = pd.DataFrame({
        "region": ["North"] * 25 + ["South"] * 25,
        "feat1": np.random.normal(0, 1, 50),
    })
    y_true = np.array([1] * 25 + [1] * 25)
    # Model predicts correctly for North (1), but fails completely for South (0)
    y_pred = np.array([1] * 25 + [0] * 25)

    analyzer = FailureSegmentAnalyzer(min_sample_count=20, drop_threshold=0.20)
    report = analyzer.discover_weak_segments(X, y_true, y_pred, metric_name="Accuracy")

    assert isinstance(report, FailureSegmentsReport)
    assert len(report.weak_segments) >= 1
    assert "South" in report.weak_segments[0].segment_description
    assert report.overall_segment_status in ("FAIL", "WARNING")
