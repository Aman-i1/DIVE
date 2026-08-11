"""Tests for Advanced Data Leakage Detection & Point-in-Time Validation."""

from __future__ import annotations

import pandas as pd
import pytest

from dive.leakage import AdvancedLeakageDetector, LeakageReport


@pytest.fixture
def leakage_df() -> pd.DataFrame:
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], name="target")
    # Feature 1 is identical to target (perfect leakage)
    # Feature 2 is normal signal
    # Feature 3 is named suspiciously
    df = pd.DataFrame({
        "perfect_leak": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "normal_feat": [10, 12, 11, 15, 20, 22, 21, 25, 26, 29],
        "post_event_result": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    return df, y


def test_leakage_detector(leakage_df) -> None:
    df, y = leakage_df
    detector = AdvancedLeakageDetector()
    report = detector.audit(df, y, problem_type="classification")

    assert isinstance(report, LeakageReport)
    assert report.has_high_risk is True
    assert len(report.warnings) >= 2

    # Check high risk warning for perfect_leak
    high_risk = [w for w in report.warnings if w.feature == "perfect_leak"]
    assert len(high_risk) > 0
    assert high_risk[0].risk_level == "HIGH"

    # Check name pattern warning for post_event_result
    name_warn = [w for w in report.warnings if w.feature == "post_event_result"]
    assert len(name_warn) > 0


def test_point_in_time_validation() -> None:
    df = pd.DataFrame({
        "signup_time": ["2023-01-01", "2023-01-01", "2023-01-01"],
        "future_event": ["2023-01-05", "2023-01-10", "2023-02-01"],
        "past_event": ["2022-12-01", "2022-12-15", "2022-12-20"],
    })
    y = pd.Series([0, 1, 0])
    detector = AdvancedLeakageDetector()
    report = detector.audit(
        df, y,
        prediction_time_col="signup_time",
        feature_time_cols=["future_event", "past_event"],
    )
    assert report.point_in_time_status.get("future_event") == "FUTURE_INFORMATION"
    assert report.point_in_time_status.get("past_event") == "AVAILABLE"
