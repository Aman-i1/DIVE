"""Tests for DiveDoctor and ProductionReadinessScore."""

from __future__ import annotations

import pandas as pd
import pytest

from dive.doctor import DiveDoctor, ProductionReadinessScore


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        "customer_id": [f"CUST_{i}" for i in range(10)],
        "target": ["no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
    })


def test_doctor_audit(sample_df: pd.DataFrame) -> None:
    doctor = DiveDoctor(target="target")
    report = doctor.analyze(sample_df)

    assert report.target == "target"
    assert report.n_samples == 10
    assert report.n_features == 3
    assert report.problem_type == "classification"
    assert isinstance(report.readiness_score, ProductionReadinessScore)
    assert report.readiness_score.overall_score > 0.0

    rendered = str(report)
    assert "DIVE ML DOCTOR" in rendered
    assert "PRODUCTION READINESS SCORE" in rendered


def test_doctor_missing_target(sample_df: pd.DataFrame) -> None:
    doctor = DiveDoctor(target="nonexistent")
    with pytest.raises(Exception):
        doctor.analyze(sample_df)
