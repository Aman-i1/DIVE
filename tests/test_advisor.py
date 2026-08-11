"""Tests for ValidationAdvisor and ModelAdvisor."""

from __future__ import annotations

import pandas as pd
import pytest

from dive.advisor import ModelAdvisor, ValidationAdvisor, ValidationAdvice


def test_validation_advisor_group() -> None:
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4"],
        "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 1, 1, 0, 0, 1, 1],
    })
    advisor = ValidationAdvisor()
    advice = advisor.advise(df, target="target", problem_type="classification")

    assert isinstance(advice, ValidationAdvice)
    assert advice.is_random_safe is False
    assert advice.group_column == "customer_id"
    assert "Group" in advice.recommended_strategy


def test_validation_advisor_time() -> None:
    df = pd.DataFrame({
        "event_time": pd.date_range("2023-01-01", periods=10),
        "feature": range(10),
        "target": [0, 1] * 5,
    })
    advisor = ValidationAdvisor()
    advice = advisor.advise(df, target="target", problem_type="classification")

    assert advice.is_random_safe is False
    assert "TimeSeriesSplit" in advice.recommended_strategy


def test_model_advisor() -> None:
    advisor = ModelAdvisor()
    advice = advisor.advise(
        n_samples=100_000,
        n_features=50,
        n_categorical=10,
        has_high_cardinality=True,
        has_missing=False,
    )
    assert "CatBoost" in advice.recommended
    assert "KNN" in advice.rejected
