"""Unit & Integration tests for Phase 2: Data Intelligence & Validation Intelligence Engine."""

from __future__ import annotations

import pandas as pd
import pytest

from dive.data_intelligence import DataIntelligence
from dive.validation_engine import ValidationIntelligenceEngine, ValidationPlan


def test_data_intelligence_structure() -> None:
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="D"),
        "amount": [10.5, 20.0, 15.0, 30.0, 50.0, 60.0],
        "churn": [0, 0, 1, 1, 0, 1],
    })

    di = DataIntelligence(target="churn")
    profile = di.analyze(df)

    assert "semantic_types" in profile
    assert "dataset_structure" in profile
    assert profile["semantic_types"]["user_id"] == ["identifier"]
    assert profile["dataset_structure"]["is_panel"] is True


def test_validation_engine_entity_leakage() -> None:
    df = pd.DataFrame({
        "customer_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
        "feature1": [1, 2, 3, 4, 5, 6],
        "target": [0, 0, 1, 1, 0, 1],
    })

    engine = ValidationIntelligenceEngine(target="target")
    plan = engine.evaluate(df, problem_type="classification")

    assert "Group" in plan.strategy
    assert plan.group_column == "customer_id"
    assert plan.risk_assessment.risk_level in ("MEDIUM", "HIGH")
    assert any("customer_id" in r for r in plan.risk_assessment.reasons)


def test_validation_engine_temporal_leakage() -> None:
    df = pd.DataFrame({
        "created_at": pd.date_range("2026-01-01", periods=6, freq="D"),
        "feature1": [1, 2, 3, 4, 5, 6],
        "target": [0, 0, 1, 1, 0, 1],
    })

    engine = ValidationIntelligenceEngine(target="target")
    plan = engine.evaluate(df, problem_type="classification")

    assert plan.strategy == "TimeSeriesSplit"
    assert plan.time_column == "created_at"
    assert any("created_at" in r for r in plan.risk_assessment.reasons)
