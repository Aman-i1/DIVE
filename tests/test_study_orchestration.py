"""Unit & Integration tests for Phase 1: Core Domain Abstractions, Decision Logging & Study Orchestrator."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from dive.decisions import DecisionLogger, DecisionRecord
from dive.orchestration import StudyConfig, StudyOrchestrator
from dive.study import Study, create_study


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50],
        "income": [30000, 45000, 60000, 75000, 90000, 105000],
        "churn": [0, 0, 1, 1, 0, 1],
    })


def test_decision_logger() -> None:
    logger = DecisionLogger()
    record = logger.log(
        component="ValidationEngine",
        decision="StratifiedKFold(n_splits=5)",
        reason="Binary target with no temporal ordering",
        confidence=0.95,
        evidence={"n_splits": 5},
    )
    assert len(logger.records) == 1
    assert record.component == "ValidationEngine"
    assert record.confidence == 0.95
    assert "StratifiedKFold" in logger.render_summary()


def test_study_flow(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    study = create_study(
        data=sample_df,
        target="churn",
        mode="fast",
        time_budget="300s",
        output_dir=tmp_path,
    )
    study.fit()

    assert study.result is not None
    assert "best_model_name" in study.result
    assert len(study.decisions) >= 3

    preds = study.predict(sample_df)
    assert len(preds) == len(sample_df)
