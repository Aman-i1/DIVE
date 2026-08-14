"""Unit & Integration tests for Prediction Contract & Data Quality with Inferred Rules."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from dive.data_quality import DataQualityEngine, DataQualityReport, InferredRelationalRule
from dive.prediction_contract import PredictionContract, PredictionContractEngine


def test_prediction_contract_inference(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(100)],
        "event_time": pd.date_range("2025-01-01", periods=100, freq="D"),
        "monthly_spend": np.random.uniform(50, 500, 100),
        "churn": np.random.choice([0, 1], 100),
    })

    engine = PredictionContractEngine()
    contract = engine.infer_contract(df, target="churn")

    assert isinstance(contract, PredictionContract)
    assert contract.target == "churn"
    assert contract.problem_type == "binary_classification"
    assert contract.entity == "customer_id"
    assert contract.prediction_time == "event_time"

    # Save and reload
    out_file = tmp_path / "contract.json"
    contract.save(out_file)
    assert out_file.exists()

    loaded = PredictionContract.load(out_file)
    assert loaded.target == "churn"
    assert loaded.entity == "customer_id"


def test_data_quality_and_inferred_rules() -> None:
    df = pd.DataFrame({
        "age": [25, 30, 45, -5, 60],  # 1 negative violation for age
        "purchase_amount": [100.0, 200.0, 150.0, 300.0, 50.0],
        "refund_amount": [20.0, 250.0, 10.0, 0.0, 0.0],  # 1 violation (refund > purchase at index 1)
        "constant_col": [1, 1, 1, 1, 1],
    })

    engine = DataQualityEngine()
    report = engine.audit(df)

    assert isinstance(report, DataQualityReport)
    assert report.constant_columns == ["constant_col"]
    assert len(report.inferred_rules) >= 2

    # Check age rule
    age_rules = [r for r in report.inferred_rules if "age" in r.rule_description]
    assert len(age_rules) > 0
    assert age_rules[0].violations_count == 1

    # Check refund rule
    bounded_rules = [r for r in report.inferred_rules if "<=" in r.rule_description]
    assert len(bounded_rules) > 0
    assert bounded_rules[0].violations_count == 1
