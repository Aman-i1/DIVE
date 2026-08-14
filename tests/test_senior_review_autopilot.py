"""Unit & Integration tests for Senior Review Engine, Autopilot Orchestrator & CLI Commands."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from dive.autopilot import AutopilotOrchestrator, AutopilotResult
from dive.cli import cli
from dive.prediction_contract import PredictionContract
from dive.senior_review import SeniorReviewEngine, SeniorReviewReport


def test_senior_review_synthesis() -> None:
    contract = PredictionContract(
        target="churn",
        problem_type="binary_classification",
        entity="cust_id",
    )

    engine = SeniorReviewEngine()
    report = engine.review(
        contract=contract,
        champion_model="CalibratedStack",
        primary_score=0.91,
    )

    assert isinstance(report, SeniorReviewReport)
    assert report.final_decision == "PASS"
    assert report.champion_model == "CalibratedStack"
    assert report.confidence == "HIGH"
    rendered = report.render()
    assert "DIVE SENIOR ML REVIEW" in rendered
    assert "FINAL DECISION" in rendered


def test_autopilot_orchestration_flow(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(40)],
        "tenure": np.random.uniform(1, 100, 40),
        "monthly_charges": np.random.uniform(20, 150, 40),
        "churn": np.random.choice([0, 1], 40),
    })

    out_dir = tmp_path / "autopilot_test_out"

    orchestrator = AutopilotOrchestrator(
        target="churn",
        mode="fast",
        time_budget="30s",
        entity_column="customer_id",
        output_dir=out_dir,
    )

    result = orchestrator.run(df)

    assert isinstance(result, AutopilotResult)
    assert result.contract.target == "churn"
    assert result.senior_review.final_decision in ("PASS", "PASS_WITH_WARNINGS")
    assert out_dir.exists()


def test_cli_contract_and_review(tmp_path: Path) -> None:
    runner = CliRunner()
    csv_file = tmp_path / "telecom.csv"
    csv_file.write_text("cust_id,tenure,churn\n1,10,0\n2,20,1\n3,30,0\n4,40,1\n5,50,0\n6,60,1\n", encoding="utf-8")

    # 1. Test dive contract
    contract_out = tmp_path / "contract.json"
    res_contract = runner.invoke(
        cli,
        [
            "contract",
            str(csv_file),
            "--target",
            "churn",
            "--entity",
            "cust_id",
            "--output",
            str(contract_out),
        ],
    )
    assert res_contract.exit_code == 0
    assert "FORMAL PREDICTION CONTRACT" in res_contract.output
    assert contract_out.exists()

    # 2. Test dive review
    review_out = tmp_path / "review.json"
    res_review = runner.invoke(
        cli,
        [
            "review",
            str(csv_file),
            "--target",
            "churn",
            "--entity",
            "cust_id",
            "--output",
            str(review_out),
        ],
    )
    assert res_review.exit_code == 0
    assert "DIVE SENIOR ML REVIEW" in res_review.output
    assert review_out.exists()
