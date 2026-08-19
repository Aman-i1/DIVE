"""Phase 13 Tests - DIVE NLP CLI Command Integration.

Verifies:
1. Root dive CLI discovers and registers the 'nlp' command group.
2. 'dive nlp profile' command execution on tabular CSV data.
3. 'dive nlp train' autonomous candidate exploration and predictor artifact serialization.
4. 'dive nlp monitor' text distribution drift detection between baseline and production batches.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from dive.cli import cli


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Fixture providing sample review CSV file."""
    csv_file = tmp_path / "reviews.csv"
    df = pd.DataFrame(
        {
            "review": [
                "Exceptional build quality and super fast shipping, loved it!",
                "Amazing performance, totally exceeded my expectations, great product!",
                "Delightful customer support, great onboarding and seamless setup.",
                "Awesome durability, loved everything about this device, top quality.",
                "Worst purchase ever made, broken on arrival, terrible item.",
                "Terrible customer service, completely unhelpful and rude, awful.",
                "Defective unit, useless customer support, refund was refused, terrible.",
                "Awful product, broke within one hour, completely useless and worst.",
            ],
            "sentiment": [
                "pos",
                "pos",
                "pos",
                "pos",
                "neg",
                "neg",
                "neg",
                "neg",
            ],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_cli_nlp_registered() -> None:
    """Verify that 'dive nlp' and 'dive ml' are registered in root CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "nlp" in result.output
    assert "ml" in result.output
    assert "Tabular & Structured Data" in result.output


def test_cli_ml_help_subcommands() -> None:
    """Verify 'dive ml --help' lists all ML subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ml", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "auto" in result.output
    assert "doctor" in result.output
    assert "predict" in result.output


def test_cli_nlp_help_subcommands() -> None:
    """Verify 'dive nlp --help' lists all NLP subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["nlp", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "profile" in result.output
    assert "serve" in result.output
    assert "monitor" in result.output


def test_cli_nlp_profile(sample_csv: Path) -> None:
    """Verify 'dive nlp profile' command output."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nlp",
            "profile",
            str(sample_csv),
            "--text-col",
            "review",
            "--target-col",
            "sentiment",
        ],
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert "DIVE NLP DATASET PROFILE REPORT" in result.output
    assert "DOCUMENT LENGTH DISTRIBUTION" in result.output


def test_cli_nlp_train(sample_csv: Path, tmp_path: Path) -> None:
    """Verify 'dive nlp train' runs AutoNLP and writes model artifact."""
    out_model = tmp_path / "nlp_champion.pkl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nlp",
            "train",
            str(sample_csv),
            "--text-col",
            "review",
            "--target-col",
            "sentiment",
            "--trials",
            "2",
            "--output",
            str(out_model),
        ],
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert "DIVE AUTONLP MODEL SELECTION LEADERBOARD" in result.output
    assert out_model.exists()


def test_cli_nlp_monitor(sample_csv: Path, tmp_path: Path) -> None:
    """Verify 'dive nlp monitor' evaluates drift between reference and current batches."""
    curr_csv = tmp_path / "current.csv"
    pd.DataFrame(
        {
            "review": [
                "Loved the build quality and rapid shipping.",
                "Terrible broken product on arrival.",
            ]
        }
    ).to_csv(curr_csv, index=False)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nlp",
            "monitor",
            str(sample_csv),
            str(curr_csv),
            "--text-col",
            "review",
        ],
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert "DIVE NLP DRIFT & MONITORING REPORT" in result.output
