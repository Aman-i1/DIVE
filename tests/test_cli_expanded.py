"""CLI integration tests for new subcommands: doctor, experiments, models, drift, reproduce, benchmark."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from click.testing import CliRunner

from dive.cli import cli


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "income": [20000, 40000, 60000, 80000, 100000],
        "churn": [0, 0, 1, 1, 0],
    })
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_cli_doctor(cli_runner: CliRunner, sample_csv: str) -> None:
    res = cli_runner.invoke(cli, ["doctor", sample_csv, "--target", "churn"])
    assert res.exit_code == 0
    assert "DIVE ML DOCTOR" in res.output
    assert "PRODUCTION READINESS SCORE" in res.output


def test_cli_experiments(cli_runner: CliRunner) -> None:
    res = cli_runner.invoke(cli, ["experiments", "list"])
    assert res.exit_code == 0


def test_cli_models(cli_runner: CliRunner) -> None:
    res = cli_runner.invoke(cli, ["models", "list"])
    assert res.exit_code == 0


def test_cli_drift(cli_runner: CliRunner, sample_csv: str) -> None:
    res = cli_runner.invoke(cli, ["drift", "--ref", sample_csv, "--curr", sample_csv])
    assert res.exit_code == 0
    assert "DATA & PREDICTION DRIFT REPORT" in res.output


def test_cli_benchmark(cli_runner: CliRunner) -> None:
    res = cli_runner.invoke(cli, ["benchmark", "--mode", "fast"])
    assert res.exit_code == 0
    assert "DIVE Scalability & Performance Benchmarking" in res.output
