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


def test_cli_upgrade(cli_runner: CliRunner) -> None:
    res = cli_runner.invoke(cli, ["upgrade"])
    assert res.exit_code == 0
    assert "DIVE AUTO-UPGRADE" in res.output


def test_cli_info(cli_runner: CliRunner, sample_csv: str) -> None:
    res = cli_runner.invoke(cli, ["info", sample_csv])
    assert res.exit_code == 0
    assert "DIVE DATASET INSPECTION" in res.output
    assert "churn" in res.output


def test_pdf_report_generation(tmp_path: Path, sample_csv: str) -> None:
    from dive.core import Dive
    from dive.reporting import generate_pdf_report

    df = pd.read_csv(sample_csv)
    model = Dive(target="churn", mode="fast")
    model.fit(df)

    pdf_file = tmp_path / "test_report.pdf"
    res_path = generate_pdf_report(model, pdf_file)
    assert res_path is not None
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 0


def test_cli_audit(cli_runner: CliRunner, sample_csv: str, tmp_path: Path) -> None:
    cert_pdf = tmp_path / "cert.pdf"
    res = cli_runner.invoke(cli, ["audit", sample_csv, "--target", "churn", "--output", str(cert_pdf)])
    assert res.exit_code == 0
    assert "DIVE COMPLIANCE AND RELIABILITY AUDITOR" in res.output
    assert cert_pdf.exists()


def test_cli_export_and_gate(cli_runner: CliRunner, sample_csv: str, tmp_path: Path) -> None:
    from dive.core import Dive

    df = pd.read_csv(sample_csv)
    model = Dive(target="churn", mode="fast")
    model.fit(df)

    model_pkl = tmp_path / "model.pkl"
    model.save(model_pkl)

    onnx_file = tmp_path / "model.onnx"
    res_export = cli_runner.invoke(cli, ["export", str(model_pkl), "--output", str(onnx_file)])
    assert res_export.exit_code == 0

    res_gate = cli_runner.invoke(cli, ["gate", str(model_pkl), "--data", sample_csv])
    assert res_gate.exit_code == 0
    assert "DEPLOYMENT APPROVED" in res_gate.output




