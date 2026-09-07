"""Integration tests for the DIVE Deep Learning CLI domain (dive dl)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from dive.cli import cli


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "feature1": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            "feature2": [10.0, 20.0, 15.0, 30.0, 25.0, 35.0, 40.0, 50.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    csv_file = tmp_path / "dl_sample.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_cli_dl_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["dl", "--help"])
    assert result.exit_code == 0
    assert "Deep Learning capability domain" in result.output
    for cmd in ("info", "train", "predict", "doctor", "benchmark", "auto"):
        assert cmd in result.output


def test_cli_dl_doctor():
    runner = CliRunner()
    result = runner.invoke(cli, ["dl", "doctor"])
    assert result.exit_code == 0
    assert "DIVE DL PRE-FLIGHT DOCTOR AUDIT" in result.output
    assert "HARDWARE & BACKENDS" in result.output
    assert "MODALITY DEPENDENCIES" in result.output


def test_cli_dl_info(sample_csv: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["dl", "info", str(sample_csv), "--target-col", "target"])
    assert result.exit_code == 0
    assert "DIVE DL DATASET INSPECTOR" in result.output
    assert "Total samples" in result.output
    assert "HARDWARE SIZING" in result.output


def test_cli_dl_train_and_predict(sample_csv: Path, tmp_path: Path):
    model_path = tmp_path / "dl_test_model.pkl"
    runner = CliRunner()

    # 1. Train
    train_result = runner.invoke(
        cli,
        [
            "dl",
            "train",
            str(sample_csv),
            "--target-col",
            "target",
            "--epochs",
            "3",
            "--output",
            str(model_path),
            "--no-torch",
        ],
    )
    assert train_result.exit_code == 0
    assert model_path.is_file()

    # 2. Predict single item via terminal
    pred_result = runner.invoke(
        cli,
        ["dl", "predict", str(model_path), "--data", str(sample_csv), "--proba"],
    )
    assert pred_result.exit_code == 0
    assert "BATCH PREDICTION SUMMARY" in pred_result.output
    assert "Total scored" in pred_result.output


def test_cli_dl_benchmark(sample_csv: Path, tmp_path: Path):
    model_path = tmp_path / "dl_bench_model.pkl"
    runner = CliRunner()

    # Train model first
    r_train = runner.invoke(
        cli,
        [
            "dl",
            "train",
            str(sample_csv),
            "--target-col",
            "target",
            "--epochs",
            "2",
            "--output",
            str(model_path),
            "--no-torch",
        ],
    )
    assert r_train.exit_code == 0

    # Benchmark
    r_bench = runner.invoke(cli, ["dl", "benchmark", str(model_path), "--samples", "10"])
    assert r_bench.exit_code == 0
    assert "DIVE DL LATENCY & THROUGHPUT BENCHMARK" in r_bench.output
    assert "p50 (median)" in r_bench.output


def test_cli_dl_auto(sample_csv: Path, tmp_path: Path):
    champ_path = tmp_path / "auto_champ.pkl"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "dl",
            "auto",
            str(sample_csv),
            "--target-col",
            "target",
            "--trials",
            "2",
            "--mode",
            "fast",
            "--output",
            str(champ_path),
        ],
    )
    assert result.exit_code == 0
    assert "DIVE AUTODL MODEL SELECTION LEADERBOARD" in result.output
    assert champ_path.is_file()
