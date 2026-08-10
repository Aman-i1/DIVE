"""End-to-end CLI tests.

These drive the real click commands through ``CliRunner`` against small
synthetic datasets, asserting on artifacts and exit codes rather than on
internal state - the contract a user actually depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from dive.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _train(runner, data_path, out_dir, target="target", extra=None):
    args = [
        "train",
        "--data", str(data_path),
        "--target", target,
        "--mode", "fast",
        "--output", str(out_dir),
    ]
    if extra:
        args.extend(extra)
    return runner.invoke(cli, args, catch_exceptions=False)


# ----------------------------------------------------------------------
# help / version
# ----------------------------------------------------------------------
def test_root_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "predict", "validate", "explain", "report", "docs"):
        assert command in result.output


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


@pytest.mark.parametrize(
    "command", ["train", "predict", "validate", "explain", "report", "docs", "deps"]
)
def test_every_command_has_help(runner, command):
    result = runner.invoke(cli, [command, "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------
def test_train_writes_all_artifacts(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    result = _train(runner, data, out)

    assert result.exit_code == 0, result.output
    assert (out / "model.pkl").is_file()
    assert (out / "leaderboard.csv").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "validation.json").is_file()
    assert (out / "report.html").is_file()
    assert (out / "plots").is_dir()


def test_train_report_is_valid_html(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert "Leaderboard" in html


def test_train_leaderboard_is_sorted(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)
    board = pd.read_csv(out / "leaderboard.csv")
    scores = board["Test Accuracy"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_train_regression(runner, regression_df, csv_factory, tmp_path):
    data = csv_factory(regression_df)
    out = tmp_path / "out"
    result = _train(runner, data, out)
    assert result.exit_code == 0, result.output
    board = pd.read_csv(out / "leaderboard.csv")
    assert "Test R2" in board.columns


def test_train_handles_messy_data(runner, messy_df, csv_factory, tmp_path):
    data = csv_factory(messy_df)
    out = tmp_path / "out"
    result = _train(runner, data, out)
    assert result.exit_code == 0, result.output


def test_train_missing_file_is_clean_error(runner, tmp_path):
    result = runner.invoke(
        cli, ["train", "--data", str(tmp_path / "nope.csv"), "--target", "y"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


def test_train_missing_target_is_clean_error(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    result = runner.invoke(
        cli, ["train", "--data", str(data), "--target", "nonexistent"]
    )
    assert result.exit_code == 1
    assert "nonexistent" in result.output
    assert "Traceback" not in result.output


def test_train_single_value_target_is_clean_error(runner, csv_factory, tmp_path):
    frame = pd.DataFrame({"a": range(40), "target": [1] * 40})
    data = csv_factory(frame)
    result = runner.invoke(cli, ["train", "--data", str(data), "--target", "target"])
    assert result.exit_code == 1
    assert "one unique value" in result.output
    assert "Traceback" not in result.output


def test_train_suggests_close_column_name(runner, binary_df, csv_factory):
    data = csv_factory(binary_df)
    result = runner.invoke(cli, ["train", "--data", str(data), "--target", "targt"])
    assert result.exit_code == 1
    assert "Did you mean" in result.output


def test_train_rejects_bad_test_size(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    result = runner.invoke(
        cli,
        ["train", "--data", str(data), "--target", "target", "--test-size", "0.99"],
    )
    assert result.exit_code == 1
    assert "test-size" in result.output


def test_train_accepts_yaml_config(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "cfg_out"
    config = tmp_path / "settings.yaml"
    config.write_text(
        f"data: {data.as_posix()}\ntarget: target\nmode: fast\noutput: {out.as_posix()}\n",
        encoding="utf-8",
    )
    result = runner.invoke(cli, ["train", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert (out / "model.pkl").is_file()


def test_train_rejects_unknown_config_key(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    config = tmp_path / "bad.yaml"
    config.write_text(
        f"data: {data.as_posix()}\ntarget: target\nnot_an_option: 5\n", encoding="utf-8"
    )
    result = runner.invoke(cli, ["train", "--config", str(config)])
    assert result.exit_code == 1
    assert "not_an_option" in result.output


def test_train_metadata_records_schema(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["target"] == "target"
    assert "input_columns" in metadata
    assert "feature_a" in metadata["input_columns"]


# ----------------------------------------------------------------------
# validate
# ----------------------------------------------------------------------
def test_validate_clean_data_exits_zero(runner, binary_df, csv_factory):
    data = csv_factory(binary_df)
    result = runner.invoke(cli, ["validate", "--data", str(data), "--target", "target"])
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_validate_catches_injected_leakage(runner, binary_df, csv_factory):
    """The headline guarantee: a planted leak must be caught."""
    leaky = binary_df.assign(leak=binary_df["target"])
    data = csv_factory(leaky, "leaky.csv")
    result = runner.invoke(cli, ["validate", "--data", str(data), "--target", "target"])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "leak" in result.output


def test_validate_writes_json(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    destination = tmp_path / "checks.json"
    result = runner.invoke(
        cli,
        ["validate", "--data", str(data), "--target", "target",
         "--output", str(destination)],
    )
    assert result.exit_code == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["checks"]


def test_validate_without_target_still_runs(runner, binary_df, csv_factory):
    data = csv_factory(binary_df)
    result = runner.invoke(cli, ["validate", "--data", str(data)])
    assert result.exit_code == 0


def test_validate_strict_fails_on_warnings(runner, csv_factory):
    frame = pd.DataFrame({"a": range(300), "target": [0] * 280 + [1] * 20})
    data = csv_factory(frame, "imbalanced.csv")
    lenient = runner.invoke(cli, ["validate", "--data", str(data), "--target", "target"])
    strict = runner.invoke(
        cli, ["validate", "--data", str(data), "--target", "target", "--strict"]
    )
    assert lenient.exit_code == 0
    assert strict.exit_code == 1


# ----------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------
def test_predict_round_trip(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)

    new_rows = binary_df.drop(columns=["target"]).head(12)
    new_path = csv_factory(new_rows, "new.csv")
    predictions = tmp_path / "preds.csv"

    result = runner.invoke(
        cli,
        ["predict", "--model", str(out / "model.pkl"), "--data", str(new_path),
         "--output", str(predictions)],
    )
    assert result.exit_code == 0, result.output
    frame = pd.read_csv(predictions)
    assert len(frame) == 12
    assert "prediction" in frame.columns


def test_predict_preserves_original_labels(runner, binary_df, csv_factory, tmp_path):
    labelled = binary_df.assign(target=binary_df["target"].map({0: "no", 1: "yes"}))
    data = csv_factory(labelled, "labelled.csv")
    out = tmp_path / "out"
    _train(runner, data, out)

    new_path = csv_factory(labelled.drop(columns=["target"]).head(10), "new.csv")
    predictions = tmp_path / "p.csv"
    runner.invoke(
        cli,
        ["predict", "--model", str(out / "model.pkl"), "--data", str(new_path),
         "--output", str(predictions)],
    )
    values = set(pd.read_csv(predictions)["prediction"].unique())
    assert values <= {"no", "yes"}


def test_predict_with_probabilities(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)

    new_path = csv_factory(binary_df.drop(columns=["target"]).head(8), "new.csv")
    predictions = tmp_path / "p.csv"
    result = runner.invoke(
        cli,
        ["predict", "--model", str(out / "model.pkl"), "--data", str(new_path),
         "--output", str(predictions), "--proba"],
    )
    assert result.exit_code == 0, result.output
    frame = pd.read_csv(predictions)
    probability_columns = [c for c in frame.columns if c.startswith("prob_")]
    assert len(probability_columns) == 2
    assert frame[probability_columns].sum(axis=1).round(4).eq(1.0).all()


def test_predict_schema_mismatch_fails_clearly(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)

    broken = binary_df.drop(columns=["target", "feature_a"]).head(5)
    broken_path = csv_factory(broken, "broken.csv")
    result = runner.invoke(
        cli, ["predict", "--model", str(out / "model.pkl"), "--data", str(broken_path)]
    )
    assert result.exit_code == 1
    assert "feature_a" in result.output
    assert "Traceback" not in result.output


def test_predict_tolerates_extra_columns(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)

    extra = binary_df.drop(columns=["target"]).head(6).assign(bonus=1)
    extra_path = csv_factory(extra, "extra.csv")
    predictions = tmp_path / "p.csv"
    result = runner.invoke(
        cli,
        ["predict", "--model", str(out / "model.pkl"), "--data", str(extra_path),
         "--output", str(predictions)],
    )
    assert result.exit_code == 0, result.output


def test_predict_missing_model_is_clean_error(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    result = runner.invoke(
        cli, ["predict", "--model", str(tmp_path / "nope.pkl"), "--data", str(data)]
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_predict_rejects_non_model_file(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    result = runner.invoke(
        cli, ["predict", "--model", str(data), "--data", str(data)]
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output


# ----------------------------------------------------------------------
# report / explain / docs
# ----------------------------------------------------------------------
def test_report_generates_html(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)

    destination = tmp_path / "r.html"
    result = runner.invoke(
        cli, ["report", "--model", str(out / "model.pkl"), "--output", str(destination)]
    )
    assert result.exit_code == 0, result.output
    assert destination.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_explain_prints_pipeline_stages(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)

    result = runner.invoke(cli, ["explain", "--model", str(out / "model.pkl")])
    assert result.exit_code == 0, result.output
    assert "Engineered features" in result.output


def test_explain_writes_html_with_repro_code(runner, binary_df, csv_factory, tmp_path):
    data = csv_factory(binary_df)
    out = tmp_path / "out"
    _train(runner, data, out)

    destination = tmp_path / "e.html"
    result = runner.invoke(
        cli, ["explain", "--model", str(out / "model.pkl"), "--output", str(destination)]
    )
    assert result.exit_code == 0, result.output
    html = destination.read_text(encoding="utf-8")
    assert "import pandas as pd" in html
    assert "train_test_split" in html


def test_docs_generates_site(runner, tmp_path):
    destination = tmp_path / "docs"
    result = runner.invoke(
        cli, ["docs", "--output", str(destination), "--no-browser"]
    )
    assert result.exit_code == 0, result.output
    for page in ("index.html", "quickstart.html", "cli-reference.html"):
        assert (destination / page).is_file()
        assert (destination / page).read_text(encoding="utf-8").startswith("<!DOCTYPE")


def test_docs_cover_every_command(runner, tmp_path):
    destination = tmp_path / "docs"
    runner.invoke(cli, ["docs", "--output", str(destination), "--no-browser"])
    reference = (destination / "cli-reference.html").read_text(encoding="utf-8")
    for command in ("train", "predict", "validate", "explain", "report", "docs"):
        assert f"dive {command}" in reference


def test_deps_lists_optional_packages(runner):
    result = runner.invoke(cli, ["deps"])
    assert result.exit_code == 0
    for package in ("xgboost", "lightgbm", "catboost", "optuna", "shap"):
        assert package in result.output
