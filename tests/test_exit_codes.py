"""Process-level exit code tests.

The rest of the CLI suite drives commands through ``CliRunner``, which invokes
the click group directly and never touches :func:`dive.cli.main`. ``main`` is
what the installed ``dive`` console script actually runs, and it is where the
click return value has to be turned back into a process exit status - so these
tests call it directly. A regression here is invisible to CliRunner but breaks
every CI gate and shell script using the tool.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dive.cli import main


@pytest.fixture
def clean_csv(binary_df, csv_factory):
    return csv_factory(binary_df, "clean.csv")


@pytest.fixture
def leaky_csv(binary_df, csv_factory):
    frame = binary_df.copy()
    frame["leaked"] = frame["target"]  # a column that is literally the answer
    return csv_factory(frame, "leaky.csv")


# ----------------------------------------------------------------------
# success paths return 0
# ----------------------------------------------------------------------
@pytest.mark.parametrize("args", [["--help"], ["--version"], ["train", "--help"]])
def test_informational_commands_exit_zero(args):
    assert main(args) == 0


def test_validate_on_clean_data_exits_zero(clean_csv):
    assert main(["validate", "--data", str(clean_csv), "--target", "target"]) == 0


def test_train_success_exits_zero(clean_csv, tmp_path):
    code = main([
        "train",
        "--data", str(clean_csv),
        "--target", "target",
        "--mode", "fast",
        "--output", str(tmp_path / "out"),
    ])
    assert code == 0


# ----------------------------------------------------------------------
# failure paths must NOT report success
# ----------------------------------------------------------------------
def test_validate_failure_exits_one(leaky_csv):
    """A caught leak has to be a non-zero exit, or `validate` is useless in CI."""
    assert main(["validate", "--data", str(leaky_csv), "--target", "target"]) == 1


def test_missing_data_file_exits_one(tmp_path):
    assert main(["train", "--data", str(tmp_path / "nope.csv"), "--target", "y"]) == 1


def test_missing_target_column_exits_one(clean_csv, tmp_path):
    code = main([
        "train",
        "--data", str(clean_csv),
        "--target", "not_a_column",
        "--mode", "fast",
        "--output", str(tmp_path / "out"),
    ])
    assert code == 1


def test_unknown_command_exits_nonzero():
    assert main(["frobnicate"]) != 0


def test_predict_schema_mismatch_exits_one(clean_csv, binary_df, csv_factory, tmp_path):
    out = tmp_path / "out"
    assert main([
        "train", "--data", str(clean_csv), "--target", "target",
        "--mode", "fast", "--output", str(out),
    ]) == 0

    # drop a column the model was trained on
    incoming = binary_df.drop(columns=["target"]).iloc[:5]
    broken = csv_factory(incoming.drop(columns=[incoming.columns[0]]), "broken.csv")

    code = main([
        "predict",
        "--model", str(out / "model.pkl"),
        "--data", str(broken),
        "--output", str(tmp_path / "preds.csv"),
    ])
    assert code == 1
