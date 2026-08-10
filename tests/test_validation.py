"""Each crosscheck against a fixture engineered to trigger it.

Every check gets both a positive case (the fault is present and detected) and a
negative case (clean data is not flagged), because a checker that fires on
everything is as useless as one that never fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dive.validation import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    check_cv_stability,
    check_duplicate_rows,
    check_missing_data,
    check_predict_schema,
    check_target_health,
    check_target_leakage,
    check_train_holdout_drift,
    run_validation_suite,
)
from dive.data_intelligence import DataIntelligence


# ----------------------------------------------------------------------
# target leakage
# ----------------------------------------------------------------------
def test_detects_numeric_binary_leakage(binary_df):
    frame = binary_df.assign(leak=binary_df["target"])
    report = run_validation_suite(frame, target="target")
    check = report.get("target_leakage")
    assert check.status == FAIL
    assert "leak" in check.metrics["leaking_features"]


def test_detects_categorical_leakage(binary_df):
    frame = binary_df.assign(
        leak=binary_df["target"].map({0: "negative", 1: "positive"})
    )
    check = run_validation_suite(frame, target="target").get("target_leakage")
    assert check.status == FAIL
    assert "leak" in check.metrics["leaking_features"]


def test_detects_regression_leakage(regression_df):
    frame = regression_df.assign(leak=regression_df["target"] * 2.0 + 1.0)
    check = run_validation_suite(frame, target="target").get("target_leakage")
    assert check.status == FAIL
    assert "leak" in check.metrics["leaking_features"]


def test_clean_data_has_no_leakage(binary_df):
    assert run_validation_suite(binary_df, target="target").get(
        "target_leakage"
    ).status == PASS


def test_clean_regression_has_no_leakage(regression_df):
    """A strong but legitimate predictor must not be called leakage."""
    assert run_validation_suite(regression_df, target="target").get(
        "target_leakage"
    ).status == PASS


def test_id_column_is_not_reported_as_leakage(binary_df):
    frame = binary_df.assign(row_id=[f"id_{i}" for i in range(len(binary_df))])
    check = run_validation_suite(frame, target="target").get("target_leakage")
    assert "row_id" not in check.metrics["leaking_features"]


# ----------------------------------------------------------------------
# target health
# ----------------------------------------------------------------------
def test_near_constant_target_fails():
    frame = pd.DataFrame({"a": range(200), "target": [0] * 199 + [1]})
    profile = DataIntelligence("target").analyze(frame)
    assert check_target_health(profile).status == FAIL


def test_severe_imbalance_warns():
    frame = pd.DataFrame(
        {"a": range(300), "target": [0] * 280 + [1] * 20}
    )
    profile = DataIntelligence("target").analyze(frame)
    result = check_target_health(profile)
    assert result.status == WARN
    assert result.metrics["imbalance_ratio"] == pytest.approx(14.0)


def test_tiny_minority_class_fails():
    frame = pd.DataFrame({"a": range(200), "target": [0] * 197 + [1] * 3})
    profile = DataIntelligence("target").analyze(frame)
    assert check_target_health(profile).status == FAIL


def test_healthy_target_passes(binary_df):
    profile = DataIntelligence("target").analyze(binary_df)
    assert check_target_health(profile).status == PASS


# ----------------------------------------------------------------------
# duplicate rows
# ----------------------------------------------------------------------
def test_detects_duplicate_rows(binary_df):
    frame = pd.concat([binary_df, binary_df.head(40)], ignore_index=True)
    result = check_duplicate_rows(frame.drop(columns=["target"]))
    assert result.status == WARN
    assert result.metrics["duplicate_rows"] == 40


def test_detects_duplicates_crossing_the_split(binary_df):
    frame = pd.concat([binary_df, binary_df.head(40)], ignore_index=True)
    check = run_validation_suite(frame, target="target").get("duplicate_rows")
    assert check.status == FAIL
    assert check.metrics["rows_in_both_splits"] > 0


def test_no_duplicates_passes(binary_df):
    assert check_duplicate_rows(binary_df.drop(columns=["target"])).status == PASS


# ----------------------------------------------------------------------
# drift
# ----------------------------------------------------------------------
def test_detects_drift_between_splits(rng):
    train = pd.DataFrame({"x": rng.normal(0, 1, 300), "y": rng.normal(0, 1, 300)})
    holdout = pd.DataFrame({"x": rng.normal(6, 1, 120), "y": rng.normal(0, 1, 120)})
    result = check_train_holdout_drift(train, holdout)
    assert result.status == WARN
    assert "x" in result.metrics["drifted_features"]
    assert "y" not in result.metrics["drifted_features"]


def test_same_distribution_passes(rng):
    train = pd.DataFrame({"x": rng.normal(size=300)})
    holdout = pd.DataFrame({"x": rng.normal(size=150)})
    assert check_train_holdout_drift(train, holdout).status == PASS


def test_drift_skipped_without_numeric_columns():
    train = pd.DataFrame({"c": ["a"] * 40})
    holdout = pd.DataFrame({"c": ["b"] * 40})
    assert check_train_holdout_drift(train, holdout).status == SKIP


# ----------------------------------------------------------------------
# missing data
# ----------------------------------------------------------------------
def test_empty_column_fails():
    result = check_missing_data(
        {"missing_pct": {"dead": 100.0, "fine": 1.0}, "total_missing_pct": 50.0}
    )
    assert result.status == FAIL


def test_heavily_missing_column_warns():
    result = check_missing_data(
        {"missing_pct": {"sparse": 60.0, "fine": 2.0}, "total_missing_pct": 31.0}
    )
    assert result.status == WARN
    assert "sparse" in result.metrics["high_missing_columns"]


def test_low_missingness_passes():
    result = check_missing_data(
        {"missing_pct": {"a": 1.0, "b": 0.0}, "total_missing_pct": 0.5}
    )
    assert result.status == PASS


# ----------------------------------------------------------------------
# cv stability
# ----------------------------------------------------------------------
def test_unstable_folds_warn():
    result = check_cv_stability([0.95, 0.55, 0.80, 0.62], model_name="RF")
    assert result.status == WARN
    assert result.metrics["relative_std"] > 0.1


def test_stable_folds_pass():
    result = check_cv_stability([0.900, 0.905, 0.898, 0.902])
    assert result.status == PASS


def test_missing_fold_scores_skip():
    assert check_cv_stability([]).status == SKIP


# ----------------------------------------------------------------------
# predict schema
# ----------------------------------------------------------------------
def test_missing_required_column_fails():
    incoming = pd.DataFrame({"a": [1], "b": [2]})
    result = check_predict_schema(["a", "b", "c"], [], incoming)
    assert result.status == FAIL
    assert result.metrics["missing"] == ["c"]


def test_matching_schema_passes():
    incoming = pd.DataFrame({"a": [1], "b": [2]})
    assert check_predict_schema(["a", "b"], [], incoming).status == PASS


def test_dropped_columns_may_be_absent():
    incoming = pd.DataFrame({"a": [1]})
    assert check_predict_schema(["a", "row_id"], ["row_id"], incoming).status == PASS


def test_extra_columns_are_tolerated():
    incoming = pd.DataFrame({"a": [1], "unexpected": [9]})
    result = check_predict_schema(["a"], [], incoming)
    assert result.status == PASS
    assert result.metrics["extra"] == ["unexpected"]


def test_dtype_change_warns():
    incoming = pd.DataFrame({"a": ["text"]})
    result = check_predict_schema(["a"], [], incoming, {"a": "float64"})
    assert result.status == WARN
    assert "a" in result.metrics["dtype_mismatches"]


# ----------------------------------------------------------------------
# report aggregation
# ----------------------------------------------------------------------
def test_clean_dataset_produces_no_failures(binary_df):
    report = run_validation_suite(binary_df, target="target")
    assert not report.has_failures
    assert report.worst_status in (PASS, WARN)


def test_report_serialises_to_json(binary_df):
    import json

    payload = run_validation_suite(binary_df, target="target").to_dict()
    json.dumps(payload)
    assert "checks" in payload and payload["checks"]


def test_report_renders_as_text(binary_df):
    text = run_validation_suite(binary_df, target="target").render()
    assert "target_leakage" in text
    assert "Summary:" in text


def test_suite_without_target_skips_target_checks(binary_df):
    report = run_validation_suite(binary_df.drop(columns=["target"]))
    assert report.get("target_health").status == SKIP
    assert report.get("duplicate_rows") is not None


def test_worst_status_reflects_failures(binary_df):
    frame = binary_df.assign(leak=binary_df["target"])
    assert run_validation_suite(frame, target="target").worst_status == FAIL
