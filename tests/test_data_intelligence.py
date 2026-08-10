"""Problem-type detection and dataset profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dive.data_intelligence import DataIntelligence
from dive.exceptions import TargetError


def _profile(frame: pd.DataFrame, target: str = "target") -> dict:
    return DataIntelligence(target).analyze(frame)


def test_detects_binary_classification(binary_df):
    assert _profile(binary_df)["problem_type"] == "classification"


def test_detects_string_labels_as_classification(binary_df):
    frame = binary_df.assign(target=binary_df["target"].map({0: "no", 1: "yes"}))
    assert _profile(frame)["problem_type"] == "classification"


def test_detects_regression(regression_df):
    assert _profile(regression_df)["problem_type"] == "regression"


def test_low_cardinality_integer_target_is_classification(rng):
    frame = pd.DataFrame(
        {"a": rng.normal(size=200), "target": rng.integers(0, 3, size=200)}
    )
    assert _profile(frame)["problem_type"] == "classification"


def test_high_cardinality_float_target_is_regression(rng):
    frame = pd.DataFrame({"a": rng.normal(size=200), "target": rng.normal(size=200)})
    assert _profile(frame)["problem_type"] == "regression"


def test_boolean_target_is_classification(rng):
    frame = pd.DataFrame(
        {"a": rng.normal(size=60), "target": rng.integers(0, 2, size=60).astype(bool)}
    )
    assert _profile(frame)["problem_type"] == "classification"


def test_reports_imbalance_ratio():
    frame = pd.DataFrame({"a": range(100), "target": [0] * 90 + [1] * 10})
    profile = _profile(frame)
    assert profile["imbalance_ratio"] == pytest.approx(9.0)
    assert profile["is_imbalanced"] is True
    assert profile["minority_class_count"] == 10


def test_balanced_target_is_not_flagged(binary_df):
    assert _profile(binary_df)["is_imbalanced"] is False


def test_identifies_structural_columns(messy_df):
    profile = _profile(messy_df)
    assert "row_id" in profile["id_like_cols"]
    assert "constant" in profile["constant_cols"]
    assert "many_levels" in profile["high_card_cols"]
    assert "few_levels" not in profile["high_card_cols"]


def test_reports_missing_data(messy_df):
    profile = _profile(messy_df)
    assert profile["has_missing"] is True
    assert profile["total_missing_pct"] > 0
    assert profile["missing_pct"]["numeric"] > 0


def test_counts_column_types(binary_df):
    profile = _profile(binary_df)
    assert profile["n_numeric"] == 3
    assert profile["n_categorical"] == 1
    assert profile["n_features"] == 4


def test_near_constant_classification_target_is_flagged():
    frame = pd.DataFrame({"a": range(200), "target": [0] * 199 + [1]})
    assert _profile(frame)["target_near_constant"] is True


def test_near_constant_regression_target_is_flagged():
    frame = pd.DataFrame({"a": range(100), "target": [5.0] * 100})
    # nunique == 1 means this is classification by the cardinality rule; the
    # near-constant flag must still fire so validation can report it.
    assert _profile(frame)["target_near_constant"] is True


def test_healthy_target_is_not_near_constant(binary_df):
    assert _profile(binary_df)["target_near_constant"] is False


def test_missing_target_column_raises(binary_df):
    with pytest.raises(TargetError, match="not present"):
        _profile(binary_df, "does_not_exist")


def test_records_target_missing_percentage():
    frame = pd.DataFrame(
        {"a": range(10), "target": [0, 1] * 4 + [np.nan, np.nan]}
    )
    assert _profile(frame)["target_missing_pct"] == pytest.approx(20.0)


def test_class_distribution_is_serialisable(binary_df):
    distribution = _profile(binary_df)["class_distribution"]
    assert all(isinstance(key, str) for key in distribution)
    import json

    json.dumps(distribution)
