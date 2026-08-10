"""FeatureEngineer behaviour on awkward data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dive.data_intelligence import DataIntelligence
from dive.feature_engineering import FeatureEngineer


def _fit(frame: pd.DataFrame, target: str = "target", **kwargs):
    """Profile, then fit an engineer on the feature columns."""
    profile = DataIntelligence(target).analyze(frame)
    engineer = FeatureEngineer(profile=profile, target=target, **kwargs)
    y = frame[target]
    X = frame.drop(columns=[target])
    return engineer, engineer.fit_transform(X, y)


def test_drops_id_and_constant_columns(messy_df):
    engineer, out = _fit(messy_df)
    assert "row_id" not in out.columns
    assert "constant" not in out.columns
    assert set(engineer.drop_cols_) >= {"row_id", "constant"}


def test_expands_datetime_into_parts(messy_df):
    _, out = _fit(messy_df)
    for part in ("year", "month", "day", "dayofweek", "quarter", "weekofyear"):
        assert f"signed_up_{part}" in out.columns
    assert "signed_up" not in out.columns


def test_output_has_no_object_columns_when_target_encoding_on(messy_df):
    pytest.importorskip("category_encoders")
    _, out = _fit(messy_df, use_target_encoding=True)
    assert not list(out.select_dtypes("object").columns)


def test_frequency_encoding_adds_columns(messy_df):
    _, out = _fit(messy_df, use_freq_encoding=True, use_target_encoding=False)
    assert "few_levels_freq" in out.columns
    # Missing categories stay NaN at fit time and are imputed by the sklearn
    # pipeline; the observed frequencies must still be valid proportions.
    observed = out["few_levels_freq"].dropna()
    assert len(observed) > 0
    assert observed.between(0, 1).all()


def test_outlier_clipping_bounds_extremes(messy_df):
    _, out = _fit(messy_df, outlier_clip=True, use_target_encoding=False)
    # The fixture plants +/-10000 outliers; clipping must pull them in.
    assert out["numeric"].max() < 100
    assert out["numeric"].min() > -100


def test_outlier_clipping_can_be_disabled(messy_df):
    _, out = _fit(messy_df, outlier_clip=False, use_target_encoding=False)
    assert out["numeric"].max() > 1000


def test_rare_categories_are_grouped(rng):
    frame = pd.DataFrame(
        {
            "cat": ["common"] * 195 + [f"rare_{i}" for i in range(5)],
            "num": rng.normal(size=200),
            "target": rng.integers(0, 2, size=200),
        }
    )
    engineer, _ = _fit(frame, use_target_encoding=False, rare_threshold=0.02)
    assert engineer.rare_maps_["cat"]


def test_transform_matches_fit_columns(messy_df):
    engineer, fitted = _fit(messy_df, use_target_encoding=False)
    transformed = engineer.transform(messy_df.drop(columns=["target"]))
    assert list(transformed.columns) == list(fitted.columns)


def test_transform_does_not_refit(messy_df):
    """Predict-time transform must reuse fitted state, never relearn it."""
    engineer, _ = _fit(messy_df, use_target_encoding=False)
    before_clip = dict(engineer.clip_bounds_)
    before_freq = {k: dict(v) for k, v in engineer.freq_maps_.items()}
    before_dates = list(engineer.datetime_cols_)

    shifted = messy_df.drop(columns=["target"]).copy()
    shifted["numeric"] = shifted["numeric"] * 100 + 500
    engineer.transform(shifted)

    assert engineer.clip_bounds_ == before_clip
    assert engineer.freq_maps_ == before_freq
    assert engineer.datetime_cols_ == before_dates


def test_unseen_category_gets_zero_frequency(rng):
    frame = pd.DataFrame(
        {
            "cat": rng.choice(["a", "b"], size=80),
            "num": rng.normal(size=80),
            "target": rng.integers(0, 2, size=80),
        }
    )
    engineer, _ = _fit(frame, use_target_encoding=False)
    new = pd.DataFrame({"cat": ["never_seen"], "num": [0.5]})
    out = engineer.transform(new)
    assert out["cat_freq"].iloc[0] == 0.0


def test_refitting_is_idempotent(messy_df):
    """Calling fit_transform twice must not accumulate duplicate state."""
    profile = DataIntelligence("target").analyze(messy_df)
    engineer = FeatureEngineer(profile=profile, target="target")
    X = messy_df.drop(columns=["target"])
    y = messy_df["target"]
    first = engineer.fit_transform(X, y)
    second = engineer.fit_transform(X, y)
    assert list(first.columns) == list(second.columns)
    assert engineer.datetime_cols_.count("signed_up") == 1


def test_missing_values_survive_to_imputation(messy_df):
    """FeatureEngineer must not impute - that belongs to the sklearn pipeline."""
    _, out = _fit(messy_df, use_target_encoding=False)
    assert out["numeric"].isna().any()


def test_floats_are_downcast_to_float32(messy_df):
    _, out = _fit(messy_df, use_target_encoding=False)
    assert not list(out.select_dtypes("float64").columns)


def test_missing_input_columns_ignores_dropped(messy_df):
    engineer, _ = _fit(messy_df, use_target_encoding=False)
    without_dropped = messy_df.drop(columns=["target", "row_id", "constant"])
    assert engineer.missing_input_columns(without_dropped) == []


def test_missing_input_columns_reports_real_gaps(messy_df):
    engineer, _ = _fit(messy_df, use_target_encoding=False)
    missing = engineer.missing_input_columns(
        messy_df.drop(columns=["target", "numeric"])
    )
    assert missing == ["numeric"]


def test_describe_reports_actions(messy_df):
    engineer, _ = _fit(messy_df, use_target_encoding=False)
    description = engineer.describe()
    assert set(description["dropped_columns"]) >= {"row_id", "constant"}
    assert "signed_up" in description["datetime_columns"]
    assert description["outlier_clip"] is True


def test_free_text_column_is_not_parsed_as_datetime(rng):
    frame = pd.DataFrame(
        {
            "notes": [f"note number {i}" for i in range(60)],
            "num": rng.normal(size=60),
            "target": rng.integers(0, 2, size=60),
        }
    )
    engineer, _ = _fit(frame, use_target_encoding=False)
    assert "notes" not in engineer.datetime_cols_
