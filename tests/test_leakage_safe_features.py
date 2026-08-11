"""Adversarial Temporal Leakage & Feature Availability Unit Tests for Phase 3."""

from __future__ import annotations

import pandas as pd
import pytest

from dive.feature_availability import FeatureAvailabilityModel
from dive.feature_selection import FeaturePruner
from dive.temporal_features import LeakageSafeTemporalEngine


def test_leakage_safe_temporal_engine() -> None:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
        "sales": [100.0, 200.0, 300.0, 400.0, 500.0],
        "target": [0, 0, 1, 1, 0],
    })

    engine = LeakageSafeTemporalEngine(time_column="timestamp")
    df_feat, availability = engine.generate_features(df, lags=(1,), rolling_windows=(2,))

    # Verify lag_1 shifts properly (first row must be NaN for lag)
    assert pd.isna(df_feat["sales_lag_1"].iloc[0])
    assert df_feat["sales_lag_1"].iloc[1] == 100.0
    assert df_feat["sales_lag_1"].iloc[2] == 200.0

    # Verify rolling_mean_2 shifts properly and does not include current row
    # Row 1 (sales=200): rolling mean of previous row (100) -> 100.0
    assert df_feat["sales_rolling_mean_2"].iloc[1] == 100.0
    # Row 2 (sales=300): rolling mean of previous 2 rows (100, 200) -> 150.0
    assert df_feat["sales_rolling_mean_2"].iloc[2] == 150.0

    # Verify feature availability model tracked metadata
    assert "sales_lag_1" in availability.registry
    assert availability.registry["sales_lag_1"].is_point_in_time_safe is True


def test_feature_pruner() -> None:
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f1_collinear": [1.0000001, 2.0000001, 3.0000001, 4.0000001, 5.0000001],
        "f_zero_var": [7.0, 7.0, 7.0, 7.0, 7.0],
    })
    y = pd.Series([0, 1, 0, 1, 0])

    pruner = FeaturePruner(correlation_threshold=0.98)
    pruned_df = pruner.fit_transform(df, y)

    assert "f_zero_var" not in pruned_df.columns
    assert "f1_collinear" not in pruned_df.columns
    assert "f1" in pruned_df.columns
