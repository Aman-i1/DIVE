"""Shared fixtures.

The fixtures here build the smallest datasets that still exercise the code
paths under test: enough rows for a stratified 3-fold split, and exactly the
column shapes (datetime, high-cardinality, constant, ID-like) each test needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Force CPU everywhere so results don't depend on the runner's hardware."""
    monkeypatch.setenv("AUTOML_NO_GPU", "1")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240607)


@pytest.fixture
def binary_df(rng) -> pd.DataFrame:
    """Small binary-classification frame with a genuine (non-perfect) signal."""
    n = 180
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "feature_a": signal,
            "feature_b": rng.normal(size=n),
            "feature_c": rng.integers(0, 5, size=n),
            "category": rng.choice(["x", "y", "z"], size=n),
            "target": (signal + rng.normal(scale=0.6, size=n) > 0).astype(int),
        }
    )


@pytest.fixture
def multiclass_df(rng) -> pd.DataFrame:
    """Three-class frame with string labels, shaped like iris.

    Multiclass is its own code path for the boosting models: XGBoost needs
    ``mlogloss`` rather than the binary ``logloss``, and label encoding has to
    round-trip back to the original strings.
    """
    n = 180
    petal = np.concatenate([rng.normal(loc, 0.4, n // 3) for loc in (1.5, 4.3, 5.6)])
    sepal = np.concatenate([rng.normal(loc, 0.4, n // 3) for loc in (5.0, 5.9, 6.6)])
    return pd.DataFrame(
        {
            "petal length (cm)": petal,
            "sepal length (cm)": sepal,
            "noise": rng.normal(size=n),
            "species": np.repeat(["setosa", "versicolor", "virginica"], n // 3),
        }
    )


@pytest.fixture
def regression_df(rng) -> pd.DataFrame:
    n = 180
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame(
        {
            "feature_a": a,
            "feature_b": b,
            "target": 3.0 * a - 2.0 * b + rng.normal(scale=0.4, size=n),
        }
    )


@pytest.fixture
def messy_df(rng) -> pd.DataFrame:
    """Every FeatureEngineer code path in one frame.

    Contains: an ID-like column, a constant column, a datetime column, a
    high-cardinality categorical, missing values, and extreme outliers.
    """
    n = 160
    frame = pd.DataFrame(
        {
            "row_id": np.arange(n),
            "constant": 7,
            "signed_up": pd.date_range("2021-03-01", periods=n, freq="D").astype(str),
            "many_levels": [f"lvl_{i % 90}" for i in range(n)],
            "few_levels": rng.choice(["red", "blue"], size=n),
            "numeric": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    frame.loc[::11, "numeric"] = np.nan
    frame.loc[::17, "few_levels"] = np.nan
    frame.loc[0, "numeric"] = 10_000.0
    frame.loc[1, "numeric"] = -10_000.0
    return frame


@pytest.fixture
def csv_factory(tmp_path):
    """Write a frame to a CSV under tmp_path and return the path."""

    def _write(frame: pd.DataFrame, name: str = "data.csv"):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    return _write
