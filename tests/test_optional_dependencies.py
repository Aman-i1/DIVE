"""Optional-dependency handling and the sklearn-only guarantee.

The core promise: with nothing but scikit-learn installed the tool still trains,
just with a smaller model zoo and a clear warning about what was skipped. These
tests simulate absence by patching the registry, so they hold on a machine where
every extra happens to be installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dive.data_intelligence import DataIntelligence
from dive.model_zoo import ModelZoo
from dive.utils import optional


@pytest.fixture
def all_extras_missing(monkeypatch):
    """Report every optional package as unavailable."""
    monkeypatch.setattr(optional, "_PROBED", True)
    for package in optional._REGISTRY.values():
        monkeypatch.setattr(package, "available", False, raising=False)
        monkeypatch.setattr(package, "module", None, raising=False)
        monkeypatch.setattr(package, "version", None, raising=False)


def _profile(frame: pd.DataFrame) -> dict:
    return DataIntelligence("target").analyze(frame)


def test_zoo_falls_back_to_sklearn_only(all_extras_missing, binary_df):
    models = ModelZoo("classification", _profile(binary_df), mode="competition").get_models()
    assert models
    for booster in ("XGBoost", "LightGBM", "CatBoost"):
        assert booster not in models
    assert "RandomForest" in models
    assert "LogisticRegression" in models


def test_regression_zoo_falls_back(all_extras_missing, regression_df):
    models = ModelZoo("regression", _profile(regression_df), mode="balanced").get_models()
    assert models
    assert "XGBoost" not in models
    assert "RandomForest" in models


def test_missing_summary_names_each_package(all_extras_missing):
    summary = optional.missing_summary()
    for package in ("xgboost", "lightgbm", "catboost", "optuna", "shap"):
        assert package in summary
    assert "pip install" in summary


def test_missing_summary_explains_the_cost(all_extras_missing):
    """The warning must say what capability is lost, not just what is absent."""
    summary = optional.missing_summary()
    assert "hyperparameter tuning" in summary
    assert "SHAP" in summary


def test_missing_summary_is_empty_when_all_present(monkeypatch):
    monkeypatch.setattr(optional, "_PROBED", True)
    for package in optional._REGISTRY.values():
        monkeypatch.setattr(package, "available", True, raising=False)
    assert optional.missing_summary() == ""


def test_training_works_without_extras(all_extras_missing, binary_df, tmp_path):
    """The end-to-end guarantee: scikit-learn alone is enough to finish a run."""
    from click.testing import CliRunner

    from dive.cli import cli

    data = tmp_path / "data.csv"
    binary_df.to_csv(data, index=False)
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli,
        ["train", "--data", str(data), "--target", "target",
         "--mode", "fast", "--output", str(out)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (out / "model.pkl").is_file()


def test_feature_engineer_skips_target_encoding_without_ce(all_extras_missing, messy_df):
    from dive.feature_engineering import FeatureEngineer

    profile = _profile(messy_df)
    engineer = FeatureEngineer(profile=profile, target="target", use_target_encoding=True)
    engineer.fit_transform(messy_df.drop(columns=["target"]), messy_df["target"])
    assert engineer.target_enc_ is None


def test_tuning_returns_untuned_pipeline_without_optuna(all_extras_missing):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from dive.tuning import OptunaOptimizer

    pipeline = Pipeline([("model", LogisticRegression())])
    optimizer = OptunaOptimizer("classification", cv=3, scoring="accuracy")
    tuned, score = optimizer.tune(pipeline, "LogisticRegression", None, None)
    assert tuned is pipeline
    assert score is None


def test_version_tuple_handles_absent_package(all_extras_missing):
    assert optional.version_tuple("xgboost") == (0, 0)


def test_version_tuple_parses_release_candidates(monkeypatch):
    monkeypatch.setattr(optional, "_PROBED", True)
    package = optional._REGISTRY["xgboost"]
    monkeypatch.setattr(package, "available", True, raising=False)
    monkeypatch.setattr(package, "version", "2.1.0rc1", raising=False)
    assert optional.version_tuple("xgboost") == (2, 1)


def test_gpu_detection_respects_env_override(monkeypatch):
    monkeypatch.setenv("AUTOML_NO_GPU", "1")
    assert optional.detect_gpu() is False
