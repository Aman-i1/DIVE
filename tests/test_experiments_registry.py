"""Tests for ExperimentTracker, DatasetHasher, ModelRegistry and PromotionGate."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from dive.experiments import DatasetHasher, ExperimentTracker
from dive.registry import ModelRegistry, PromotionGate


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as exp_dir, tempfile.TemporaryDirectory() as reg_dir:
        yield Path(exp_dir), Path(reg_dir)


def test_dataset_hasher() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    res = DatasetHasher.hash_dataframe(df)
    assert "dataset_hash" in res
    assert res["n_rows"] == 3
    assert res["n_cols"] == 2


def test_experiment_tracker(temp_dirs) -> None:
    exp_dir, _ = temp_dirs
    tracker = ExperimentTracker(storage_dir=exp_dir)
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})

    rec = tracker.record_run(
        dataset_name="test_ds",
        df=df,
        target="target",
        problem_type="classification",
        validation_strategy="KFold",
        model_name="LightGBM",
        hyperparameters={"n_estimators": 100},
        metrics={"Accuracy": 0.95},
        training_time_sec=1.5,
    )

    assert rec.experiment_id.startswith("EXP-")
    exps = tracker.list_experiments()
    assert len(exps) == 1
    assert exps[0]["model_name"] == "LightGBM"


def test_model_registry_and_promotion_gate(temp_dirs) -> None:
    _, reg_dir = temp_dirs
    registry = ModelRegistry(registry_dir=reg_dir)

    # Create dummy artifact file
    dummy_file = reg_dir / "dummy_model.pkl"
    dummy_file.write_text("model data")

    v1_dir = registry.register_model(
        model_name="churn_model",
        model_artifact_path=dummy_file,
        metrics={"Macro F1": 0.85},
        schema={},
        stage="candidate",
    )

    assert v1_dir.exists()
    models = registry.list_models("churn_model")
    assert len(models) == 1
    assert models[0]["version"] == "v1"

    # Promote to production
    gate_check = registry.promote_model("churn_model", "v1", target_stage="production")
    assert gate_check.approved is True
    prod = registry.get_production_model("churn_model")
    assert prod is not None
    assert prod["version"] == "v1"
