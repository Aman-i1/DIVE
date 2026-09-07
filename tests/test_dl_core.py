"""Tests for DIVE Deep Learning core configuration, trainer, predictor, and capability assessment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dive.dl import (
    AutoDL,
    DLConfig,
    DLDataConfig,
    DLLeaderboard,
    DLPredictor,
    DLTask,
    DLTrainer,
    DLTrainingConfig,
    DLTrial,
    Modality,
    assess_dl_workload,
    load_dl_predictor,
    project_dl_memory_mb,
    save_dl_predictor,
    select_device,
)
from dive.dl.exceptions import DLConfigError, DLTrainingError
from dive.dl.modalities.tabular import TabularAdapter


def test_dl_config_validation():
    cfg = DLConfig(modality="tabular", task="classification", mode="fast")
    assert cfg.modality_enum == Modality.TABULAR
    assert cfg.task_enum == DLTask.CLASSIFICATION

    cfg.apply_mode()
    assert cfg.training.epochs == 3

    with pytest.raises(DLConfigError):
        DLConfig(modality="non_existent_modality")

    with pytest.raises(DLConfigError):
        DLConfig(task="invalid_task")

    with pytest.raises(DLConfigError):
        DLConfig(mode="invalid_mode")


def test_dl_device_selection():
    device = select_device("cpu")
    assert device.name == "cpu"
    assert not device.is_accelerated
    assert "CPU" in device.render()


def test_dl_capability_assessment():
    mb = project_dl_memory_mb("tabular", n_samples=1000)
    assert mb > 0

    verdict = assess_dl_workload(Modality.TABULAR, n_samples=100)
    assert verdict.decision in ("ALLOW", "DEGRADE", "REFUSE")


def test_dl_trainer_sklearn_fallback():
    # 20 samples, 4 features
    rng = np.random.RandomState(42)
    X = rng.randn(20, 4).astype(np.float32)
    y = np.array([0, 1] * 10)

    trainer = DLTrainer(
        task=DLTask.CLASSIFICATION,
        config=DLTrainingConfig(epochs=5, hidden_sizes=[16]),
        force_fallback=True,
    )
    trainer.fit(X, y)
    assert trainer.backend == "sklearn-mlp"

    preds = trainer.predict(X)
    assert len(preds) == 20
    assert set(map(str, preds)).issubset({"0", "1"})

    probs = trainer.predict_proba(X)
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_dl_predictor_save_and_load(tmp_path: Path):
    rng = np.random.RandomState(42)
    X = rng.randn(10, 4).astype(np.float32)
    y = np.array([0, 1] * 5)

    trainer = DLTrainer(
        task=DLTask.CLASSIFICATION,
        config=DLTrainingConfig(epochs=3, hidden_sizes=[8]),
        force_fallback=True,
    )
    trainer.fit(X, y)

    adapter = TabularAdapter()
    adapter.fitted = True
    adapter.feature_names = ["f0", "f1", "f2", "f3"]

    predictor = DLPredictor(
        adapter=adapter,
        trainer=trainer,
        modality=Modality.TABULAR,
        task=DLTask.CLASSIFICATION,
    )

    save_path = tmp_path / "test_predictor.pkl"
    save_dl_predictor(predictor, save_path)
    assert save_path.is_file()

    loaded = load_dl_predictor(save_path)
    assert loaded.modality == "tabular"
    assert loaded.backend == trainer.backend

    desc = loaded.describe()
    assert desc["modality"] == "tabular"
    assert desc["has_proba"] is True


def test_dl_leaderboard_rendering():
    board = DLLeaderboard(primary_metric="accuracy")
    cfg = DLConfig()

    trial1 = DLTrial(
        trial_id=1,
        config=cfg,
        model_name="MLP(128-64)",
        backend="sklearn-mlp",
        primary_metric="accuracy",
        primary_metric_score=0.92,
        composite_score=0.90,
        train_time_ms=120.0,
        inference_latency_ms=0.45,
    )
    trial2 = DLTrial(
        trial_id=2,
        config=cfg,
        model_name="MLP(64)",
        backend="sklearn-mlp",
        primary_metric="accuracy",
        primary_metric_score=0.88,
        composite_score=0.86,
        train_time_ms=80.0,
        inference_latency_ms=0.30,
    )
    board.add_trial(trial1)
    board.add_trial(trial2)

    assert board.champion_trial.model_name == "MLP(128-64)"
    rendered = board.render()
    assert "DIVE AUTODL MODEL SELECTION LEADERBOARD" in rendered
    assert "MLP(128-64)" in rendered
    assert "[TOP]" in rendered
