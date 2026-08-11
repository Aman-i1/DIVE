"""Unit & Integration tests for Phase 4: Model Intelligence & Capability Registry."""

from __future__ import annotations

import pytest

from dive.capability_registry import CapabilityRegistry, ModelCapability


def test_capability_registry_defaults() -> None:
    reg = CapabilityRegistry()
    cap = reg.get("RandomForest")

    assert cap is not None
    assert "classification" in cap.task_types
    assert "regression" in cap.task_types
    assert cap.max_reasonable_rows == 500_000


def test_capability_registry_recommendations() -> None:
    reg = CapabilityRegistry()
    recs = reg.recommend(
        problem_type="classification",
        n_samples=50_000,
        n_features=25,
        has_missing=True,
        has_categorical=True,
    )

    assert "recommended" in recs
    assert "acceptable" in recs
    assert "rejected" in recs
    assert "HistGradientBoosting" in recs["recommended"]
    assert "RandomForest" in recs["acceptable"]
