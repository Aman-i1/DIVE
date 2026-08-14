"""Unit & Integration tests for Diversity-Aware Ensembling & Error Correlation Selection."""

from __future__ import annotations

import numpy as np
import pytest

from dive.ensemble_diversity import DiversityMatrix, ModelDiversityEvaluator


def test_diversity_matrix_calculation() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Model A is accurate
    preds_a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # Model B is identical to A (redundant)
    preds_b = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # Model C has complementary mistakes
    preds_c = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1])

    evaluator = ModelDiversityEvaluator(problem_type="classification")
    matrix = evaluator.compute_diversity_matrix(
        {"model_a": preds_a, "model_b": preds_b, "model_c": preds_c},
        y_true=y_true,
    )

    assert matrix.model_names == ["model_a", "model_b", "model_c"]
    # Correlation between model_a and model_b should be 1.0 (identical)
    assert matrix.correlation_matrix[0, 1] == pytest.approx(1.0)
    # Correlation between model_a and model_c should be lower
    assert matrix.correlation_matrix[0, 2] < 1.0


def test_diverse_subset_selection() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # 4 models with varying accuracy and diversity
    preds_dict = {
        "gbdt_1": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),  # perfect
        "gbdt_2": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),  # duplicate
        "rf": np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]),      # diverse
        "linear": np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1]),  # diverse
    }

    evaluator = ModelDiversityEvaluator(problem_type="classification")
    selected = evaluator.select_diverse_subset(preds_dict, y_true, max_models=3, diversity_weight=0.3)

    assert len(selected) <= 3
    assert "gbdt_1" in selected
    # Redundant clone gbdt_2 should be penalized / skipped if diverse models available
    assert "rf" in selected or "linear" in selected
