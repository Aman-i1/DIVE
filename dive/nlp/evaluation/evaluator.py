"""NLP Evaluation & Statistical Scoring Engine - `dive/nlp/evaluation/evaluator.py`.

Computes comprehensive metrics for text classification and text regression:
- Multi-class accuracy, balanced accuracy, Macro/Weighted/Micro F1, Precision, Recall
- Multi-class Log Loss and Brier score when calibrated probabilities are present
- Continuous regression metrics: R², MAE, MSE, RMSE
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class NLPEvaluator:
    """Computes standard NLP evaluation metrics across tasks."""

    def __init__(self, task_type: str = "text_classification") -> None:
        self.task_type = task_type

    def evaluate(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics dictionary for predictions."""
        if self.task_type in ("text_regression", "regression"):
            return self._evaluate_regression(y_true, y_pred)
        return self._evaluate_classification(y_true, y_pred, y_proba, class_names)

    def _evaluate_classification(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)

        acc = float(accuracy_score(y_t, y_p))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
        precision = float(precision_score(y_t, y_p, average="macro", zero_division=0))
        recall = float(recall_score(y_t, y_p, average="macro", zero_division=0))

        metrics: Dict[str, Any] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_samples": len(y_t),
        }

        # Multi-class balanced accuracy
        try:
            metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y_t, y_p)), 4)
        except Exception:
            pass

        # Log loss when probability distributions are supplied
        if y_proba is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics["log_loss"] = round(float(log_loss(y_t, y_proba)), 4)
            except Exception:
                pass

        return metrics

    def _evaluate_regression(
        self, y_true: Sequence[Any], y_pred: Sequence[Any]
    ) -> Dict[str, Any]:
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        r2 = float(r2_score(y_t, y_p))
        mae = float(mean_absolute_error(y_t, y_p))
        mse = float(mean_squared_error(y_t, y_p))
        rmse = float(np.sqrt(mse))

        return {
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "n_samples": len(y_t),
        }


def evaluate_nlp_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    task_type: str = "text_classification",
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convenience functional evaluation helper."""
    evaluator = NLPEvaluator(task_type=task_type)
    return evaluator.evaluate(y_true=y_true, y_pred=y_pred, y_proba=y_proba)
