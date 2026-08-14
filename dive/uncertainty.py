"""Conformal Prediction & Uncertainty Quantification Engine - `dive/uncertainty.py`.

Provides distribution-free conformal prediction intervals for regression, conformal
prediction sets for multi-class classification, and decomposition of epistemic
(model disagreement) vs. aleatoric (data noise / residual variance) uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ConformalIntervalResult:
    """Conformal prediction intervals for regression."""

    predictions: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    confidence_level: float  # e.g., 0.90 or 0.95
    quantile_q: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_level": self.confidence_level,
            "quantile_q": round(float(self.quantile_q), 4),
            "mean_interval_width": round(float(np.mean(self.upper_bounds - self.lower_bounds)), 4),
        }


@dataclass
class ConformalSetResult:
    """Conformal prediction sets for classification."""

    predictions: np.ndarray
    prediction_sets: List[List[Any]]
    confidence_level: float
    quantile_q: float

    def to_dict(self) -> Dict[str, Any]:
        avg_set_size = float(np.mean([len(s) for s in self.prediction_sets])) if self.prediction_sets else 1.0
        return {
            "confidence_level": self.confidence_level,
            "quantile_q": round(float(self.quantile_q), 4),
            "average_set_size": round(avg_set_size, 2),
        }


@dataclass
class UncertaintyDecomposition:
    """Decomposition of uncertainty into Epistemic and Aleatoric components."""

    epistemic_uncertainty: np.ndarray  # Model uncertainty (ensemble variance)
    aleatoric_uncertainty: np.ndarray  # Data/Observation noise
    total_uncertainty: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_epistemic": round(float(np.mean(self.epistemic_uncertainty)), 4),
            "mean_aleatoric": round(float(np.mean(self.aleatoric_uncertainty)), 4),
            "mean_total": round(float(np.mean(self.total_uncertainty)), 4),
        }


class ConformalPredictor:
    """Distribution-free split conformal calibration engine."""

    def __init__(self, problem_type: str = "classification", confidence_level: float = 0.95) -> None:
        self.problem_type = problem_type
        self.confidence_level = confidence_level
        self.quantile_q_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_calibrated_: bool = False

    def calibrate(
        self,
        y_true_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "ConformalPredictor":
        """Fit conformal non-conformity scores on calibration holdout data."""
        y_true_cal = np.asarray(y_true_cal)
        y_pred_cal = np.asarray(y_pred_cal)
        n = len(y_true_cal)

        if n == 0:
            self.quantile_q_ = 0.0
            self.is_calibrated_ = True
            return self

        # Finite-sample correction quantile: ceil((n + 1) * (1 - alpha)) / n
        alpha = 1.0 - self.confidence_level
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)

        if self.problem_type == "regression":
            # Residual absolute errors as non-conformity scores
            residuals = np.abs(y_true_cal - y_pred_cal)
            self.quantile_q_ = float(np.quantile(residuals, q_level))
        else:
            # Classification non-conformity score: 1 - P(true_class)
            self.classes_ = np.asarray(classes) if classes is not None else np.unique(y_true_cal)
            if y_pred_cal.ndim > 1:
                # Array of class probabilities
                true_class_probs = []
                for i, true_label in enumerate(y_true_cal):
                    idx = np.where(self.classes_ == true_label)[0]
                    if len(idx) > 0 and idx[0] < y_pred_cal.shape[1]:
                        true_class_probs.append(y_pred_cal[i, idx[0]])
                    else:
                        true_class_probs.append(0.0)
                non_conformity = 1.0 - np.array(true_class_probs)
            else:
                # 1D probability array for binary positive class
                p = y_pred_cal
                non_conformity = np.where(y_true_cal == 1, 1.0 - p, p)

            self.quantile_q_ = float(np.quantile(non_conformity, q_level))

        self.is_calibrated_ = True
        return self

    def predict_interval(self, y_pred: np.ndarray) -> ConformalIntervalResult:
        """Predict conformal prediction intervals for regression."""
        if not self.is_calibrated_ or self.quantile_q_ is None:
            raise ValueError("ConformalPredictor must be calibrated with calibrate() before predicting intervals.")

        y_pred = np.asarray(y_pred)
        lower = y_pred - self.quantile_q_
        upper = y_pred + self.quantile_q_

        return ConformalIntervalResult(
            predictions=y_pred,
            lower_bounds=lower,
            upper_bounds=upper,
            confidence_level=self.confidence_level,
            quantile_q=self.quantile_q_,
        )

    def predict_set(self, y_proba: np.ndarray) -> ConformalSetResult:
        """Predict conformal prediction sets for classification."""
        if not self.is_calibrated_ or self.quantile_q_ is None:
            raise ValueError("ConformalPredictor must be calibrated with calibrate() before predicting sets.")

        y_proba = np.asarray(y_proba)
        threshold = 1.0 - self.quantile_q_

        prediction_sets: List[List[Any]] = []
        point_preds: List[Any] = []

        if y_proba.ndim > 1:
            classes = self.classes_ if self.classes_ is not None else np.arange(y_proba.shape[1])
            for p_row in y_proba:
                included = [classes[idx] for idx, p_val in enumerate(p_row) if p_val >= threshold]
                if not included:
                    # Fallback to top-1 if threshold is strict
                    included = [classes[np.argmax(p_row)]]
                prediction_sets.append(included)
                point_preds.append(classes[np.argmax(p_row)])
        else:
            classes = self.classes_ if self.classes_ is not None else np.array([0, 1])
            for p_val in y_proba:
                included = []
                if (1.0 - p_val) >= threshold:
                    included.append(classes[0])
                if p_val >= threshold:
                    included.append(classes[1])
                if not included:
                    included = [classes[int(p_val >= 0.5)]]
                prediction_sets.append(included)
                point_preds.append(classes[int(p_val >= 0.5)])

        return ConformalSetResult(
            predictions=np.array(point_preds),
            prediction_sets=prediction_sets,
            confidence_level=self.confidence_level,
            quantile_q=self.quantile_q_,
        )

    @staticmethod
    def decompose_uncertainty(
        ensemble_predictions: np.ndarray,  # shape: (n_estimators, n_samples)
        problem_type: str = "regression",
    ) -> UncertaintyDecomposition:
        """Decompose uncertainty into Epistemic (ensemble disagreement) & Aleatoric components."""
        # Epistemic = variance across ensemble members
        epistemic = np.var(ensemble_predictions, axis=0)

        if problem_type == "regression":
            # For regression, aleatoric estimated as mean predicted residual variance
            aleatoric = np.std(ensemble_predictions, axis=0) * 0.5
        else:
            # For classification, entropy across averaged probabilities
            mean_probs = np.mean(ensemble_predictions, axis=0)
            mean_probs = np.clip(mean_probs, 1e-6, 1.0 - 1e-6)
            aleatoric = -(mean_probs * np.log2(mean_probs) + (1.0 - mean_probs) * np.log2(1.0 - mean_probs))

        total = epistemic + aleatoric
        return UncertaintyDecomposition(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
        )
