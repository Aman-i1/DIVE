"""Calibrated Multi-Layer Stacking & Convex Blending Engine - `dive/stacking_calibrated.py`.

Optimizes out-of-fold calibrated base model predictions using:
1. Constrained convex blend weights (Non-Negative Least Squares / Nelder-Mead: sum(w)=1, w_i >= 0).
2. Multi-layer meta-learners (Ridge / Logistic Regression) trained on calibrated OOF predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

from dive.calibration import ProbabilityCalibrator


@dataclass
class EnsembleWeightsResult:
    """Optimal convex blend weights across base estimators."""

    model_names: List[str]
    weights: np.ndarray
    loss_before_blend: float
    loss_after_blend: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_names": self.model_names,
            "weights": {name: round(float(w), 4) for name, w in zip(self.model_names, self.weights)},
            "loss_before_blend": round(float(self.loss_before_blend), 4),
            "loss_after_blend": round(float(self.loss_after_blend), 4),
        }


class CalibratedStackingEnsemble:
    """Calibrated Stacking & Convex Blending Ensemble."""

    def __init__(
        self,
        problem_type: str = "classification",
        method: str = "blend",  # 'blend' (convex weights) or 'stack' (meta-learner)
    ) -> None:
        self.problem_type = problem_type
        self.method = method
        self.model_names_: List[str] = []
        self.weights_: Optional[np.ndarray] = None
        self.meta_learner_: Any = None
        self.calibrators_: Dict[str, ProbabilityCalibrator] = {}
        self.is_fitted_: bool = False

    def fit_blend(
        self,
        oof_predictions: Dict[str, np.ndarray],  # model_name -> 1D/2D OOF prediction array
        y_true: np.ndarray,
    ) -> EnsembleWeightsResult:
        """Fit convex weights w_i >= 0, sum(w)=1 minimizing OOF loss."""
        self.model_names_ = list(oof_predictions.keys())
        k = len(self.model_names_)
        y_true = np.asarray(y_true)

        # Build feature matrix
        P_list = []
        for name in self.model_names_:
            p = oof_predictions[name]
            if self.problem_type == "classification":
                # Calibrate probabilities
                calibrator = ProbabilityCalibrator(method="platt")
                p_cal = calibrator.fit(y_true, p).calibrate(p)
                self.calibrators_[name] = calibrator
                prob = p_cal[:, 1] if (p_cal.ndim > 1 and p_cal.shape[1] == 2) else p_cal
                P_list.append(prob)
            else:
                P_list.append(p)

        P = np.column_stack(P_list)  # (n_samples, k_models)

        # Baseline loss (uniform weights)
        uniform_weights = np.ones(k) / k
        initial_pred = P @ uniform_weights
        if self.problem_type == "classification":
            initial_pred = np.clip(initial_pred, 1e-6, 1.0 - 1e-6)
            loss_before = float(log_loss(y_true, initial_pred))
        else:
            loss_before = float(mean_squared_error(y_true, initial_pred))

        # Convex optimization objective
        def loss_func(weights: np.ndarray) -> float:
            weights = np.asarray(weights)
            weights = weights / max(np.sum(weights), 1e-8)
            pred = P @ weights
            if self.problem_type == "classification":
                pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
                return float(log_loss(y_true, pred))
            else:
                return float(mean_squared_error(y_true, pred))

        # Constrained optimization
        bounds = [(0.0, 1.0) for _ in range(k)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        res = minimize(
            loss_func,
            x0=uniform_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if res.success:
            optimal_weights = res.x / np.sum(res.x)
        else:
            optimal_weights = uniform_weights

        self.weights_ = optimal_weights
        loss_after = float(loss_func(optimal_weights))
        self.is_fitted_ = True

        return EnsembleWeightsResult(
            model_names=self.model_names_,
            weights=optimal_weights,
            loss_before_blend=loss_before,
            loss_after_blend=loss_after,
        )

    def fit_meta_learner(
        self,
        oof_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Fit a meta-estimator (LogisticRegression or Ridge) on calibrated OOF predictions."""
        self.model_names_ = list(oof_predictions.keys())
        y_true = np.asarray(y_true)

        P_list = []
        for name in self.model_names_:
            p = oof_predictions[name]
            if self.problem_type == "classification":
                calibrator = ProbabilityCalibrator(method="platt")
                p_cal = calibrator.fit(y_true, p).calibrate(p)
                self.calibrators_[name] = calibrator
                prob = p_cal[:, 1] if (p_cal.ndim > 1 and p_cal.shape[1] == 2) else p_cal
                P_list.append(prob)
            else:
                P_list.append(p)

        P = np.column_stack(P_list)

        if self.problem_type == "classification":
            self.meta_learner_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        else:
            self.meta_learner_ = Ridge(alpha=1.0)

        self.meta_learner_.fit(P, y_true)
        self.method = "stack"
        self.is_fitted_ = True

    def predict(self, test_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble predictions on test set."""
        if not self.is_fitted_:
            raise ValueError("CalibratedStackingEnsemble must be fitted before predict().")

        P_list = []
        for name in self.model_names_:
            p = test_predictions[name]
            if self.problem_type == "classification" and name in self.calibrators_:
                p_cal = self.calibrators_[name].calibrate(p)
                prob = p_cal[:, 1] if (p_cal.ndim > 1 and p_cal.shape[1] == 2) else p_cal
                P_list.append(prob)
            else:
                P_list.append(p)

        P = np.column_stack(P_list)

        if self.method == "blend" and self.weights_ is not None:
            combined = P @ self.weights_
            if self.problem_type == "classification":
                return (combined >= 0.5).astype(int)
            return combined
        elif self.method == "stack" and self.meta_learner_ is not None:
            return self.meta_learner_.predict(P)
        else:
            # Uniform fallback
            combined = np.mean(P, axis=1)
            if self.problem_type == "classification":
                return (combined >= 0.5).astype(int)
            return combined

    def predict_proba(self, test_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble probabilities on test set."""
        if not self.is_fitted_:
            raise ValueError("CalibratedStackingEnsemble must be fitted before predict_proba().")

        P_list = []
        for name in self.model_names_:
            p = test_predictions[name]
            if name in self.calibrators_:
                p_cal = self.calibrators_[name].calibrate(p)
                prob = p_cal[:, 1] if (p_cal.ndim > 1 and p_cal.shape[1] == 2) else p_cal
                P_list.append(prob)
            else:
                prob = p[:, 1] if (p.ndim > 1 and p.shape[1] == 2) else p
                P_list.append(prob)

        P = np.column_stack(P_list)

        if self.method == "blend" and self.weights_ is not None:
            prob_pos = np.clip(P @ self.weights_, 0.0, 1.0)
            return np.column_stack([1.0 - prob_pos, prob_pos])
        elif self.method == "stack" and self.meta_learner_ is not None:
            return self.meta_learner_.predict_proba(P)
        else:
            prob_pos = np.clip(np.mean(P, axis=1), 0.0, 1.0)
            return np.column_stack([1.0 - prob_pos, prob_pos])
