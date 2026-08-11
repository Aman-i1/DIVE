"""Probability Calibration & Reliability Diagnostics.

Fits post-hoc probability calibrators (Platt scaling / Sigmoid, Isotonic regression)
on out-of-fold predictions. Evaluates Brier score, Expected Calibration Error (ECE),
reliability curves, and optimal decision threshold selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score


@dataclass
class CalibrationReport:
    """Outcome of probability calibration evaluation."""

    method: str
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    optimal_threshold: float
    best_f1: float
    reliability_bins_before: Dict[str, List[float]]
    reliability_bins_after: Dict[str, List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "brier_before": round(self.brier_before, 4),
            "brier_after": round(self.brier_after, 4),
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "best_f1": round(self.best_f1, 4),
            "reliability_bins_before": self.reliability_bins_before,
            "reliability_bins_after": self.reliability_bins_after,
        }

    def render(self) -> str:
        lines = [
            "PROBABILITY CALIBRATION REPORT",
            "==============================",
            f"Method              : {self.method.upper()}",
            f"Brier Score         : {self.brier_before:.4f} -> {self.brier_after:.4f} "
            f"({'✓ IMPROVED' if self.brier_after < self.brier_before else 'NO CHANGE'})",
            f"Expected Calib Error: {self.ece_before:.4f} -> {self.ece_after:.4f}",
            f"Optimal Threshold   : {self.optimal_threshold:.4f} (Peak F1: {self.best_f1:.4f})",
        ]
        return "\n".join(lines)


class ProbabilityCalibrator:
    """Fits Platt scaling or Isotonic regression to calibrate predicted probabilities."""

    def __init__(self, method: str = "platt", n_bins: int = 10) -> None:
        self.method = method.lower()
        self.n_bins = n_bins
        self.calibrator_: Any = None
        self.is_fitted_: bool = False

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration model on out-of-fold probabilities."""
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        if y_proba.ndim > 1 and y_proba.shape[1] == 2:
            p = y_proba[:, 1]
        else:
            p = y_proba

        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(p, y_true)
        else:  # platt
            # Logit transformation for LogisticRegression Platt scaling
            p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
            logits = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
            self.calibrator_ = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator_.fit(logits, y_true)

        self.is_fitted_ = True
        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities using fitted calibrator."""
        if not self.is_fitted_:
            return y_proba

        p = y_proba[:, 1] if (y_proba.ndim > 1 and y_proba.shape[1] == 2) else y_proba

        if self.method == "isotonic":
            calibrated_p = self.calibrator_.predict(p)
        else:
            p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
            logits = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
            calibrated_p = self.calibrator_.predict_proba(logits)[:, 1]

        calibrated_p = np.clip(calibrated_p, 0.0, 1.0)

        if y_proba.ndim > 1 and y_proba.shape[1] == 2:
            return np.column_stack([1.0 - calibrated_p, calibrated_p])
        return calibrated_p

    def evaluate(self, y_true: np.ndarray, y_proba: np.ndarray) -> CalibrationReport:
        """Compare calibration metrics before & after fitting."""
        p_before = y_proba[:, 1] if (y_proba.ndim > 1 and y_proba.shape[1] == 2) else y_proba
        
        # Fit calibrator on copy if not already fitted
        if not self.is_fitted_:
            self.fit(y_true, p_before)

        p_after = self.calibrate(p_before)
        if p_after.ndim > 1:
            p_after = p_after[:, 1]

        brier_before = float(brier_score_loss(y_true, p_before))
        brier_after = float(brier_score_loss(y_true, p_after))

        ece_before = float(self._compute_ece(y_true, p_before, self.n_bins))
        ece_after = float(self._compute_ece(y_true, p_after, self.n_bins))

        # Reliability curves
        prob_true_b, prob_pred_b = calibration_curve(y_true, p_before, n_bins=self.n_bins)
        prob_true_a, prob_pred_a = calibration_curve(y_true, p_after, n_bins=self.n_bins)

        # Threshold search
        thresholds = np.linspace(0.05, 0.95, 91)
        best_thresh = 0.5
        best_f1 = 0.0
        for t in thresholds:
            preds = (p_after >= t).astype(int)
            f1 = float(f1_score(y_true, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        return CalibrationReport(
            method=self.method,
            brier_before=brier_before,
            brier_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
            optimal_threshold=best_thresh,
            best_f1=best_f1,
            reliability_bins_before={"true": prob_true_b.tolist(), "pred": prob_pred_b.tolist()},
            reliability_bins_after={"true": prob_true_a.tolist(), "pred": prob_pred_a.tolist()},
        )

    @staticmethod
    def _compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_proba[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        return float(ece)
