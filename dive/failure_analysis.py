"""Post-Training Model Failure & Segment Performance Analyzer.

Analyzes model errors, residual distributions, false positive/negative confidence profiles,
and discovers poor-performing data slices ('performance segments').
Distinguishes correlation from causation in segment findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class FailureAnalysisResult:
    """Output of ModelFailureAnalyzer.analyze()."""

    problem_type: str
    overall_metrics: Dict[str, float]
    performance_segments: List[Dict[str, Any]] = field(default_factory=list)
    worst_predictions: List[Dict[str, Any]] = field(default_factory=list)
    confusion_matrix_data: Optional[List[List[int]]] = None
    residual_stats: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "overall_metrics": self.overall_metrics,
            "performance_segments": self.performance_segments,
            "worst_predictions": self.worst_predictions,
            "confusion_matrix": self.confusion_matrix_data,
            "residual_stats": self.residual_stats,
        }

    def render(self) -> str:
        lines = [
            "MODEL FAILURE & SEGMENT ANALYSIS",
            "================================",
        ]
        lines.append("Overall Evaluation Metrics:")
        for k, v in self.overall_metrics.items():
            lines.append(f"  - {k:<20}: {v:.4f}")

        if self.performance_segments:
            lines.append("")
            lines.append("Subgroup Performance Segments (Correlation, Not Causation):")
            for seg in self.performance_segments:
                lines.append(
                    f"  ⚠ Slice '{seg['feature']} == {seg['value']}': "
                    f"Sample count: {seg['n_samples']}, Metric: {seg['metric_name']} = {seg['metric_value']:.4f} "
                    f"(vs Baseline {seg['baseline_value']:.4f})"
                )

        if self.worst_predictions:
            lines.append("")
            lines.append(f"Top {len(self.worst_predictions)} Worst Predictions:")
            for wp in self.worst_predictions[:5]:
                lines.append(f"  - Row {wp.get('index')}: True={wp.get('true')}, Pred={wp.get('pred')}, Error={wp.get('error', 0):.4f}")

        return "\n".join(lines)


class ModelFailureAnalyzer:
    """Post-fit failure diagnostic engine."""

    def __init__(self, problem_type: str = "classification") -> None:
        self.problem_type = problem_type

    def analyze(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        X_test: Optional[pd.DataFrame] = None,
        top_k_segments: int = 5,
    ) -> FailureAnalysisResult:
        """Run complete failure audit."""
        y_true_clean = y_true.reset_index(drop=True)

        if self.problem_type == "classification":
            return self._analyze_classification(
                y_true_clean, y_pred, y_proba, X_test, top_k_segments
            )
        else:
            return self._analyze_regression(
                y_true_clean, y_pred, X_test, top_k_segments
            )

    def _analyze_classification(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        X_test: Optional[pd.DataFrame],
        top_k: int,
    ) -> FailureAnalysisResult:
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        metrics = {
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Precision": prec,
            "Recall": rec,
        }

        cm = confusion_matrix(y_true, y_pred).tolist()

        if y_proba is not None:
            try:
                if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                    p = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    metrics["ROC AUC"] = float(roc_auc_score(y_true, p))
            except Exception:
                pass

        # Subgroup failure slice detection
        segments = []
        if X_test is not None and len(X_test) == len(y_true):
            segments = self._find_classification_segments(
                X_test, y_true, y_pred, macro_f1, top_k
            )

        # Worst predictions (highest error confidence)
        worst = []
        if y_proba is not None:
            p_correct = np.where(y_pred == y_true, 1.0, 0.0)
            errors = 1.0 - p_correct
            top_err_idx = np.argsort(-errors)[:5]
            for idx in top_err_idx:
                worst.append({
                    "index": int(idx),
                    "true": str(y_true.iloc[idx]),
                    "pred": str(y_pred[idx]),
                    "error": float(errors[idx]),
                })

        return FailureAnalysisResult(
            problem_type="classification",
            overall_metrics=metrics,
            performance_segments=segments,
            worst_predictions=worst,
            confusion_matrix_data=cm,
        )

    def _analyze_regression(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X_test: Optional[pd.DataFrame],
        top_k: int,
    ) -> FailureAnalysisResult:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        residuals = y_true.to_numpy() - y_pred
        abs_residuals = np.abs(residuals)

        res_stats = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "max_error": float(np.max(abs_residuals)),
        }

        # Subgroup slice analysis
        segments = []
        if X_test is not None and len(X_test) == len(y_true):
            segments = self._find_regression_segments(
                X_test, y_true, y_pred, mae, top_k
            )

        # Top worst predictions
        top_err_idx = np.argsort(-abs_residuals)[:5]
        worst = [
            {
                "index": int(idx),
                "true": float(y_true.iloc[idx]),
                "pred": float(y_pred[idx]),
                "error": float(abs_residuals[idx]),
            }
            for idx in top_err_idx
        ]

        return FailureAnalysisResult(
            problem_type="regression",
            overall_metrics=metrics,
            performance_segments=segments,
            worst_predictions=worst,
            residual_stats=res_stats,
        )

    def _find_classification_segments(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        baseline_f1: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        segments = []
        for col in X.columns[:20]:
            series = X[col]
            if pd.api.types.is_categorical_dtype(series) or series.dtype == object or series.nunique() < 10:
                for val, grp_idx in series.groupby(series).groups.items():
                    if len(grp_idx) >= 15:
                        sub_true = y_true.iloc[grp_idx]
                        sub_pred = y_pred[grp_idx]
                        sub_f1 = float(f1_score(sub_true, sub_pred, average="macro", zero_division=0))
                        if sub_f1 < baseline_f1 - 0.08:
                            segments.append({
                                "feature": str(col),
                                "value": str(val),
                                "n_samples": len(grp_idx),
                                "metric_name": "Macro F1",
                                "metric_value": sub_f1,
                                "baseline_value": baseline_f1,
                            })
        segments.sort(key=lambda s: s["metric_value"])
        return segments[:top_k]

    def _find_regression_segments(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        baseline_mae: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        segments = []
        abs_err = np.abs(y_true.to_numpy() - y_pred)
        for col in X.columns[:20]:
            series = X[col]
            if pd.api.types.is_categorical_dtype(series) or series.dtype == object or series.nunique() < 10:
                for val, grp_idx in series.groupby(series).groups.items():
                    if len(grp_idx) >= 15:
                        sub_mae = float(np.mean(abs_err[grp_idx]))
                        if sub_mae > baseline_mae * 1.25:
                            segments.append({
                                "feature": str(col),
                                "value": str(val),
                                "n_samples": len(grp_idx),
                                "metric_name": "MAE",
                                "metric_value": sub_mae,
                                "baseline_value": baseline_mae,
                            })
        segments.sort(key=lambda s: -s["metric_value"])
        return segments[:top_k]
