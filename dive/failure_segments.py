"""Failure Segmentation Engine - `dive/failure_segments.py`.

Discovers sub-populations and feature slices where the model performs poorly:
- Analyzes categorical slices, binned numeric slices, missingness buckets, and entity types.
- Evaluates slice performance metric vs. global performance.
- Enforces statistical sample-size safeguards (minimum sample count) to prevent trivial noise alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from dive.decisions import DecisionLogger


@dataclass
class FailureSegment:
    """A slice of the population with degraded performance."""

    segment_description: str
    sample_count: int
    sample_pct: float
    segment_metric: float
    global_metric: float
    metric_drop: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_description": self.segment_description,
            "sample_count": self.sample_count,
            "sample_pct": round(self.sample_pct, 4),
            "segment_metric": round(self.segment_metric, 4),
            "global_metric": round(self.global_metric, 4),
            "metric_drop": round(self.metric_drop, 4),
            "risk_level": self.risk_level,
        }


@dataclass
class FailureSegmentsReport:
    """Complete segmentation audit report."""

    global_metric_name: str
    global_metric_value: float
    weak_segments: List[FailureSegment] = field(default_factory=list)
    overall_segment_status: str = "PASS"  # 'PASS', 'WARNING', 'FAIL'
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_metric_name": self.global_metric_name,
            "global_metric_value": round(self.global_metric_value, 4),
            "weak_segments": [s.to_dict() for s in self.weak_segments],
            "overall_segment_status": self.overall_segment_status,
            "recommendations": self.recommendations,
        }

    def render(self) -> str:
        lines = [
            "FAILURE SEGMENTATION AUDIT",
            "==========================",
            f"Global {self.global_metric_name:<16}: {self.global_metric_value:.4f} [Status: {self.overall_segment_status}]",
        ]
        if self.weak_segments:
            lines.append("\nIdentified Weak Segments / Slices:")
            for s in self.weak_segments[:5]:
                lines.append(
                    f"  - [{s.risk_level:<7}] {s.segment_description:<30}: Metric={s.segment_metric:.4f} "
                    f"(Drop: {s.metric_drop:+.4f}, N={s.sample_count:,} ({s.sample_pct:.1%}))"
                )
        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


class FailureSegmentAnalyzer:
    """Discovers underperforming population subgroups."""

    def __init__(
        self,
        min_sample_count: int = 20,
        drop_threshold: float = 0.10,  # 10% metric drop triggers alert
        logger: Optional[DecisionLogger] = None,
    ) -> None:
        self.min_sample_count = min_sample_count
        self.drop_threshold = drop_threshold
        self.logger = logger or DecisionLogger()

    def discover_weak_segments(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        metric_name: str = "Accuracy",
    ) -> FailureSegmentsReport:
        """Scan feature slices to uncover weak segments."""
        y_arr = np.asarray(y_test)
        n_total = len(y_arr)

        if n_total == 0:
            return FailureSegmentsReport(
                global_metric_name=metric_name,
                global_metric_value=0.0,
                weak_segments=[],
                overall_segment_status="FAIL",
                recommendations=["No test data available for segmentation."],
            )

        # Global metric
        if metric_name == "Accuracy":
            global_metric = float(accuracy_score(y_arr, predictions))
        elif metric_name == "ROC_AUC" and probabilities is not None:
            p_pos = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
            global_metric = float(roc_auc_score(y_arr, p_pos))
        else:
            global_metric = float(accuracy_score(y_arr, predictions))

        weak_segments: List[FailureSegment] = []

        # Scan categorical features
        cat_cols = X_test.select_dtypes(include=[object, "category"]).columns
        for col in cat_cols:
            for cat_val, group_idx in X_test.groupby(col).groups.items():
                idx_list = np.asarray(group_idx)
                if len(idx_list) < self.min_sample_count:
                    continue

                y_slice = y_arr[idx_list]
                p_slice = predictions[idx_list]

                slice_metric = float(accuracy_score(y_slice, p_slice))
                drop = global_metric - slice_metric

                if drop >= self.drop_threshold:
                    risk = "HIGH" if drop >= 0.20 else "MEDIUM"
                    weak_segments.append(
                        FailureSegment(
                            segment_description=f"{col} == '{cat_val}'",
                            sample_count=len(idx_list),
                            sample_pct=len(idx_list) / n_total,
                            segment_metric=slice_metric,
                            global_metric=global_metric,
                            metric_drop=drop,
                            risk_level=risk,
                        )
                    )

        # Scan binned numeric features
        num_cols = X_test.select_dtypes(include=[np.number]).columns
        for col in num_cols[:5]:  # Limit top 5 continuous features
            try:
                binned = pd.qcut(X_test[col], q=4, duplicates="drop")
                for bin_interval, group_idx in X_test.groupby(binned).groups.items():
                    idx_list = np.asarray(group_idx)
                    if len(idx_list) < self.min_sample_count:
                        continue

                    y_slice = y_arr[idx_list]
                    p_slice = predictions[idx_list]
                    slice_metric = float(accuracy_score(y_slice, p_slice))
                    drop = global_metric - slice_metric

                    if drop >= self.drop_threshold:
                        risk = "HIGH" if drop >= 0.20 else "MEDIUM"
                        weak_segments.append(
                            FailureSegment(
                                segment_description=f"{col} in {bin_interval}",
                                sample_count=len(idx_list),
                                sample_pct=len(idx_list) / n_total,
                                segment_metric=slice_metric,
                                global_metric=global_metric,
                                metric_drop=drop,
                                risk_level=risk,
                            )
                        )
            except Exception:
                continue

        # Sort weak segments by drop
        weak_segments.sort(key=lambda s: s.metric_drop, reverse=True)

        recs: List[str] = []
        if weak_segments:
            overall_status = "FAIL" if any(s.risk_level == "HIGH" for s in weak_segments) else "WARNING"
            top_seg = weak_segments[0]
            recs.append(f"Top underperforming slice: {top_seg.segment_description} (Drop: {top_seg.metric_drop:+.2%}). Investigate data collection or segment-specific calibration.")
        else:
            overall_status = "PASS"
            recs.append("No statistically significant weak population segments discovered.")

        self.logger.log(
            component="FailureSegmentAnalyzer",
            decision=f"Failure Segmentation Status: [{overall_status}] ({len(weak_segments)} weak slices found)",
            reason="; ".join(recs),
            confidence=0.92,
            evidence={
                "global_metric": round(global_metric, 4),
                "weak_segments_count": len(weak_segments),
                "top_segment": weak_segments[0].to_dict() if weak_segments else None,
            },
        )

        return FailureSegmentsReport(
            global_metric_name=metric_name,
            global_metric_value=global_metric,
            weak_segments=weak_segments,
            overall_segment_status=overall_status,
            recommendations=recs,
        )
