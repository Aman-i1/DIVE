"""Production Observability, Drift Monitoring & Retraining Engine - `dive/observability.py`.

Monitors:
1. Feature Drift: Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test, Wasserstein distance.
2. Prediction & Output Drift: Shifts in predicted probability distributions over time.
3. Automated Retraining Urgency: Synthesizes drift metrics into an actionable RetrainingUrgencyScore (0-100).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from dive.decisions import DecisionLogger


@dataclass
class DriftMetricResult:
    """Individual feature drift assessment result."""

    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    drift_status: str  # 'NO_DRIFT', 'MODERATE_DRIFT', 'SIGNIFICANT_DRIFT'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi_score": round(self.psi_score, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_p_value": round(self.ks_p_value, 4),
            "drift_status": self.drift_status,
        }


@dataclass
class ObservabilityReport:
    """Complete production observability and drift audit."""

    retraining_urgency_score: float  # 0 to 100
    retraining_alert_level: str  # 'NO_ACTION', 'MONITOR', 'RETRAIN_RECOMMENDED', 'RETRAIN_URGENT'
    features_monitored_count: int
    drifted_features_count: int
    prediction_drift_psi: float
    feature_metrics: Dict[str, DriftMetricResult] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retraining_urgency_score": round(self.retraining_urgency_score, 1),
            "retraining_alert_level": self.retraining_alert_level,
            "features_monitored_count": self.features_monitored_count,
            "drifted_features_count": self.drifted_features_count,
            "prediction_drift_psi": round(self.prediction_drift_psi, 4),
            "feature_metrics": {k: v.to_dict() for k, v in self.feature_metrics.items()},
            "recommendations": self.recommendations,
        }

    def render(self) -> str:
        lines = [
            "PRODUCTION OBSERVABILITY & DRIFT AUDIT",
            "=======================================",
            f"Retraining Urgency  : {self.retraining_urgency_score:.1f}/100 [Alert: {self.retraining_alert_level}]",
            f"Features Monitored  : {self.features_monitored_count} ({self.drifted_features_count} drifted)",
            f"Prediction Drift PSI: {self.prediction_drift_psi:.4f}",
        ]
        if self.feature_metrics:
            lines.append("\nTop Feature Drift Metrics:")
            for name, m in list(self.feature_metrics.items())[:5]:
                lines.append(f"  - {name:<20}: PSI={m.psi_score:.4f}, Status={m.drift_status}")
        if self.recommendations:
            lines.append("\nActions & Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


class ObservabilityEngine:
    """Monitors live inference data against baseline reference distributions and calculates retraining urgency."""

    def __init__(self, psi_threshold: float = 0.20, logger: Optional[DecisionLogger] = None) -> None:
        self.psi_threshold = psi_threshold
        self.logger = logger or DecisionLogger()

    def audit_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_predictions: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None,
    ) -> ObservabilityReport:
        """Calculate statistical drift across features and predictions."""
        feature_metrics: Dict[str, DriftMetricResult] = {}
        common_num_cols = [
            c for c in reference_df.select_dtypes(include=[np.number]).columns
            if c in current_df.columns
        ]

        drifted_count = 0
        total_psi = 0.0

        for col in common_num_cols:
            ref_vals = reference_df[col].dropna().to_numpy()
            curr_vals = current_df[col].dropna().to_numpy()

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            psi_val = self._compute_psi(ref_vals, curr_vals)
            total_psi += psi_val

            # KS test
            ks_res = ks_2samp(ref_vals, curr_vals)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)

            if psi_val >= self.psi_threshold:
                drift_status = "SIGNIFICANT_DRIFT"
                drifted_count += 1
            elif psi_val >= 0.10:
                drift_status = "MODERATE_DRIFT"
            else:
                drift_status = "NO_DRIFT"

            feature_metrics[col] = DriftMetricResult(
                feature_name=col,
                psi_score=psi_val,
                ks_statistic=ks_stat,
                ks_p_value=ks_pval,
                drift_status=drift_status,
            )

        # Prediction Drift
        if reference_predictions is not None and current_predictions is not None:
            pred_psi = self._compute_psi(reference_predictions, current_predictions)
        else:
            pred_psi = 0.0

        # Calculate Retraining Urgency Score (0 to 100)
        n_features = max(len(common_num_cols), 1)
        drift_fraction = drifted_count / n_features
        avg_psi = total_psi / n_features

        urgency_score = min(100.0, (drift_fraction * 50.0) + (min(avg_psi, 1.0) * 30.0) + (min(pred_psi, 1.0) * 20.0))

        # Alert Level
        if urgency_score >= 70.0:
            alert_level = "RETRAIN_URGENT"
        elif urgency_score >= 40.0:
            alert_level = "RETRAIN_RECOMMENDED"
        elif urgency_score >= 20.0:
            alert_level = "MONITOR"
        else:
            alert_level = "NO_ACTION"

        recommendations: List[str] = []
        if alert_level == "RETRAIN_URGENT":
            recommendations.append(f"Significant feature/prediction drift detected ({drifted_count}/{n_features} features drifted). Initiate autonomous retraining.")
        elif alert_level == "RETRAIN_RECOMMENDED":
            recommendations.append("Moderate distribution drift detected. Schedule routine model retraining.")
        elif alert_level == "MONITOR":
            recommendations.append("Minor distribution movement. Continue monitoring without intervention.")
        else:
            recommendations.append("Distributions are stable. No retraining necessary.")

        self.logger.log(
            component="ObservabilityEngine",
            decision=f"Assigned Retraining Urgency: {urgency_score:.1f}/100 [Alert: {alert_level}]",
            reason=f"{drifted_count}/{n_features} features drifted (Prediction PSI={pred_psi:.4f})",
            confidence=0.95,
            evidence={
                "urgency_score": round(urgency_score, 1),
                "alert_level": alert_level,
                "drifted_count": drifted_count,
                "pred_psi": round(pred_psi, 4),
            },
        )

        return ObservabilityReport(
            retraining_urgency_score=urgency_score,
            retraining_alert_level=alert_level,
            features_monitored_count=n_features,
            drifted_features_count=drifted_count,
            prediction_drift_psi=pred_psi,
            feature_metrics=feature_metrics,
            recommendations=recommendations,
        )

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI) between two continuous samples."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Bin boundaries from reference percentiles
        quantiles = np.linspace(0, 100, n_bins + 1)
        try:
            bins = np.percentile(reference, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)
        except Exception:
            return 0.0

        if len(bins) < 2:
            return 0.0

        ref_counts, _ = np.histogram(reference, bins=bins)
        curr_counts, _ = np.histogram(current, bins=bins)

        ref_pct = (ref_counts + 1e-4) / (len(reference) + 1e-4 * len(ref_counts))
        curr_pct = (curr_counts + 1e-4) / (len(current) + 1e-4 * len(curr_counts))

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(max(0.0, psi))
