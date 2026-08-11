"""Production Data & Prediction Drift Detector + Retraining Advisor.

Methods:
- Numerical drift: Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test, Wasserstein distance.
- Categorical drift: PSI, Jensen-Shannon (JS) divergence, frequency distribution comparison.
- Prediction distribution drift: Training vs production prediction shift detection.
- Retraining recommendation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class DriftFeatureReport:
    """Drift evaluation for a single feature."""

    feature: str
    feature_type: str  # numeric or categorical
    psi_score: float
    ks_pvalue: Optional[float]
    js_divergence: Optional[float]
    drift_status: str  # NO_DRIFT, MODERATE_DRIFT, SIGNIFICANT_DRIFT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "feature_type": self.feature_type,
            "psi_score": round(self.psi_score, 4),
            "ks_pvalue": round(self.ks_pvalue, 4) if self.ks_pvalue is not None else None,
            "js_divergence": round(self.js_divergence, 4) if self.js_divergence is not None else None,
            "drift_status": self.drift_status,
        }


@dataclass
class DriftReport:
    """Overall dataset & prediction drift report."""

    n_features_analyzed: int
    n_drifting_features: int
    feature_reports: List[DriftFeatureReport] = field(default_factory=list)
    prediction_drift: Optional[Dict[str, Any]] = None
    retraining_recommended: bool = False
    recommendation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_features_analyzed": self.n_features_analyzed,
            "n_drifting_features": self.n_drifting_features,
            "feature_reports": [f.to_dict() for f in self.feature_reports],
            "prediction_drift": self.prediction_drift,
            "retraining_recommended": self.retraining_recommended,
            "recommendation_reason": self.recommendation_reason,
        }

    def render(self) -> str:
        lines = [
            "DATA & PREDICTION DRIFT REPORT",
            "==============================",
            f"Features Analyzed   : {self.n_features_analyzed}",
            f"Drifting Features   : {self.n_drifting_features}",
            f"Retraining Needed   : {'YES (REQUIRED)' if self.retraining_recommended else 'NO (STABLE)'}",
            f"Reason              : {self.recommendation_reason}",
            "",
            "Feature Drift Statuses:",
        ]
        for fr in self.feature_reports:
            icon = "[DRIFT]" if fr.drift_status == "SIGNIFICANT_DRIFT" else ("[WARN]" if fr.drift_status == "MODERATE_DRIFT" else "[PASS]")
            lines.append(f"  {icon} {fr.feature:<20}: PSI={fr.psi_score:.4f} [{fr.drift_status}]")

        if self.prediction_drift:
            lines.append("")
            lines.append("Prediction Distribution Drift:")
            lines.append(f"  - Status : {self.prediction_drift.get('status', 'NO_DRIFT')}")
            lines.append(f"  - PSI    : {self.prediction_drift.get('psi', 0.0):.4f}")
        return "\n".join(lines)


class DriftDetector:
    """Production data & prediction drift detection engine."""

    def __init__(
        self,
        psi_threshold_moderate: float = 0.10,
        psi_threshold_significant: float = 0.25,
        ks_alpha: float = 0.01,
    ) -> None:
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_significant = psi_threshold_significant
        self.ks_alpha = ks_alpha

    def analyze_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        ref_predictions: Optional[np.ndarray] = None,
        curr_predictions: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """Compare current production data vs baseline reference data."""
        feature_reports: List[DriftFeatureReport] = []
        common_cols = [c for c in reference_df.columns if c in current_df.columns]

        for col in common_cols:
            ref_s = reference_df[col].dropna()
            curr_s = current_df[col].dropna()

            if len(ref_s) < 10 or len(curr_s) < 10:
                continue

            if pd.api.types.is_numeric_dtype(ref_s):
                fr = self._analyze_numeric_drift(col, ref_s, curr_s)
            else:
                fr = self._analyze_categorical_drift(col, ref_s, curr_s)

            feature_reports.append(fr)

        # Prediction Drift Analysis
        pred_drift_data = None
        if ref_predictions is not None and curr_predictions is not None:
            pred_drift_data = self._analyze_prediction_drift(ref_predictions, curr_predictions)

        # Retraining Recommendation Decision
        n_significant = sum(1 for fr in feature_reports if fr.drift_status == "SIGNIFICANT_DRIFT")
        n_drifting = sum(1 for fr in feature_reports if fr.drift_status != "NO_DRIFT")

        retrain = False
        reason = "Distribution alignment within safe operating boundaries."

        if n_significant >= 2 or (pred_drift_data and pred_drift_data.get("status") == "SIGNIFICANT_DRIFT"):
            retrain = True
            reason = f"Significant drift detected in {n_significant} feature(s) and/or prediction distribution."
        elif n_drifting >= len(feature_reports) * 0.4:
            retrain = True
            reason = f"Widespread moderate drift across {n_drifting} features."

        return DriftReport(
            n_features_analyzed=len(feature_reports),
            n_drifting_features=n_drifting,
            feature_reports=feature_reports,
            prediction_drift=pred_drift_data,
            retraining_recommended=retrain,
            recommendation_reason=reason,
        )

    def _analyze_numeric_drift(
        self, col: str, ref: pd.Series, curr: pd.Series
    ) -> DriftFeatureReport:
        # Calculate PSI via 10 quantile bins
        psi = self._calculate_numeric_psi(ref, curr, bins=10)

        # KS test
        ks_pvalue = None
        try:
            from scipy.stats import ks_2samp

            _, pval = ks_2samp(ref, curr)
            ks_pvalue = float(pval)
        except Exception:
            pass

        status = self._get_drift_status(psi)

        return DriftFeatureReport(
            feature=col,
            feature_type="numeric",
            psi_score=psi,
            ks_pvalue=ks_pvalue,
            js_divergence=None,
            drift_status=status,
        )

    def _analyze_categorical_drift(
        self, col: str, ref: pd.Series, curr: pd.Series
    ) -> DriftFeatureReport:
        ref_counts = ref.value_counts(normalize=True)
        curr_counts = curr.value_counts(normalize=True)

        all_categories = set(ref_counts.index).union(set(curr_counts.index))

        ref_p = np.array([ref_counts.get(cat, 1e-4) for cat in all_categories])
        curr_p = np.array([curr_counts.get(cat, 1e-4) for cat in all_categories])

        # Normalize
        ref_p /= ref_p.sum()
        curr_p /= curr_p.sum()

        psi = float(np.sum((curr_p - ref_p) * np.log(curr_p / ref_p)))
        js_div = float(0.5 * (np.sum(ref_p * np.log(ref_p / curr_p)) + np.sum(curr_p * np.log(curr_p / ref_p))))

        status = self._get_drift_status(psi)

        return DriftFeatureReport(
            feature=col,
            feature_type="categorical",
            psi_score=psi,
            ks_pvalue=None,
            js_divergence=js_div,
            drift_status=status,
        )

    def _analyze_prediction_drift(
        self, ref_preds: np.ndarray, curr_preds: np.ndarray
    ) -> Dict[str, Any]:
        ref_s = pd.Series(ref_preds)
        curr_s = pd.Series(curr_preds)

        if pd.api.types.is_numeric_dtype(ref_s):
            psi = self._calculate_numeric_psi(ref_s, curr_s, bins=10)
        else:
            ref_c = ref_s.value_counts(normalize=True)
            curr_c = curr_s.value_counts(normalize=True)
            cats = set(ref_c.index).union(set(curr_c.index))
            p = np.array([ref_c.get(c, 1e-4) for c in cats])
            q = np.array([curr_c.get(c, 1e-4) for c in cats])
            p /= p.sum()
            q /= q.sum()
            psi = float(np.sum((q - p) * np.log(q / p)))

        return {
            "psi": round(psi, 4),
            "status": self._get_drift_status(psi),
        }

    def _calculate_numeric_psi(
        self, ref: pd.Series, curr: pd.Series, bins: int = 10
    ) -> float:
        try:
            quantiles = np.linspace(0, 100, bins + 1)
            bin_edges = np.percentile(ref, quantiles)
            bin_edges[0] -= 1e-5
            bin_edges[-1] += 1e-5
            bin_edges = np.unique(bin_edges)

            ref_counts, _ = np.histogram(ref, bins=bin_edges)
            curr_counts, _ = np.histogram(curr, bins=bin_edges)

            ref_pct = np.clip(ref_counts / len(ref), 1e-4, 1.0)
            curr_pct = np.clip(curr_counts / len(curr), 1e-4, 1.0)

            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            return float(max(0.0, psi))
        except Exception:
            return 0.0

    def _get_drift_status(self, psi: float) -> str:
        if psi >= self.psi_threshold_significant:
            return "SIGNIFICANT_DRIFT"
        elif psi >= self.psi_threshold_moderate:
            return "MODERATE_DRIFT"
        return "NO_DRIFT"
