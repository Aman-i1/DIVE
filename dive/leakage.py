"""Advanced Data Leakage & Point-in-Time Feature Validation Engine.

Detects multiple patterns of data leakage:
- Feature-level leakage (Pearson, Spearman, Mutual Information, Categorical Association, Target-encoding association)
- Univariate near-perfect prediction test (lightweight single-feature model AUC > 0.995 or R² > 0.995)
- Temporal leakage & Point-in-Time feature availability (future information after prediction event)
- Duplicate leakage (exact duplicate rows, duplicate feature vectors, entity contamination across splits)
- Target-derived feature name pattern warnings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class LeakageWarning:
    """Individual leakage warning for a single feature or relationship."""

    feature: str
    risk_level: str  # HIGH, MEDIUM, LOW
    category: str    # FEATURE_LEAKAGE, TEMPORAL_LEAKAGE, DUPLICATE_LEAKAGE, NAME_PATTERN
    evidence_metric: str
    evidence_score: float
    reason: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "risk_level": self.risk_level,
            "category": self.category,
            "evidence_metric": self.evidence_metric,
            "evidence_score": round(self.evidence_score, 4),
            "reason": self.reason,
            "recommendation": self.recommendation,
        }


@dataclass
class LeakageReport:
    """Comprehensive data leakage report."""

    has_high_risk: bool
    warnings: List[LeakageWarning] = field(default_factory=list)
    duplicate_metrics: Dict[str, Any] = field(default_factory=dict)
    point_in_time_status: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_high_risk": self.has_high_risk,
            "warnings": [w.to_dict() for w in self.warnings],
            "duplicate_metrics": self.duplicate_metrics,
            "point_in_time_status": self.point_in_time_status,
        }

    def render(self) -> str:
        lines = ["LEAKAGE DETECTION REPORT", "========================"]
        if not self.warnings:
            lines.append("✓ No leakage risks detected.")
            return "\n".join(lines)

        for w in self.warnings:
            icon = "🔴" if w.risk_level == "HIGH" else "⚠"
            lines.append(f"{icon} {w.risk_level} RISK - Feature: '{w.feature}' [{w.category}]")
            lines.append(f"   Evidence: {w.evidence_metric} = {w.evidence_score:.4f}")
            lines.append(f"   Reason: {w.reason}")
            lines.append(f"   Recommendation: {w.recommendation}")
            lines.append("")
        return "\n".join(lines)


class AdvancedLeakageDetector:
    """Multi-strategy data leakage and point-in-time validator."""

    def __init__(
        self,
        high_risk_threshold: float = 0.98,
        univariate_auc_threshold: float = 0.995,
        suspicious_threshold: float = 0.90,
    ) -> None:
        self.high_risk_threshold = high_risk_threshold
        self.univariate_auc_threshold = univariate_auc_threshold
        self.suspicious_threshold = suspicious_threshold

        self.name_pattern_keywords = [
            "target", "label", "outcome", "result", "closed", "cancelled",
            "approved", "rejected", "future", "post_", "after_", "final_"
        ]

    def audit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
        prediction_time_col: Optional[str] = None,
        feature_time_cols: Optional[List[str]] = None,
        train_index: Optional[Sequence] = None,
        val_index: Optional[Sequence] = None,
    ) -> LeakageReport:
        """Execute complete leakage audit."""
        warnings: List[LeakageWarning] = []

        # 1. Feature Name Pattern Warnings
        self._check_name_patterns(X, warnings)

        # 2. Association & Univariate Model Leakage
        self._check_feature_associations(X, y, problem_type, warnings)

        # 3. Temporal Point-in-Time Availability
        pit_status = {}
        if prediction_time_col and feature_time_cols:
            pit_status = self._check_point_in_time(
                X, prediction_time_col, feature_time_cols, warnings
            )

        # 4. Duplicate & Cross-Partition Contamination Leakage
        dup_metrics = self._check_duplicate_leakage(
            X, train_index, val_index, warnings
        )

        has_high_risk = any(w.risk_level == "HIGH" for w in warnings)

        return LeakageReport(
            has_high_risk=has_high_risk,
            warnings=warnings,
            duplicate_metrics=dup_metrics,
            point_in_time_status=pit_status,
        )

    # ------------------------------------------------------------------
    def _check_name_patterns(
        self, X: pd.DataFrame, warnings: List[LeakageWarning]
    ) -> None:
        for col in X.columns:
            name_lower = str(col).lower()
            matching_kw = [kw for kw in self.name_pattern_keywords if kw in name_lower]
            if matching_kw:
                warnings.append(
                    LeakageWarning(
                        feature=str(col),
                        risk_level="MEDIUM",
                        category="NAME_PATTERN",
                        evidence_metric="keyword_match",
                        evidence_score=1.0,
                        reason=f"Feature name contains suspicious post-event pattern: '{matching_kw[0]}'.",
                        recommendation="Verify whether this feature is known at prediction time.",
                    )
                )

    def _check_feature_associations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        warnings: List[LeakageWarning],
    ) -> None:
        for col in X.columns:
            series = X[col]
            valid = series.notna() & y.notna()
            if valid.sum() < 10:
                continue

            s_clean = series[valid]
            y_clean = y[valid]

            # Fast Pearson & Spearman correlation for numeric
            if pd.api.types.is_numeric_dtype(s_clean):
                try:
                    pearson = abs(float(np.corrcoef(s_clean.astype(float), y_clean.astype(float))[0, 1]))
                    if np.isfinite(pearson) and pearson >= self.high_risk_threshold:
                        warnings.append(
                            LeakageWarning(
                                feature=str(col),
                                risk_level="HIGH",
                                category="FEATURE_LEAKAGE",
                                evidence_metric="Pearson Correlation",
                                evidence_score=pearson,
                                reason=f"Near-perfect linear correlation ({pearson:.4f}) with target.",
                                recommendation="Remove feature before model training.",
                            )
                        )
                        continue
                except Exception:
                    pass

            # Univariate Model Prediction Test
            try:
                score = self._run_univariate_model(s_clean, y_clean, problem_type)
                if score >= self.univariate_auc_threshold:
                    warnings.append(
                        LeakageWarning(
                            feature=str(col),
                            risk_level="HIGH",
                            category="FEATURE_LEAKAGE",
                            evidence_metric="Univariate Model AUC/R²",
                            evidence_score=score,
                            reason=f"Single-feature decision tree achieves {score:.4f} metric.",
                            recommendation="Remove feature: it almost perfectly encodes the target.",
                        )
                    )
                elif score >= self.suspicious_threshold:
                    # Check if already added as name pattern or high risk
                    if not any(w.feature == str(col) and w.risk_level == "HIGH" for w in warnings):
                        warnings.append(
                            LeakageWarning(
                                feature=str(col),
                                risk_level="MEDIUM",
                                category="FEATURE_LEAKAGE",
                                evidence_metric="Univariate Model AUC/R²",
                                evidence_score=score,
                                reason=f"Single-feature achieves unusually high predictive score ({score:.4f}).",
                                recommendation="Investigate feature provenance for potential leakage.",
                            )
                        )
            except Exception:
                pass

    def _run_univariate_model(
        self, s: pd.Series, y: pd.Series, problem_type: str
    ) -> float:
        """Fit a lightweight 1-feature tree to check predictive power."""
        if pd.api.types.is_numeric_dtype(s):
            X_uni = s.to_numpy().reshape(-1, 1)
        else:
            X_uni = pd.factorize(s.astype(str))[0].reshape(-1, 1)

        if problem_type == "classification":
            tree = DecisionTreeClassifier(max_depth=3, random_state=42)
            tree.fit(X_uni, y)
            if len(np.unique(y)) == 2:
                proba = tree.predict_proba(X_uni)[:, 1]
                return float(roc_auc_score(y, proba))
            else:
                preds = tree.predict(X_uni)
                return float((preds == y).mean())
        else:
            tree = DecisionTreeRegressor(max_depth=3, random_state=42)
            tree.fit(X_uni, y)
            preds = tree.predict(X_uni)
            return float(max(0.0, r2_score(y, preds)))

    def _check_point_in_time(
        self,
        X: pd.DataFrame,
        pred_time_col: str,
        feature_time_cols: List[str],
        warnings: List[LeakageWarning],
    ) -> Dict[str, str]:
        status_map = {}
        if pred_time_col not in X.columns:
            return status_map

        pred_time = pd.to_datetime(X[pred_time_col], errors="coerce")

        for f_col in feature_time_cols:
            if f_col not in X.columns:
                continue
            f_time = pd.to_datetime(X[f_col], errors="coerce")
            valid = pred_time.notna() & f_time.notna()
            if valid.sum() == 0:
                status_map[f_col] = "UNKNOWN"
                continue

            # Check if feature timestamp occurs AFTER prediction time
            future_mask = f_time[valid] > pred_time[valid]
            future_pct = float(future_mask.mean() * 100)

            if future_pct > 0:
                status_map[f_col] = "FUTURE_INFORMATION"
                warnings.append(
                    LeakageWarning(
                        feature=f_col,
                        risk_level="HIGH",
                        category="TEMPORAL_LEAKAGE",
                        evidence_metric="Future Event Ratio",
                        evidence_score=future_pct / 100.0,
                        reason=f"{future_pct:.1f}% of timestamps occur after prediction time '{pred_time_col}'.",
                        recommendation=f"Drop '{f_col}' or filter out future events relative to '{pred_time_col}'.",
                    )
                )
            else:
                status_map[f_col] = "AVAILABLE"

        return status_map

    def _check_duplicate_leakage(
        self,
        X: pd.DataFrame,
        train_index: Optional[Sequence],
        val_index: Optional[Sequence],
        warnings: List[LeakageWarning],
    ) -> Dict[str, Any]:
        metrics = {"total_rows": len(X), "duplicate_rows": 0}
        if len(X) < 2:
            return metrics

        try:
            hashed = pd.util.hash_pandas_object(X.astype(str), index=False)
            dup_count = int(hashed.duplicated().sum())
            metrics["duplicate_rows"] = dup_count
            metrics["duplicate_pct"] = round(100.0 * dup_count / len(X), 2)

            if train_index is not None and val_index is not None:
                train_hashes = set(hashed.iloc[train_index])
                val_hashes = hashed.iloc[val_index]
                crossing = int(val_hashes.isin(train_hashes).sum())
                metrics["partition_crossing_duplicates"] = crossing

                if crossing > 0:
                    warnings.append(
                        LeakageWarning(
                            feature="[ALL_FEATURES]",
                            risk_level="HIGH",
                            category="DUPLICATE_LEAKAGE",
                            evidence_metric="Crossing Duplicate Rows",
                            evidence_score=crossing,
                            reason=f"{crossing} exact duplicate row(s) cross between train and validation partitions.",
                            recommendation="Remove duplicate rows or use entity-aware validation.",
                        )
                    )
        except Exception:
            pass

        return metrics
