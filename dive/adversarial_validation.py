"""Adversarial Validation & Distribution Shift Discriminator - `dive/adversarial_validation.py`.

Trains a binary classifier to distinguish between training data (label 0) and validation/production data (label 1).
If the classifier achieves high ROC-AUC (e.g. > 0.70), train and target distributions are significantly shifted.
Identifies the top contributing distribution-shift features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dive.decisions import DecisionLogger


@dataclass
class AdversarialValidationReport:
    """Findings from adversarial validation distribution shift audit."""

    adversarial_auc: float  # ~0.50 means identical distributions, >0.70 means strong covariate shift
    shift_status: str  # 'SAFE_IID', 'MODERATE_SHIFT', 'SEVERE_COVARIATE_SHIFT'
    top_drift_features: List[Tuple[str, float]]  # (feature_name, importance_score)
    interpretation: str
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adversarial_auc": round(self.adversarial_auc, 4),
            "shift_status": self.shift_status,
            "top_drift_features": [[feat, round(float(imp), 4)] for feat, imp in self.top_drift_features],
            "interpretation": self.interpretation,
            "recommendations": self.recommendations,
        }

    def render(self) -> str:
        lines = [
            "ADVERSARIAL VALIDATION DISTRIBUTION SHIFT AUDIT",
            "===============================================",
            f"Adversarial ROC-AUC      : {self.adversarial_auc:.4f} [Status: {self.shift_status}]",
            f"Interpretation           : {self.interpretation}",
        ]
        if self.top_drift_features:
            lines.append("\nTop Features Driving Distribution Shift:")
            for feat, imp in self.top_drift_features[:5]:
                lines.append(f"  - {feat:<20}: Importance = {imp:.4f}")
        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


class AdversarialValidator:
    """Evaluates covariate shift between training and evaluation/production partitions."""

    def __init__(self, logger: Optional[DecisionLogger] = None) -> None:
        self.logger = logger or DecisionLogger()

    def evaluate_shift(
        self,
        train_df: pd.DataFrame,
        target_df: pd.DataFrame,
        target_column: Optional[str] = None,
        max_samples: int = 5000,
    ) -> AdversarialValidationReport:
        """Fit an adversarial discriminator and evaluate ROC-AUC."""
        X_train = train_df.drop(columns=[target_column]) if target_column and target_column in train_df.columns else train_df
        X_target = target_df.drop(columns=[target_column]) if target_column and target_column in target_df.columns else target_df

        common_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns if c in X_target.columns]
        if not common_cols:
            return AdversarialValidationReport(
                adversarial_auc=0.50,
                shift_status="SAFE_IID",
                top_drift_features=[],
                interpretation="No common numeric features available for adversarial validation.",
                recommendations=["Ensure numeric features are present in both train and target sets."],
            )

        # Sample for bounded latency
        if len(X_train) > max_samples:
            X_train = X_train.sample(max_samples, random_state=42)
        if len(X_target) > max_samples:
            X_target = X_target.sample(max_samples, random_state=42)

        X_tr = X_train[common_cols].fillna(0.0)
        X_tg = X_target[common_cols].fillna(0.0)

        # Create discriminator dataset: 0 = Train, 1 = Target
        X_adv = pd.concat([X_tr, X_tg], ignore_index=True)
        y_adv = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_tg))])

        # 3-Fold Cross-Validation Discriminator
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y_adv))
        feature_importances = np.zeros(len(common_cols))

        for train_idx, val_idx in cv.split(X_adv, y_adv):
            clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(X_adv.iloc[train_idx], y_adv[train_idx])
            oof_preds[val_idx] = clf.predict_proba(X_adv.iloc[val_idx])[:, 1]
            feature_importances += clf.feature_importances_ / 3.0

        adv_auc = float(roc_auc_score(y_adv, oof_preds))

        # Sort feature importances
        sorted_idx = np.argsort(feature_importances)[::-1]
        top_features = [(common_cols[i], float(feature_importances[i])) for i in sorted_idx[:5]]

        recs: List[str] = []
        if adv_auc >= 0.80:
            status = "SEVERE_COVARIATE_SHIFT"
            interpretation = "Training data and Target Population are strongly distinguishable (Severe Covariate Shift)."
            recs.append(f"Top shifting features: {', '.join([f[0] for f in top_features[:3]])}. Consider dropping or adversarial weight adjustment.")
            recs.append("Random CV will overestimate test performance. Rely on realistic temporal/out-of-time splits.")
        elif adv_auc >= 0.65:
            status = "MODERATE_SHIFT"
            interpretation = "Moderate distribution movement detected between datasets."
            recs.append("Monitor top drifting features and evaluate model stability under subgroup splits.")
        else:
            status = "SAFE_IID"
            interpretation = "Training and Target populations appear indistinguishable (IID distribution match)."
            recs.append("Distributions are well aligned.")

        self.logger.log(
            component="AdversarialValidator",
            decision=f"Adversarial ROC-AUC: {adv_auc:.4f} [Status: {status}]",
            reason=interpretation,
            confidence=0.95,
            evidence={
                "adversarial_auc": round(adv_auc, 4),
                "status": status,
                "top_features": top_features,
            },
        )

        return AdversarialValidationReport(
            adversarial_auc=adv_auc,
            shift_status=status,
            top_drift_features=top_features,
            interpretation=interpretation,
            recommendations=recs,
        )
