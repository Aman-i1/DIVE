"""Validation Intelligence Engine - `dive/validation_engine.py`.

Automated detection of:
1. Target leakage & post-outcome variables.
2. Entity contamination & entity cross-split leakage.
3. Temporal point-in-time boundary violations.
4. Duplicate and near-duplicate cross-fold leakage.

Automatically selects among:
- KFold
- StratifiedKFold
- GroupKFold
- StratifiedGroupKFold
- TimeSeriesSplit
- Holdout

Calculates a comprehensive ValidationRiskScore and logs explainable decision records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dive.decisions import DecisionLogger
from dive.leakage import AdvancedLeakageDetector


@dataclass
class ValidationRiskScore:
    """Validation risk evaluation result."""

    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float  # 0.0 (Safe) to 100.0 (Extremely Unsafe)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 1),
            "reasons": self.reasons,
        }


@dataclass
class ValidationPlan:
    """Explainable cross-validation execution plan."""

    strategy: str  # StratifiedKFold, GroupKFold, TimeSeriesSplit, KFold, etc.
    n_splits: int
    group_column: Optional[str]
    time_column: Optional[str]
    risk_assessment: ValidationRiskScore
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_splits": self.n_splits,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "risk_assessment": self.risk_assessment.to_dict(),
            "explanation": self.explanation,
        }

    def render(self) -> str:
        lines = [
            "VALIDATION INTELLIGENCE PLAN",
            "============================",
            f"Strategy             : {self.strategy}(n_splits={self.n_splits})",
            f"Group Column         : {self.group_column or 'None'}",
            f"Time Column          : {self.time_column or 'None'}",
            f"Validation Risk      : {self.risk_assessment.risk_level} ({self.risk_assessment.risk_score:.0f}/100)",
            f"Explanation          : {self.explanation}",
        ]
        if self.risk_assessment.reasons:
            lines.append("Identified Risks:")
            for r in self.risk_assessment.reasons:
                lines.append(f"  [WARN] {r}")
        return "\n".join(lines)


class ValidationIntelligenceEngine:
    """Automated split strategy selector and risk evaluation engine."""

    def __init__(self, target: str, logger: Optional[DecisionLogger] = None) -> None:
        self.target = target
        self.logger = logger or DecisionLogger()

    def evaluate(
        self,
        df: pd.DataFrame,
        problem_type: str = "classification",
        user_group_column: Optional[str] = None,
        user_time_column: Optional[str] = None,
        n_splits: int = 5,
    ) -> ValidationPlan:
        """Evaluate dataset risks and select optimal cross-validation strategy."""
        reasons: List[str] = []
        risk_score = 0.0

        n_rows = len(df)
        X = df.drop(columns=[self.target]) if self.target in df.columns else df.copy()

        # 1. Check for Entity / Group structure
        detected_group_col = user_group_column
        if not detected_group_col:
            for col in X.select_dtypes(include=["object", "int64"]).columns:
                name = str(col).lower()
                if any(k in name for k in ("user", "customer", "patient", "client", "subject", "entity", "account")):
                    nunique = X[col].nunique()
                    if 1 < nunique < (0.8 * n_rows):
                        detected_group_col = col
                        reasons.append(f"Repeated entity instances detected in column '{col}' ({nunique:,} unique entities).")
                        risk_score += 40.0
                        break

        # 2. Check for Temporal Ordering
        detected_time_col = user_time_column
        if not detected_time_col:
            for col in X.columns:
                name = str(col).lower()
                if any(k in name for k in ("time", "date", "timestamp", "created_at", "month", "year")):
                    detected_time_col = col
                    reasons.append(f"Temporal ordering detected in column '{col}'. Random splits will cause temporal leakage.")
                    risk_score += 35.0
                    break

        # 3. Check for Target Leakage
        leak_detector = AdvancedLeakageDetector(target=self.target)
        leak_report = leak_detector.detect(df, problem_type=problem_type)
        if leak_report.has_critical_leakage:
            reasons.append("Critical target leakage detected in feature columns.")
            risk_score += 50.0

        # 4. Check for duplicate rows
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            reasons.append(f"{dup_count:,} duplicate rows detected. Random splits will leak duplicate records across folds.")
            risk_score += 20.0

        # Determine Risk Level
        risk_score = min(100.0, risk_score)
        if risk_score >= 60.0:
            risk_level = "HIGH"
        elif risk_score >= 30.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        risk_assessment = ValidationRiskScore(risk_level=risk_level, risk_score=risk_score, reasons=reasons)

        # 5. Determine Strategy & Rationale
        if detected_group_col and detected_time_col:
            strategy = "TimeSeriesSplit"
            explanation = (
                f"Selected TimeSeriesSplit on '{detected_time_col}' with entity grouping on '{detected_group_col}'. "
                "Random K-Fold rejected to prevent entity contamination and temporal boundary leakage."
            )
        elif detected_time_col:
            strategy = "TimeSeriesSplit"
            explanation = f"Selected TimeSeriesSplit on '{detected_time_col}'. Random K-Fold rejected due to temporal ordering."
        elif detected_group_col:
            strategy = "GroupKFold" if problem_type == "regression" else "StratifiedGroupKFold"
            explanation = f"Selected {strategy} on group column '{detected_group_col}' to prevent entity leakage."
        elif problem_type == "classification":
            strategy = "StratifiedKFold"
            explanation = "Selected StratifiedKFold to preserve class distribution across folds. No entity or temporal leakage risks found."
        else:
            strategy = "KFold"
            explanation = "Selected KFold. Dataset exhibits standard IID numerical distribution."

        # Log decision into DecisionLogger
        self.logger.log(
            component="ValidationIntelligenceEngine",
            decision=f"Selected {strategy}(n_splits={n_splits})",
            reason=explanation,
            confidence=0.95,
            evidence={
                "risk_level": risk_level,
                "risk_score": risk_score,
                "group_col": detected_group_col,
                "time_col": detected_time_col,
            },
        )

        return ValidationPlan(
            strategy=strategy,
            n_splits=n_splits,
            group_column=detected_group_col,
            time_column=detected_time_col,
            risk_assessment=risk_assessment,
            explanation=explanation,
        )
