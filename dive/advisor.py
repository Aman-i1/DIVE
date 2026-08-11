"""Validation & Model Advisory System.

Includes:
- ValidationAdvisor: Automatic cross-validation strategy selection & group leakage prevention.
- ModelAdvisor: Explainable model recommendation engine based on dataset characteristics & hardware budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


@dataclass
class ValidationAdvice:
    """Output of ValidationAdvisor.advise()."""

    recommended_strategy: str
    strategy_name: str
    is_random_safe: bool
    group_column: Optional[str]
    time_column: Optional[str]
    reason: str
    group_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_strategy": self.recommended_strategy,
            "strategy_name": self.strategy_name,
            "is_random_safe": self.is_random_safe,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "reason": self.reason,
            "group_stats": self.group_stats,
        }

    def render(self) -> str:
        lines = [
            "VALIDATION ADVISOR RECOMMENDATION",
            "=================================",
            f"Recommended Strategy : {self.recommended_strategy}",
            f"Random Split Safety  : {'✓ SAFE' if self.is_random_safe else '🔴 UNSAFE'}",
            f"Explanation          : {self.reason}",
        ]
        if self.group_stats:
            lines.append("Group Structure:")
            lines.append(f"  - Unique Groups    : {self.group_stats.get('n_groups', 0):,}")
            lines.append(f"  - Rows per Group   : {self.group_stats.get('mean_rows_per_group', 0):.1f} avg (max: {self.group_stats.get('max_rows_per_group', 0)})")
            if self.group_stats.get("crossing_groups_pct", 0) > 0:
                lines.append(f"  - Entity Contamination: ⚠ {self.group_stats.get('crossing_groups_pct', 0):.1f}% of groups cross random splits")
        return "\n".join(lines)


class ValidationAdvisor:
    """Selects the optimal CV strategy and flags entity/temporal contamination."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.group_keywords = [
            "customer_id", "user_id", "patient_id", "account_id", "device_id",
            "session_id", "transaction_group", "client_id", "entity_id", "subject_id"
        ]

    def advise(
        self,
        df: pd.DataFrame,
        target: str,
        problem_type: str = "classification",
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
        n_folds: int = 5,
    ) -> ValidationAdvice:
        """Analyze dataset and return optimal validation advice."""
        X = df.drop(columns=[target]) if target in df.columns else df
        y = df[target] if target in df.columns else None

        # Auto-detect group or time columns if not explicitly provided
        detected_group = group_column or self.detect_group_column(X)
        detected_time = time_column or self.detect_time_column(X)

        group_stats = {}
        if detected_group and detected_group in X.columns:
            group_stats = self._analyze_group_structure(X, detected_group, n_folds)

        is_imbalanced = False
        if problem_type == "classification" and y is not None:
            counts = y.value_counts()
            if len(counts) > 1 and (counts.max() / counts.min()) > 3.0:
                is_imbalanced = True

        # Decision Matrix
        if detected_time:
            return ValidationAdvice(
                recommended_strategy=f"TimeSeriesSplit(n_splits={n_folds})",
                strategy_name="TimeSeriesSplit",
                is_random_safe=False,
                group_column=detected_group,
                time_column=detected_time,
                reason=f"Random validation is unsafe. Datetime column '{detected_time}' detected. Future data leakage occurs with random splits.",
                group_stats=group_stats,
            )

        if detected_group:
            crossing_pct = group_stats.get("crossing_groups_pct", 0)
            if is_imbalanced and problem_type == "classification":
                strat = f"StratifiedGroupKFold(n_splits={n_folds}, group='{detected_group}')"
                strat_name = "StratifiedGroupKFold"
            else:
                strat = f"GroupKFold(n_splits={n_folds}, group='{detected_group}')"
                strat_name = "GroupKFold"

            return ValidationAdvice(
                recommended_strategy=strat,
                strategy_name=strat_name,
                is_random_safe=False,
                group_column=detected_group,
                time_column=None,
                reason=f"Random validation is unsafe. {crossing_pct:.1f}% of unique entity groups in '{detected_group}' cross partitions under random splitting.",
                group_stats=group_stats,
            )

        if is_imbalanced and problem_type == "classification":
            return ValidationAdvice(
                recommended_strategy=f"StratifiedKFold(n_splits={n_folds}, shuffle=True)",
                strategy_name="StratifiedKFold",
                is_random_safe=True,
                group_column=None,
                time_column=None,
                reason="Target is imbalanced. StratifiedKFold recommended to maintain class ratios across all folds.",
                group_stats=group_stats,
            )

        return ValidationAdvice(
            recommended_strategy=f"KFold(n_splits={n_folds}, shuffle=True)",
            strategy_name="KFold",
            is_random_safe=True,
            group_column=None,
            time_column=None,
            reason="Dataset is IID without group or temporal dependencies. Standard random KFold CV is safe.",
            group_stats=group_stats,
        )

    def detect_group_column(self, X: pd.DataFrame) -> Optional[str]:
        """Detect entity ID columns with repeated rows per entity."""
        for col in X.columns:
            name_lower = str(col).lower()
            if any(kw in name_lower for kw in self.group_keywords):
                n_rows = len(X)
                n_unique = X[col].nunique(dropna=True)
                if 1 < n_unique < 0.95 * n_rows:
                    return str(col)
        return None

    def detect_time_column(self, X: pd.DataFrame) -> Optional[str]:
        """Detect timestamp or date columns."""
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                return str(col)
            name_lower = str(col).lower()
            if any(kw in name_lower for kw in ["timestamp", "date", "event_time"]):
                return str(col)
        return None

    def _analyze_group_structure(
        self, X: pd.DataFrame, group_col: str, n_folds: int
    ) -> Dict[str, Any]:
        series = X[group_col].dropna()
        n_unique = series.nunique()
        counts = series.value_counts()

        # Simulate random KFold split to calculate entity crossing percentage
        np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(len(series))
        split_point = int(len(series) * 0.8)
        train_groups = set(series.iloc[shuffled_indices[:split_point]])
        val_groups = set(series.iloc[shuffled_indices[split_point:]])
        crossing_groups = train_groups.intersection(val_groups)
        crossing_pct = (len(crossing_groups) / max(1, n_unique)) * 100.0

        return {
            "group_column": group_col,
            "n_groups": int(n_unique),
            "mean_rows_per_group": float(counts.mean()),
            "max_rows_per_group": int(counts.max()),
            "min_rows_per_group": int(counts.min()),
            "crossing_groups_pct": float(crossing_pct),
        }


@dataclass
class ModelAdvice:
    """Output of ModelAdvisor.advise()."""

    recommended: List[str]
    acceptable: List[str]
    deprioritized: List[str]
    rejected: List[str]
    decisions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended": self.recommended,
            "acceptable": self.acceptable,
            "deprioritized": self.deprioritized,
            "rejected": self.rejected,
            "decisions": self.decisions,
        }

    def render(self) -> str:
        lines = ["MODEL ADVISOR RECOMMENDATIONS", "============================="]
        lines.append(f"⭐ Recommended  : {', '.join(self.recommended) if self.recommended else 'None'}")
        lines.append(f"✓ Acceptable   : {', '.join(self.acceptable) if self.acceptable else 'None'}")
        lines.append(f"⚠ Deprioritized: {', '.join(self.deprioritized) if self.deprioritized else 'None'}")
        lines.append(f"✗ Rejected     : {', '.join(self.rejected) if self.rejected else 'None'}")
        lines.append("")
        lines.append("Decision Rationales:")
        for model, reason in self.decisions.items():
            lines.append(f"  - {model:<18}: {reason}")
        return "\n".join(lines)


class ModelAdvisor:
    """Evaluates dataset characteristics & hardware budget to select models."""

    def advise(
        self,
        n_samples: int,
        n_features: int,
        n_categorical: int,
        has_high_cardinality: bool,
        has_missing: bool,
        available_ram_gb: float = 8.0,
        has_gpu: bool = False,
    ) -> ModelAdvice:
        recommended = []
        acceptable = []
        deprioritized = []
        rejected = []
        decisions = {}

        # 1. CatBoost
        if has_high_cardinality or n_categorical > 5:
            recommended.append("CatBoost")
            decisions["CatBoost"] = "Recommended: Superior native handling of high-cardinality categorical features."
        else:
            recommended.append("CatBoost")
            decisions["CatBoost"] = "Recommended: Robust gradient boosting algorithm."

        # 2. LightGBM & XGBoost
        recommended.append("LightGBM")
        decisions["LightGBM"] = "Recommended: Fast histogram-based tree boosting with high memory efficiency."
        recommended.append("XGBoost")
        decisions["XGBoost"] = "Recommended: High performance gradient boosting framework."

        # 3. RandomForest & HistGradientBoosting
        acceptable.append("RandomForest")
        decisions["RandomForest"] = "Acceptable: Strong ensemble baseline."
        acceptable.append("HistGradientBoosting")
        decisions["HistGradientBoosting"] = "Acceptable: Fast native scikit-learn tree boosting."

        # 4. KNN
        if n_samples > 50_000:
            rejected.append("KNN")
            decisions["KNN"] = f"Rejected: Dataset size ({n_samples:,} rows) exceeds O(N^2) distance computation limit."
        else:
            deprioritized.append("KNN")
            decisions["KNN"] = "Deprioritized: Sensitive to feature scaling and high dimensions."

        # 5. MLP
        if n_samples > 200_000 or available_ram_gb < 4.0:
            deprioritized.append("MLP")
            decisions["MLP"] = f"Deprioritized: High memory/CPU training complexity for {n_samples:,} rows."
        else:
            acceptable.append("MLP")
            decisions["MLP"] = "Acceptable: Multi-layer perceptron neural network."

        return ModelAdvice(
            recommended=recommended,
            acceptable=acceptable,
            deprioritized=deprioritized,
            rejected=rejected,
            decisions=decisions,
        )
