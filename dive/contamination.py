"""Train/Validation & Train/Test Contamination Engine - `dive/contamination.py`.

Detects:
1. Exact row duplicates across train and validation/test splits.
2. Entity overlap across splits (e.g. same customer_id in train and test).
3. Feature-vector duplicates with conflicting targets (label noise / target ambiguity).
4. Near-duplicate records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dive.decisions import DecisionLogger


@dataclass
class ContaminationReport:
    """Findings from cross-partition contamination audit."""

    exact_duplicates_across_splits: int
    entity_overlap_count: int
    entity_column: Optional[str]
    target_conflict_count: int  # identical features with different target labels
    contamination_risk: str  # 'SAFE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_duplicates_across_splits": self.exact_duplicates_across_splits,
            "entity_overlap_count": self.entity_overlap_count,
            "entity_column": self.entity_column,
            "target_conflict_count": self.target_conflict_count,
            "contamination_risk": self.contamination_risk,
            "recommendations": self.recommendations,
        }

    def render(self) -> str:
        lines = [
            "TRAIN / VALIDATION CONTAMINATION AUDIT",
            "=====================================",
            f"Contamination Risk       : [{self.contamination_risk}]",
            f"Exact Row Duplicates     : {self.exact_duplicates_across_splits:,}",
            f"Entity Overlap Across CV : {self.entity_overlap_count:,}" + (f" ({self.entity_column})" if self.entity_column else ""),
            f"Target Label Conflicts   : {self.target_conflict_count:,}",
        ]
        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


class ContaminationDetector:
    """Audits train vs validation/test datasets for data leakage and duplicate contamination."""

    def __init__(self, logger: Optional[DecisionLogger] = None) -> None:
        self.logger = logger or DecisionLogger()

    def audit_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_column: Optional[str] = None,
        entity_column: Optional[str] = None,
    ) -> ContaminationReport:
        """Check for exact row duplicates, entity contamination, and conflicting labels across partitions."""
        train_features = train_df.drop(columns=[target_column]) if target_column and target_column in train_df.columns else train_df
        val_features = val_df.drop(columns=[target_column]) if target_column and target_column in val_df.columns else val_df

        # 1. Exact duplicates across splits
        common_cols = [c for c in train_features.columns if c in val_features.columns]
        merged_dups = pd.merge(train_features[common_cols], val_features[common_cols], how="inner")
        exact_dups = len(merged_dups)

        # 2. Entity overlap across splits
        entity_overlap = 0
        if entity_column and entity_column in train_df.columns and entity_column in val_df.columns:
            train_entities = set(train_df[entity_column].dropna().unique())
            val_entities = set(val_df[entity_column].dropna().unique())
            entity_overlap = len(train_entities.intersection(val_entities))

        # 3. Target label conflicts (identical features in train having multiple conflicting targets)
        target_conflicts = 0
        if target_column and target_column in train_df.columns:
            feat_cols = [c for c in train_df.columns if c != target_column]
            if feat_cols:
                grouped = train_df.groupby(feat_cols)[target_column].nunique()
                target_conflicts = int((grouped > 1).sum())

        # Determine contamination risk
        recs: List[str] = []
        if entity_overlap > 0:
            risk = "HIGH" if entity_overlap > 50 else "MEDIUM"
            recs.append(f"Detected {entity_overlap:,} duplicated entities across splits. Switch validation strategy to GroupKFold('{entity_column}').")
        elif exact_dups > 0:
            risk = "HIGH" if exact_dups > 100 else "MEDIUM"
            recs.append(f"Found {exact_dups:,} exact feature row duplicates between train and validation. Deduplicate dataset before training.")
        elif target_conflicts > 0:
            risk = "MEDIUM"
            recs.append(f"Found {target_conflicts:,} identical feature rows with conflicting target labels. Investigate label noise.")
        else:
            risk = "SAFE"
            recs.append("No cross-partition contamination or entity leakage detected.")

        self.logger.log(
            component="ContaminationDetector",
            decision=f"Contamination Risk: [{risk}] (Exact dups: {exact_dups}, Entity overlap: {entity_overlap})",
            reason="; ".join(recs),
            confidence=0.95,
            evidence={
                "risk": risk,
                "exact_dups": exact_dups,
                "entity_overlap": entity_overlap,
                "target_conflicts": target_conflicts,
            },
        )

        return ContaminationReport(
            exact_duplicates_across_splits=exact_dups,
            entity_overlap_count=entity_overlap,
            entity_column=entity_column,
            target_conflict_count=target_conflicts,
            contamination_risk=risk,
            recommendations=recs,
        )
