"""Targetless Dataset Inspection Engine - `dive info`.

Provides a comprehensive structural and semantic summary of an unknown dataset without
requiring a prior target column specification. Automatically infers candidate target columns,
problem types (classification vs regression), data types, missingness, and column roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dive.data_intelligence import looks_like_datetime
from dive.utils.logging import Style


@dataclass
class CandidateTarget:
    """Inferred target column candidate with problem type and confidence score."""

    column_name: str
    problem_type: str  # classification or regression
    confidence_score: float  # 0.0 to 1.0
    n_unique: int
    dtype: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "problem_type": self.problem_type,
            "confidence_score": round(self.confidence_score, 2),
            "n_unique": self.n_unique,
            "dtype": self.dtype,
            "reason": self.reason,
        }


@dataclass
class ColumnProfile:
    """Individual column summary profile."""

    name: str
    dtype: str
    inferred_role: str  # CANDIDATE_TARGET, ID_LIKE, DATETIME, NUMERIC_FEATURE, CATEGORICAL_FEATURE, CONSTANT
    missing_pct: float
    n_unique: int
    example_values: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "inferred_role": self.inferred_role,
            "missing_pct": round(self.missing_pct, 2),
            "n_unique": self.n_unique,
            "example_values": self.example_values,
        }


@dataclass
class DatasetInfoReport:
    """Full dataset summary report produced by DatasetInspector.inspect()."""

    n_rows: int
    n_cols: int
    total_memory_mb: float
    total_missing_pct: float
    candidate_targets: List[CandidateTarget]
    column_profiles: List[ColumnProfile]
    numeric_count: int
    categorical_count: int
    datetime_count: int
    constant_cols: List[str]
    id_cols: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "total_memory_mb": round(self.total_memory_mb, 2),
            "total_missing_pct": round(self.total_missing_pct, 2),
            "candidate_targets": [c.to_dict() for c in self.candidate_targets],
            "column_profiles": [c.to_dict() for c in self.column_profiles],
            "numeric_count": self.numeric_count,
            "categorical_count": self.categorical_count,
            "datetime_count": self.datetime_count,
            "constant_cols": self.constant_cols,
            "id_cols": self.id_cols,
        }

    def render(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║                  DIVE DATASET INSPECTION                     ║",
            "╚══════════════════════════════════════════════════════════════╝",
            f"OVERVIEW         Rows: {self.n_rows:,} | Cols: {self.n_cols} | RAM: {self.total_memory_mb:.2f} MB | Missing: {self.total_missing_pct:.1f}%",
            f"TYPES            Numeric: {self.numeric_count} | Categorical: {self.categorical_count} | Datetime: {self.datetime_count} | ID-like: {len(self.id_cols)}",
            "",
            "INFERRED TARGET CANDIDATES:",
        ]

        if not self.candidate_targets:
            lines.append("  (No obvious target candidates detected. Specify --target manually.)")
        else:
            for i, ct in enumerate(self.candidate_targets[:3], 1):
                star = "[TOP]" if i == 1 else "     "
                lines.append(
                    f"  {star} {i}. Candidate Column: '{ct.column_name}'"
                )
                lines.append(
                    f"        Inferred Problem: {ct.problem_type.upper()} ({ct.n_unique} unique values, {ct.dtype})"
                )
                lines.append(f"        Reason: {ct.reason}")

        lines.append("")
        lines.append("COLUMN PROFILES (Preview):")
        lines.append(f"  {'Column':<22} {'Role':<18} {'Dtype':<10} {'Missing%':<10} {'Unique':<8} {'Examples'}")
        lines.append("  " + "-" * 78)

        for cp in self.column_profiles[:15]:
            ex_str = ", ".join(map(str, cp.example_values[:3]))
            if len(ex_str) > 22:
                ex_str = ex_str[:19] + "..."
            lines.append(
                f"  {cp.name:<22} {cp.inferred_role:<18} {cp.dtype:<10} {cp.missing_pct:<10.1f} {cp.n_unique:<8} {ex_str}"
            )

        if len(self.column_profiles) > 15:
            lines.append(f"  ... (+{len(self.column_profiles) - 15} more columns)")

        lines.append("")
        lines.append("RECOMMENDED NEXT STEPS:")
        if self.candidate_targets:
            top_target = self.candidate_targets[0].column_name
            lines.append(f"  1. Audit ML Readiness: dive doctor <data_file> --target {top_target}")
            lines.append(f"  2. Train AutoML Model : dive train  <data_file> --target {top_target}")
        else:
            lines.append("  1. Choose your target column from the list above.")
            lines.append("  2. Run: dive doctor <data_file> --target <your_target>")

        return "\n".join(lines)


class DatasetInspector:
    """Targetless dataset inspection and target inference engine."""

    def __init__(self) -> None:
        self.target_keywords = [
            "target", "label", "class", "churn", "status", "y", "outcome",
            "sale_price", "saleprice", "price", "revenue", "is_churned",
            "is_fraud", "fraud", "converted", "approved", "rejected", "diagnosis",
        ]

    def inspect(self, df: pd.DataFrame) -> DatasetInfoReport:
        """Inspect dataframe and infer target candidates, roles, and types."""
        n_rows, n_cols = df.shape
        raw_bytes = df.memory_usage(deep=True).sum()
        total_mem_mb = raw_bytes / (1024 * 1024)
        total_missing_pct = float((df.isna().sum().sum() / max(1, n_rows * n_cols)) * 100.0)

        # 1. Infer Column Roles & Datetimes
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == object and looks_like_datetime(df[col]):
                datetime_cols.append(col)

        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        id_cols = self._detect_id_cols(df)

        # 2. Score Candidate Targets
        candidates = self._score_target_candidates(df, constant_cols, id_cols)

        # 3. Create Column Profiles
        candidate_names = {c.column_name for c in candidates}
        profiles = []

        num_count = 0
        cat_count = 0

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            missing_pct = float(series.isna().mean() * 100.0)
            n_unq = int(series.nunique(dropna=True))
            ex_vals = [str(v) for v in series.dropna().head(3).tolist()]

            # Determine role
            if col in candidate_names:
                role = "CANDIDATE_TARGET"
            elif col in constant_cols:
                role = "CONSTANT"
            elif col in id_cols:
                role = "ID_LIKE"
            elif col in datetime_cols:
                role = "DATETIME"
            elif pd.api.types.is_numeric_dtype(series):
                role = "NUMERIC_FEATURE"
                num_count += 1
            else:
                role = "CATEGORICAL_FEATURE"
                cat_count += 1

            profiles.append(
                ColumnProfile(
                    name=str(col),
                    dtype=dtype_str,
                    inferred_role=role,
                    missing_pct=missing_pct,
                    n_unique=n_unq,
                    example_values=ex_vals,
                )
            )

        return DatasetInfoReport(
            n_rows=n_rows,
            n_cols=n_cols,
            total_memory_mb=total_mem_mb,
            total_missing_pct=total_missing_pct,
            candidate_targets=candidates,
            column_profiles=profiles,
            numeric_count=num_count,
            categorical_count=cat_count,
            datetime_count=len(datetime_cols),
            constant_cols=constant_cols,
            id_cols=id_cols,
        )

    def _score_target_candidates(
        self, df: pd.DataFrame, constant_cols: List[str], id_cols: List[str]
    ) -> List[CandidateTarget]:
        n_rows = len(df)
        candidates = []

        for col in df.columns:
            if col in constant_cols or col in id_cols:
                continue

            series = df[col]
            col_lower = str(col).lower()
            n_unq = series.nunique(dropna=True)
            dtype_str = str(series.dtype)

            score = 0.0
            reasons = []

            # Keyword match bonus
            kw_match = [kw for kw in self.target_keywords if kw in col_lower or col_lower == kw]
            if kw_match:
                score += 0.50
                reasons.append(f"Name matches target pattern '{kw_match[0]}'")

            # Position heuristic: last column in dataset is often the target
            if col == df.columns[-1]:
                score += 0.25
                reasons.append("Position is the last column in dataset")

            # Cardinality / Problem type inference
            problem_type = "classification"
            if series.dtype == object or pd.api.types.is_string_dtype(series) or series.dtype == bool:
                problem_type = "classification"
                if 2 <= n_unq <= 20:
                    score += 0.20
                    reasons.append(f"Categorical with {n_unq} distinct class labels")
            elif pd.api.types.is_numeric_dtype(series):
                if n_unq == 2:
                    problem_type = "classification"
                    score += 0.25
                    reasons.append("Binary 0/1 numeric target signature")
                elif 2 < n_unq <= 15:
                    problem_type = "classification"
                    score += 0.15
                    reasons.append(f"Low-cardinality discrete numeric ({n_unq} classes)")
                else:
                    problem_type = "regression"
                    if n_unq < n_rows * 0.9:
                        score += 0.10
                        reasons.append(f"Continuous numeric range ({n_unq} unique values)")

            if score > 0.15:
                candidates.append(
                    CandidateTarget(
                        column_name=str(col),
                        problem_type=problem_type,
                        confidence_score=min(1.0, score),
                        n_unique=n_unq,
                        dtype=dtype_str,
                        reason="; ".join(reasons),
                    )
                )

        candidates.sort(key=lambda c: -c.confidence_score)
        return candidates

    def _detect_id_cols(self, df: pd.DataFrame) -> List[str]:
        id_names = {"id", "index", "row_id", "rowid", "uuid", "guid", "key", "customer_id", "user_id"}
        id_suffixes = ("_id", "_uuid", "_guid", "_key", "_no")
        n_rows = len(df)
        id_cols = []

        for col in df.columns:
            name_lower = str(col).lower()
            n_unq = df[col].nunique(dropna=True)
            if (name_lower in id_names or name_lower.endswith(id_suffixes)) and n_unq > 0.85 * n_rows:
                id_cols.append(str(col))
        return id_cols
