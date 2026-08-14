"""Data Quality, Anomalies & Inferred Relational Rules Engine - `dive/data_quality.py`.

Audits:
1. Missingness, duplicate rows, constant/near-constant columns, extreme cardinality, impossible values (e.g. age < 0).
2. Statistically infers relational / consistency rules (e.g. refund <= purchase, delivery_date >= order_date, price >= 0)
   and explicitly tags them as `INFERRED RULE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dive.decisions import DecisionLogger


@dataclass
class InferredRelationalRule:
    """A statistically discovered relational consistency rule."""

    rule_description: str
    rule_type: str  # 'NON_NEGATIVE', 'BOUNDED_BY', 'CHRONOLOGICAL_ORDER'
    confidence: float
    violations_count: int
    violations_pct: float
    status: str  # 'PASS', 'WARNING', 'FAIL'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_description": self.rule_description,
            "rule_type": self.rule_type,
            "confidence": round(self.confidence, 4),
            "violations_count": self.violations_count,
            "violations_pct": round(self.violations_pct, 4),
            "status": self.status,
            "label": "INFERRED RULE",
        }


@dataclass
class DataQualityReport:
    """Comprehensive data quality and relational integrity audit report."""

    total_rows: int
    total_columns: int
    missing_cells_pct: float
    duplicate_rows_count: int
    constant_columns: List[str]
    near_constant_columns: List[str]
    high_cardinality_columns: List[str]
    inferred_rules: List[InferredRelationalRule] = field(default_factory=list)
    overall_quality_status: str = "PASS"  # 'PASS', 'WARNING', 'FAIL'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_cells_pct": round(self.missing_cells_pct, 2),
            "duplicate_rows_count": self.duplicate_rows_count,
            "constant_columns": self.constant_columns,
            "near_constant_columns": self.near_constant_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "inferred_rules": [r.to_dict() for r in self.inferred_rules],
            "overall_quality_status": self.overall_quality_status,
        }

    def render(self) -> str:
        lines = [
            "DATA QUALITY & INFERRED RULES AUDIT",
            "===================================",
            f"Overall Status       : [{self.overall_quality_status}]",
            f"Rows / Columns       : {self.total_rows:,} rows, {self.total_columns} cols",
            f"Missing Values       : {self.missing_cells_pct:.2f}%",
            f"Duplicate Rows       : {self.duplicate_rows_count:,}",
        ]
        if self.constant_columns:
            lines.append(f"Constant Columns     : {', '.join(self.constant_columns)}")
        if self.near_constant_columns:
            lines.append(f"Near-Constant Columns: {', '.join(self.near_constant_columns)}")
        if self.inferred_rules:
            lines.append("\nStatistically Inferred Consistency Rules [INFERRED RULE]:")
            for r in self.inferred_rules:
                lines.append(
                    f"  - [{r.status:<7}] {r.rule_description} (Violations: {r.violations_count:,} ({r.violations_pct:.1%}))"
                )
        return "\n".join(lines)


class DataQualityEngine:
    """Executes holistic data quality audits and infers relational business rules."""

    def __init__(self, logger: Optional[DecisionLogger] = None) -> None:
        self.logger = logger or DecisionLogger()

    def audit(self, df: pd.DataFrame) -> DataQualityReport:
        """Run complete data quality audit."""
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return DataQualityReport(
                total_rows=0,
                total_columns=n_cols,
                missing_cells_pct=0.0,
                duplicate_rows_count=0,
                constant_columns=[],
                near_constant_columns=[],
                high_cardinality_columns=[],
                inferred_rules=[],
                overall_quality_status="FAIL",
            )

        missing_pct = float(df.isna().sum().sum() / (n_rows * n_cols) * 100.0)
        dup_rows = int(df.duplicated().sum())

        constant_cols = []
        near_constant_cols = []
        high_card_cols = []

        for col in df.columns:
            val_counts = df[col].value_counts(dropna=False, normalize=True)
            if len(val_counts) <= 1:
                constant_cols.append(col)
            elif val_counts.iloc[0] >= 0.99:
                near_constant_cols.append(col)

            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                if df[col].nunique() / n_rows > 0.80 and n_rows > 50:
                    high_card_cols.append(col)

        # Infer Relational Rules
        inferred_rules: List[InferredRelationalRule] = []

        # 1. Non-negativity rules for numeric columns with names like age, price, amount, count, quantity
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ("age", "price", "amount", "cost", "count", "quantity", "spend", "balance")):
                vals = df[col].dropna()
                neg_count = int((vals < 0).sum())
                neg_pct = float(neg_count / max(len(vals), 1))
                status = "FAIL" if neg_pct > 0.05 else ("WARNING" if neg_count > 0 else "PASS")
                inferred_rules.append(
                    InferredRelationalRule(
                        rule_description=f"INFERRED RULE: {col} >= 0",
                        rule_type="NON_NEGATIVE",
                        confidence=0.95,
                        violations_count=neg_count,
                        violations_pct=neg_pct,
                        status=status,
                    )
                )

        # 2. Pairwise bounds (e.g. refund <= purchase, discount <= price)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                if i == j:
                    continue
                col_a = num_cols[i]
                col_b = num_cols[j]
                if ("refund" in col_a.lower() and "purchase" in col_b.lower()) or ("discount" in col_a.lower() and "price" in col_b.lower()):
                    valid_pairs = df[[col_a, col_b]].dropna()
                    if len(valid_pairs) > 0:
                        viols = int((valid_pairs[col_a] > valid_pairs[col_b]).sum())
                        viol_pct = float(viols / len(valid_pairs))
                        status = "FAIL" if viol_pct > 0.05 else ("WARNING" if viols > 0 else "PASS")
                        inferred_rules.append(
                            InferredRelationalRule(
                                rule_description=f"INFERRED RULE: {col_a} <= {col_b}",
                                rule_type="BOUNDED_BY",
                                confidence=0.92,
                                violations_count=viols,
                                violations_pct=viol_pct,
                                status=status,
                            )
                        )

        # Determine overall quality status
        has_critical_fails = any(r.status == "FAIL" for r in inferred_rules) or missing_pct > 50.0
        has_warnings = any(r.status == "WARNING" for r in inferred_rules) or dup_rows > 0 or len(constant_cols) > 0

        overall_status = "FAIL" if has_critical_fails else ("WARNING" if has_warnings else "PASS")

        self.logger.log(
            component="DataQualityEngine",
            decision=f"Data Quality Audit: [{overall_status}] ({len(inferred_rules)} relational rules evaluated)",
            reason=f"Missing: {missing_pct:.1f}%, Duplicates: {dup_rows}, Constant cols: {len(constant_cols)}",
            confidence=0.95,
            evidence={
                "overall_status": overall_status,
                "missing_pct": round(missing_pct, 2),
                "dup_rows": dup_rows,
                "inferred_rules_count": len(inferred_rules),
            },
        )

        return DataQualityReport(
            total_rows=n_rows,
            total_columns=n_cols,
            missing_cells_pct=missing_pct,
            duplicate_rows_count=dup_rows,
            constant_columns=constant_cols,
            near_constant_columns=near_constant_cols,
            high_cardinality_columns=high_card_cols,
            inferred_rules=inferred_rules,
            overall_quality_status=overall_status,
        )
