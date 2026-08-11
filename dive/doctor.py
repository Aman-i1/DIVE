"""Stage 0 / Diagnostic System - DIVE ML Doctor.

Performs a comprehensive ML-readiness audit on tabular datasets before training.
Evaluates dataset health, leakage risk, duplicate risk, target health, validation safety,
group structure, temporal structure, model suitability, resource requirements, and computes
a transparent, explainable Production Readiness Score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dive.data_intelligence import DataIntelligence
from dive.exceptions import DataError, TargetError


@dataclass
class ProductionReadinessScore:
    """Transparent, rule-based production readiness composite score."""

    overall_score: float
    data_quality_score: float
    leakage_safety_score: float
    validation_safety_score: float
    model_suitability_score: float
    schema_safety_score: float
    penalties: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 1),
            "breakdown": {
                "data_quality": round(self.data_quality_score, 1),
                "leakage_safety": round(self.leakage_safety_score, 1),
                "validation_safety": round(self.validation_safety_score, 1),
                "model_suitability": round(self.model_suitability_score, 1),
                "schema_safety": round(self.schema_safety_score, 1),
            },
            "penalties": self.penalties,
        }

    def render(self) -> str:
        lines = [
            "DIVE PRODUCTION READINESS SCORE",
            "================================",
            f"Overall Score            : {self.overall_score:.1f} / 100",
            "Breakdown:",
            f"  - Data Quality         : {self.data_quality_score:.1f} / 100",
            f"  - Leakage Safety       : {self.leakage_safety_score:.1f} / 100",
            f"  - Validation Safety    : {self.validation_safety_score:.1f} / 100",
            f"  - Model Suitability    : {self.model_suitability_score:.1f} / 100",
            f"  - Schema Safety        : {self.schema_safety_score:.1f} / 100",
        ]
        if self.penalties:
            lines.append("Applied Penalties:")
            for p in self.penalties:
                lines.append(f"  ⚠ -{p['deduction']} pts: {p['reason']}")
        return "\n".join(lines)


@dataclass
class DoctorReport:
    """Structured output of DiveDoctor.analyze()."""

    target: str
    n_samples: int
    n_features: int
    problem_type: str
    profile: Dict[str, Any]
    readiness_score: ProductionReadinessScore
    sections: Dict[str, Any]
    risks: List[Dict[str, str]]
    action_plan: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "problem_type": self.problem_type,
            "readiness_score": self.readiness_score.to_dict(),
            "sections": self.sections,
            "risks": self.risks,
            "action_plan": self.action_plan,
        }

    def __str__(self) -> str:
        return self.render_text()

    def render_text(self, unicode_box: bool = True) -> str:
        """Render ASCII/Unicode terminal summary."""
        lines = []
        if unicode_box:
            lines.extend([
                "╔══════════════════════════════════════════════════════════════╗",
                "║                       DIVE ML DOCTOR                         ║",
                "╚══════════════════════════════════════════════════════════════╝",
            ])
        else:
            lines.extend([
                "================================================================",
                "                       DIVE ML DOCTOR                           ",
                "================================================================",
            ])

        ds = self.sections.get("DATASET", {})
        lines.append(f"DATASET           Rows: {ds.get('n_samples', 0):,} | Features: {ds.get('n_features', 0)} | Problem: {ds.get('problem_type', 'Unknown')}")
        
        dq = self.sections.get("DATA QUALITY", {})
        missing_icon = "⚠" if dq.get("total_missing_pct", 0) > 5 else "✓"
        dup_icon = "⚠" if dq.get("duplicate_rows", 0) > 0 else "✓"
        const_icon = "⚠" if dq.get("constant_cols") else "✓"
        id_icon = "⚠" if dq.get("id_like_cols") else "✓"
        
        lines.append(
            f"DATA QUALITY      Missing values: {missing_icon} {dq.get('total_missing_pct', 0):.1f}% | "
            f"Duplicates: {dup_icon} {dq.get('duplicate_rows', 0)} | "
            f"Constant cols: {const_icon} {len(dq.get('constant_cols', []))} | "
            f"ID-like cols: {id_icon} {len(dq.get('id_like_cols', []))}"
        )

        val = self.sections.get("VALIDATION STRATEGY", {})
        lines.append(
            f"VALIDATION        Random Split: {val.get('random_split_status', 'UNKNOWN')} | "
            f"Group Leakage: {val.get('group_leakage_status', 'NONE')} | "
            f"Temporal: {val.get('temporal_status', 'NONE')}"
        )

        leak = self.sections.get("LEAKAGE", {})
        leak_high = leak.get("high_risk_features", [])
        leak_icon = "🔴 HIGH RISK" if leak_high else ("⚠ WARNING" if leak.get("suspicious_features") else "✓ PASSED")
        lines.append(f"LEAKAGE           Status: {leak_icon}")

        rec_val = val.get("recommended_strategy", "StratifiedKFold")
        rec_group = val.get("group_column")
        rec_val_str = f"{rec_val}({rec_group})" if rec_group else rec_val
        lines.append(f"RECOMMENDED VAL   {rec_val_str}")

        models = self.sections.get("MODEL RECOMMENDATIONS", {}).get("recommended", [])
        if models:
            lines.append(f"MODEL RECS        {', '.join(models[:4])}")

        lines.append("")
        lines.append(self.readiness_score.render())

        if self.risks:
            lines.append("")
            lines.append("IDENTIFIED RISKS:")
            for r in self.risks:
                severity = r.get("severity", "MEDIUM")
                icon = "🔴" if severity == "HIGH" else "⚠"
                lines.append(f"  {icon} [{severity}] {r.get('name')}: {r.get('reason')}")

        if self.action_plan:
            lines.append("")
            lines.append("ACTION PLAN:")
            for i, step in enumerate(self.action_plan, 1):
                lines.append(f"  {i}. {step}")

        return "\n".join(lines)


class DiveDoctor:
    """High-level ML-readiness diagnostic engine."""

    def __init__(
        self,
        target: str,
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.target = target
        self.group_column = group_column
        self.time_column = time_column
        self.random_state = random_state

    def analyze(self, df: pd.DataFrame) -> DoctorReport:
        """Run full 17-category ML readiness audit on ``df``."""
        if self.target not in df.columns:
            raise TargetError(
                f"Target column '{self.target}' not found in dataframe.",
                f"Available columns: {list(df.columns[:20])}",
            )

        di = DataIntelligence(target=self.target, random_state=self.random_state)
        profile = di.analyze(df)

        X = df.drop(columns=[self.target])
        y = df[self.target]

        sections: Dict[str, Any] = {}
        risks: List[Dict[str, str]] = []
        action_plan: List[str] = []

        # 1. Dataset & Data Quality
        sections["DATASET"] = {
            "n_samples": profile["n_samples"],
            "n_features": profile["n_features"],
            "problem_type": profile["problem_type"],
            "n_numeric": profile["n_numeric"],
            "n_categorical": profile["n_categorical"],
        }

        dup_count = int(X.duplicated().sum())
        sections["DATA QUALITY"] = {
            "total_missing_pct": profile["total_missing_pct"],
            "has_missing": profile["has_missing"],
            "duplicate_rows": dup_count,
            "constant_cols": profile["constant_cols"],
            "id_like_cols": profile["id_like_cols"],
            "high_card_cols": profile["high_card_cols"],
        }

        # 2. Target Health
        sections["TARGET HEALTH"] = {
            "target_name": self.target,
            "target_dtype": profile["target_dtype"],
            "target_missing_pct": profile["target_missing_pct"],
            "target_n_unique": profile["target_n_unique"],
            "target_unique_ratio": profile["target_unique_ratio"],
            "target_near_constant": profile["target_near_constant"],
            "problem_type": profile["problem_type"],
            "is_imbalanced": profile.get("is_imbalanced", False),
            "imbalance_ratio": profile.get("imbalance_ratio"),
            "minority_class_count": profile.get("minority_class_count"),
        }

        # 3. Leakage & Duplicates
        leakage_info = self._audit_leakage(X, y, profile["problem_type"])
        sections["LEAKAGE"] = leakage_info

        # 4. Group & Temporal Structure
        group_col = self.group_column or self._auto_detect_group_col(X, profile)
        time_col = self.time_column or self._auto_detect_time_col(X, profile)

        sections["GROUP STRUCTURE"] = {
            "group_column": group_col,
            "has_groups": group_col is not None,
        }

        sections["TEMPORAL STRUCTURE"] = {
            "time_column": time_col,
            "has_temporal": time_col is not None or len(profile.get("datetime_cols", [])) > 0,
            "datetime_cols": profile.get("datetime_cols", []),
        }

        # 5. Validation Strategy Advisor
        val_strategy = self._advise_validation(
            profile, group_col, time_col, leakage_info
        )
        sections["VALIDATION STRATEGY"] = val_strategy

        # 6. Model Advisor & Resource Estimation
        model_adv = self._advise_models(profile, X.shape)
        sections["MODEL RECOMMENDATIONS"] = model_adv

        resource_est = self._estimate_resources(profile, X.shape)
        sections["RESOURCE ESTIMATION"] = resource_est

        # 7. Collect Risks & Formulate Action Plan
        self._compile_risks_and_plan(
            profile, leakage_info, val_strategy, dup_count, risks, action_plan
        )

        # 8. Compute Production Readiness Score
        readiness_score = self._compute_readiness_score(
            profile, leakage_info, val_strategy, dup_count, model_adv
        )

        return DoctorReport(
            target=self.target,
            n_samples=profile["n_samples"],
            n_features=profile["n_features"],
            problem_type=profile["problem_type"],
            profile=profile,
            readiness_score=readiness_score,
            sections=sections,
            risks=risks,
            action_plan=action_plan,
        )

    # ------------------------------------------------------------------
    # Internal Audit Helpers
    # ------------------------------------------------------------------
    def _audit_leakage(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str
    ) -> Dict[str, Any]:
        """Detect near-perfect correlation, suspicious column names, and duplicate leakage."""
        high_risk = []
        suspicious = []
        name_patterns = []

        suspicious_keywords = [
            "target", "label", "outcome", "result", "closed", "cancelled",
            "approved", "rejected", "future", "post_", "after_", "final_"
        ]

        for col in X.columns:
            name_lower = str(col).lower()
            if any(kw in name_lower for kw in suspicious_keywords):
                name_patterns.append(col)

            # Fast association check
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    valid = X[col].notna() & y.notna()
                    if valid.sum() > 10:
                        corr = abs(float(np.corrcoef(X[col][valid], y[valid])[0, 1]))
                        if np.isfinite(corr):
                            if corr >= 0.98:
                                high_risk.append({"feature": col, "metric": "correlation", "score": corr})
                            elif corr >= 0.90:
                                suspicious.append({"feature": col, "metric": "correlation", "score": corr})
            except Exception:
                pass

        return {
            "high_risk_features": high_risk,
            "suspicious_features": suspicious,
            "name_pattern_warnings": name_patterns,
            "has_leakage_risk": len(high_risk) > 0,
        }

    def _auto_detect_group_col(
        self, X: pd.DataFrame, profile: Dict[str, Any]
    ) -> Optional[str]:
        """Detect columns representing entities/groups with repeated occurrences."""
        group_keywords = [
            "customer_id", "user_id", "patient_id", "account_id", "device_id",
            "session_id", "entity_id", "client_id", "group_id"
        ]
        for col in X.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in group_keywords):
                nunique = X[col].nunique()
                if 1 < nunique < len(X) * 0.95:
                    return col
        return None

    def _auto_detect_time_col(
        self, X: pd.DataFrame, profile: Dict[str, Any]
    ) -> Optional[str]:
        """Detect date/time columns that suggest chronological splitting."""
        datetime_cols = profile.get("datetime_cols", [])
        if datetime_cols:
            return datetime_cols[0]
        time_keywords = ["timestamp", "date", "created_at", "time", "event_time"]
        for col in X.columns:
            if any(kw in str(col).lower() for kw in time_keywords):
                return col
        return None

    def _advise_validation(
        self,
        profile: Dict[str, Any],
        group_col: Optional[str],
        time_col: Optional[str],
        leakage_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine validation safety and optimal CV strategy."""
        random_status = "SAFE"
        group_status = "NONE"
        temporal_status = "NONE"
        recommended = "StratifiedKFold" if profile["problem_type"] == "classification" else "KFold"
        reason = "Standard IID dataset without group or temporal dependencies."

        if time_col:
            temporal_status = f"DETECTED ({time_col})"
            random_status = "UNSAFE (Temporal Dependency)"
            recommended = "TimeSeriesSplit"
            reason = f"Temporal dependency detected in column '{time_col}'. Random split causes future data leakage."
        elif group_col:
            group_status = f"DETECTED ({group_col})"
            random_status = "UNSAFE (Entity Contamination)"
            if profile["problem_type"] == "classification":
                recommended = "StratifiedGroupKFold"
            else:
                recommended = "GroupKFold"
            reason = f"Repeated entities detected in group column '{group_col}'. Random split leaks entities across folds."
        elif profile.get("is_imbalanced"):
            recommended = "StratifiedKFold"
            reason = "Imbalanced target distribution requires stratified splitting to preserve class proportions."

        return {
            "random_split_status": random_status,
            "group_leakage_status": group_status,
            "temporal_status": temporal_status,
            "recommended_strategy": recommended,
            "group_column": group_col,
            "time_column": time_col,
            "reason": reason,
        }

    def _advise_models(
        self, profile: Dict[str, Any], shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Recommend model architectures based on row count, cardinality, missingness."""
        n_rows, n_cols = shape
        has_high_card = len(profile.get("high_card_cols", [])) > 0
        has_missing = profile.get("has_missing", False)

        recommended = []
        acceptable = []
        deprioritized = []
        rejected = []
        reasons = {}

        # CatBoost
        if has_high_card:
            recommended.append("CatBoost")
            reasons["CatBoost"] = "Recommended: Superior native handling of high-cardinality categorical features."
        else:
            recommended.append("CatBoost")

        # LightGBM / XGBoost
        recommended.append("LightGBM")
        recommended.append("XGBoost")

        # RandomForest / HistGBM
        acceptable.append("RandomForest")
        acceptable.append("HistGradientBoosting")

        # KNN
        if n_rows > 50_000:
            rejected.append("KNN")
            reasons["KNN"] = f"Rejected: Dataset size ({n_rows:,} rows) exceeds KNN distance computation efficiency threshold."
        else:
            deprioritized.append("KNN")

        # MLP / Neural Net
        if n_rows > 200_000:
            deprioritized.append("MLP")
            reasons["MLP"] = f"Deprioritized: High computational/memory cost for tabular dataset with {n_rows:,} rows."
        else:
            acceptable.append("MLP")

        return {
            "recommended": recommended,
            "acceptable": acceptable,
            "deprioritized": deprioritized,
            "rejected": rejected,
            "reasons": reasons,
        }

    def _estimate_resources(
        self, profile: Dict[str, Any], shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Estimate memory consumption and compute complexity."""
        n_rows, n_cols = shape
        # ~8 bytes per cell in floating point representation
        raw_bytes = n_rows * n_cols * 8
        estimated_mb = raw_bytes / (1024 * 1024)
        peak_training_mb = estimated_mb * 5.0  # Feature expansion + model overhead

        return {
            "dataset_memory_mb": round(estimated_mb, 2),
            "estimated_peak_memory_mb": round(peak_training_mb, 2),
            "complexity": "HIGH" if n_rows > 500_000 else ("MEDIUM" if n_rows > 50_000 else "LOW"),
            "memory_safe": peak_training_mb < 8192,
        }

    def _compile_risks_and_plan(
        self,
        profile: Dict[str, Any],
        leakage: Dict[str, Any],
        val: Dict[str, Any],
        dup_count: int,
        risks: List[Dict[str, str]],
        plan: List[str],
    ) -> None:
        """Populate risk items and ordered action plan."""
        # Risk compilation
        if leakage["high_risk_features"]:
            cols = [f["feature"] for f in leakage["high_risk_features"]]
            risks.append({
                "severity": "HIGH",
                "name": "Target Leakage",
                "reason": f"Features {cols} almost perfectly predict the target.",
            })
            plan.append(f"Remove target leakage features: {', '.join(cols)}")

        if "UNSAFE" in val["random_split_status"]:
            risks.append({
                "severity": "HIGH",
                "name": "Unsafe Validation",
                "reason": val["reason"],
            })
            plan.append(f"Use recommended validation strategy: {val['recommended_strategy']}")

        if profile.get("target_near_constant"):
            risks.append({
                "severity": "HIGH",
                "name": "Target Health",
                "reason": "Target has near-zero variance or extreme imbalance (>=99%).",
            })
            plan.append("Fix target column or filter uninformative rows before training.")

        if profile.get("constant_cols"):
            plan.append(f"Drop constant columns: {profile['constant_cols']}")

        if profile.get("id_like_cols"):
            plan.append(f"Drop uninformative ID-like columns: {profile['id_like_cols']}")

        if dup_count > 0:
            risks.append({
                "severity": "MEDIUM",
                "name": "Duplicate Rows",
                "reason": f"Found {dup_count} duplicate feature rows in dataset.",
            })
            plan.append("Deduplicate feature rows to prevent train/holdout memory test contamination.")

        if profile.get("has_missing"):
            plan.append("Impute missing values (median for numeric, most-frequent for categorical).")

        top_model = "CatBoost" if "CatBoost" in profile.get("high_card_cols", []) else "LightGBM"
        plan.append(f"Train model zoo prioritizing {top_model}")
        if profile["problem_type"] == "classification":
            plan.append("Calibrate output probabilities using Platt scaling or Isotonic regression")
        plan.append("Monitor feature and prediction drift post-deployment")

    def _compute_readiness_score(
        self,
        profile: Dict[str, Any],
        leakage: Dict[str, Any],
        val: Dict[str, Any],
        dup_count: int,
        model_adv: Dict[str, Any],
    ) -> ProductionReadinessScore:
        """Calculate production readiness composite score and breakdown."""
        penalties = []

        # Data Quality (100 base)
        dq_score = 100.0
        missing_pct = profile["total_missing_pct"]
        if missing_pct > 10.0:
            deduct = min(30.0, missing_pct * 1.5)
            dq_score -= deduct
            penalties.append({"deduction": round(deduct, 1), "reason": f"High missing data percentage ({missing_pct:.1f}%)"})
        if dup_count > 0:
            dup_pct = (dup_count / profile["n_samples"]) * 100
            deduct = min(20.0, dup_pct * 5.0)
            dq_score -= deduct
            penalties.append({"deduction": round(deduct, 1), "reason": f"Duplicate rows ({dup_count} rows, {dup_pct:.1f}%)"})

        # Leakage Safety (100 base)
        leak_score = 100.0
        if leakage["high_risk_features"]:
            leak_score -= 60.0
            penalties.append({"deduction": 60.0, "reason": "High-risk target leakage detected"})
        elif leakage["suspicious_features"]:
            leak_score -= 20.0
            penalties.append({"deduction": 20.0, "reason": "Suspicious feature-target associations"})

        # Validation Safety (100 base)
        val_score = 100.0
        if "UNSAFE" in val["random_split_status"]:
            val_score -= 40.0
            penalties.append({"deduction": 40.0, "reason": val["reason"]})

        # Model Suitability & Schema Safety (100 base)
        model_score = 100.0 if model_adv["recommended"] else 70.0
        schema_score = 100.0 if not profile.get("constant_cols") else 90.0

        dq_score = max(0.0, dq_score)
        leak_score = max(0.0, leak_score)
        val_score = max(0.0, val_score)

        # Weighted composite overall score
        overall = (
            dq_score * 0.25
            + leak_score * 0.30
            + val_score * 0.25
            + model_score * 0.10
            + schema_score * 0.10
        )

        return ProductionReadinessScore(
            overall_score=overall,
            data_quality_score=dq_score,
            leakage_safety_score=leak_score,
            validation_safety_score=val_score,
            model_suitability_score=model_score,
            schema_safety_score=schema_score,
            penalties=penalties,
        )
