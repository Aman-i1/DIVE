"""Production Deployment Gate Engine - `dive gate`.

Evaluates production dataset batches against model invariants, schema checks,
data leakage, and PSI drift limits, returning exit code 0 (PASS) or 1 (FAIL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from dive.drift import DriftDetector
from dive.leakage import AdvancedLeakageDetector
from dive.predictor import DivePredictor
from dive.utils.logging import Console, get_console


@dataclass
class GateVerdict:
    """Deployment gate evaluation verdict."""

    passed: bool
    status: str  # DEPLOYMENT_APPROVED, DEPLOYMENT_REJECTED
    schema_ok: bool
    leakage_ok: bool
    drift_ok: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "schema_ok": self.schema_ok,
            "leakage_ok": self.leakage_ok,
            "drift_ok": self.drift_ok,
            "reasons": self.reasons,
        }


class DeploymentGate:
    """Production deployment gatekeeper."""

    def __init__(self, max_psi_threshold: float = 0.25, strict: bool = False) -> None:
        self.max_psi_threshold = max_psi_threshold
        self.strict = strict

    def evaluate(
        self,
        predictor: DivePredictor,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> GateVerdict:
        """Evaluate new batch against model schema, leakage, and drift."""
        reasons: List[str] = []
        schema_ok = True
        leakage_ok = True
        drift_ok = True

        # 1. Schema check
        req_cols = predictor.required_columns
        missing = [c for c in req_cols if c not in current_df.columns]
        if missing:
            schema_ok = False
            reasons.append(f"Missing required columns: {', '.join(missing)}")

        # 2. Leakage check
        leakage_detector = AdvancedLeakageDetector(target=predictor.target)
        leakage_report = leakage_detector.detect(current_df)
        if leakage_report.has_critical_leakage:
            leakage_ok = False
            reasons.append("Critical target leakage detected in current data batch.")

        # 3. Drift check if reference dataframe is provided
        if reference_df is not None:
            drift_detector = DriftDetector()
            drift_report = drift_detector.detect(reference_df, current_df, target=predictor.target)
            if drift_report.overall_drift_detected:
                drift_ok = False
                reasons.append("Significant distribution drift detected (PSI > threshold).")

        passed = schema_ok and leakage_ok and drift_ok
        status = "DEPLOYMENT_APPROVED" if passed else "DEPLOYMENT_REJECTED"

        return GateVerdict(
            passed=passed,
            status=status,
            schema_ok=schema_ok,
            leakage_ok=leakage_ok,
            drift_ok=drift_ok,
            reasons=reasons,
        )
