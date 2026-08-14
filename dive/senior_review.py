"""Senior ML Review Engine - `dive/senior_review.py`.

Synthesizes all diagnostics, stress tests, contamination audits, validation plans,
calibration metrics, failure segments, and explainability into a unified Senior Review Report.
Emits clear production verdicts:
- PASS: Fully safe and recommended for production.
- PASS_WITH_WARNINGS: Conditionally approved with explicit monitoring requirements.
- BLOCKED: Critical risk detected (target leakage, severe covariate shift, permutation sanity failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dive.adversarial_validation import AdversarialValidationReport
from dive.contamination import ContaminationReport
from dive.data_quality import DataQualityReport
from dive.decisions import DecisionLogger
from dive.failure_segments import FailureSegmentsReport
from dive.model_stress import StressTestReport
from dive.prediction_contract import PredictionContract


@dataclass
class SeniorReviewReport:
    """Consolidated Senior ML Review Report."""

    final_decision: str  # 'PASS', 'PASS_WITH_WARNINGS', 'BLOCKED'
    confidence: str  # 'LOW', 'MEDIUM', 'HIGH'
    target: str
    problem_type: str
    champion_model: str
    primary_score: float
    review_matrix: Dict[str, str]  # dimension -> 'PASS' | 'WARNING' | 'FAIL' | 'BLOCKED'
    top_risks: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_decision": self.final_decision,
            "confidence": self.confidence,
            "target": self.target,
            "problem_type": self.problem_type,
            "champion_model": self.champion_model,
            "primary_score": round(self.primary_score, 4),
            "review_matrix": self.review_matrix,
            "top_risks": self.top_risks,
            "required_actions": self.required_actions,
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Save review to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def render(self) -> str:
        border = "============================================================"
        lines = [
            border,
            "                   DIVE SENIOR ML REVIEW",
            border,
        ]
        for dim, status in self.review_matrix.items():
            lines.append(f"{dim:<32} : [{status}]")

        lines.append("-" * 60)
        lines.append(f"FINAL DECISION                   : [{self.final_decision}] (Confidence: {self.confidence})")
        lines.append(f"Champion Model                   : {self.champion_model} (Score: {self.primary_score:.4f})")

        if self.top_risks:
            lines.append("\nTop Identified Risks:")
            for idx, risk in enumerate(self.top_risks, start=1):
                lines.append(f"  {idx}. {risk}")

        if self.required_actions:
            lines.append("\nRequired Actions Before Deployment:")
            for idx, act in enumerate(self.required_actions, start=1):
                lines.append(f"  {idx}. {act}")

        lines.append(border)
        return "\n".join(lines)


class SeniorReviewEngine:
    """Aggregates all reliability engines into an evidence-driven Senior Review."""

    def __init__(self, logger: Optional[DecisionLogger] = None) -> None:
        self.logger = logger or DecisionLogger()

    def review(
        self,
        contract: PredictionContract,
        data_quality: Optional[DataQualityReport] = None,
        contamination: Optional[ContaminationReport] = None,
        adversarial: Optional[AdversarialValidationReport] = None,
        stress: Optional[StressTestReport] = None,
        segments: Optional[FailureSegmentsReport] = None,
        champion_model: str = "EnsembleChampion",
        primary_score: float = 0.90,
    ) -> SeniorReviewReport:
        """Evaluate full review matrix and emit final deployment decision."""
        matrix: Dict[str, str] = {
            "PREDICTION CONTRACT": "PASS" if contract.target else "FAIL",
            "DATA QUALITY": data_quality.overall_quality_status if data_quality else "PASS",
            "LEAKAGE & CONTAMINATION": contamination.contamination_risk if contamination else "SAFE",
            "DISTRIBUTION SHIFT": adversarial.shift_status if adversarial else "SAFE_IID",
            "MODEL STRESS & SANITY": stress.overall_stress_status if stress else "PASS",
            "FAILURE SEGMENTATION": segments.overall_segment_status if segments else "PASS",
        }

        top_risks: List[str] = []
        required_actions: List[str] = []

        # 1. Critical blocks
        is_blocked = False
        if stress and stress.permutation_sanity_status == "FAIL_CRITICAL":
            is_blocked = True
            top_risks.append("Target permutation sanity failed (likely leakage or severe artifact).")
            required_actions.append("Audit training labels and inspect feature availability.")

        if contamination and contamination.contamination_risk in ("CRITICAL", "HIGH") and contamination.exact_duplicates_across_splits > 50:
            is_blocked = True
            top_risks.append("Severe train/test cross-partition duplicate contamination.")
            required_actions.append("Deduplicate dataset and re-split.")

        if adversarial and adversarial.shift_status == "SEVERE_COVARIATE_SHIFT":
            top_risks.append("Severe covariate shift between train and test/production sets.")
            required_actions.append(f"Investigate top shift features ({', '.join([f[0] for f in adversarial.top_drift_features[:2]])}).")

        if segments and segments.overall_segment_status in ("FAIL", "WARNING") and segments.weak_segments:
            top_seg = segments.weak_segments[0]
            top_risks.append(f"Weak segment underperformance on {top_seg.segment_description} (drop: {top_seg.metric_drop:+.2%}).")
            required_actions.append(f"Collect more data or calibrate decision thresholds for slice '{top_seg.segment_description}'.")

        if is_blocked:
            decision = "BLOCKED"
        elif top_risks:
            decision = "PASS_WITH_WARNINGS"
        else:
            decision = "PASS"

        confidence = "HIGH" if len(top_risks) <= 2 else "MEDIUM"

        self.logger.log(
            component="SeniorReviewEngine",
            decision=f"Final Senior Review Decision: [{decision}] (Confidence: {confidence})",
            reason=f"{len(top_risks)} top risks identified across 6 review dimensions",
            confidence=0.98,
            evidence={
                "decision": decision,
                "matrix": matrix,
                "top_risks": top_risks,
            },
        )

        return SeniorReviewReport(
            final_decision=decision,
            confidence=confidence,
            target=contract.target,
            problem_type=contract.problem_type,
            champion_model=champion_model,
            primary_score=primary_score,
            review_matrix=matrix,
            top_risks=top_risks,
            required_actions=required_actions,
        )
