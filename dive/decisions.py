"""Structured Decision Record Logging Engine - `dive/decisions.py`.

Core Principle: DIVE Must Make & Explain Decisions.
Every subsystem logs structured DecisionRecords detailing:
1. Component / Engine name.
2. Action / Decision taken.
3. Rationale / Reasons (human & machine readable).
4. Confidence score (0.0 to 1.0).
5. Empirical evidence metrics backing the decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionRecord:
    """Structured record of an automated engineering decision made by DIVE."""

    component: str
    decision: str
    reason: str
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "decision": self.decision,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }

    def render(self) -> str:
        conf_str = f"{int(self.confidence * 100)}%"
        lines = [
            f"[{self.component}] Decision: {self.decision} (Confidence: {conf_str})",
            f"  Reason: {self.reason}",
        ]
        if self.evidence:
            ev_str = ", ".join(f"{k}={v}" for k, v in self.evidence.items())
            lines.append(f"  Evidence: {ev_str}")
        return "\n".join(lines)


class DecisionLogger:
    """Central decision registry for an AutoML Study."""

    def __init__(self) -> None:
        self.records: List[DecisionRecord] = []

    def log(
        self,
        component: str,
        decision: str,
        reason: str,
        confidence: float = 1.0,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> DecisionRecord:
        """Log an explicit decision made by DIVE."""
        record = DecisionRecord(
            component=component,
            decision=decision,
            reason=reason,
            confidence=confidence,
            evidence=evidence or {},
        )
        self.records.append(record)
        return record

    def get_by_component(self, component: str) -> List[DecisionRecord]:
        return [r for r in self.records if r.component.lower() == component.lower()]

    def to_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def render_summary(self) -> str:
        if not self.records:
            return "No decisions recorded."
        lines = ["DIVE AUTOMATED DECISION LOG", "==========================="]
        for r in self.records:
            lines.append(r.render())
            lines.append("")
        return "\n".join(lines)
