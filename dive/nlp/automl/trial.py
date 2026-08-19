"""AutoNLP Candidate Trial Representation - `dive/nlp/automl/trial.py`.

Encapsulates individual trial execution metadata, representation & model choices,
statistical metrics, inference latency measurements, and multi-objective composite scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NLPTrial:
    """Record of an individual model candidate trial in AutoNLP exploration."""

    trial_id: int
    representation_type: str
    model_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    primary_metric_name: str = "macro_f1"
    primary_metric_score: float = 0.0
    train_time_ms: float = 0.0
    inference_latency_ms: float = 0.0
    composite_score: float = 0.0
    status: str = "PENDING"  # PENDING, SUCCESS, FAILED, TIMEOUT
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trial summary to dictionary format."""
        return {
            "trial_id": self.trial_id,
            "representation_type": self.representation_type,
            "model_name": self.model_name,
            "primary_metric": self.primary_metric_name,
            "primary_score": round(self.primary_metric_score, 4),
            "composite_score": round(self.composite_score, 4),
            "train_time_ms": round(self.train_time_ms, 2),
            "inference_latency_ms": round(self.inference_latency_ms, 3),
            "status": self.status,
            "metrics": self.metrics,
            "error_msg": self.error_msg,
        }
