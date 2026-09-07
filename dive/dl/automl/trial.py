"""AutoDL Trial record dataclass - ``dive/dl/automl/trial.py``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dive.dl.config import DLConfig
from dive.dl.core.callbacks import EpochRecord


@dataclass
class DLTrial:
    """Encapsulates a single architecture / hyperparameter trial in AutoDL."""

    trial_id: int
    config: DLConfig
    model_name: str
    backend: str
    primary_metric: str = "accuracy"
    primary_metric_score: float = 0.0
    composite_score: float = 0.0
    train_time_ms: float = 0.0
    inference_latency_ms: float = 0.0
    status: str = "SUCCESS"
    error: Optional[str] = None
    history: List[EpochRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "model_name": self.model_name,
            "backend": self.backend,
            "primary_metric": self.primary_metric,
            "primary_metric_score": round(self.primary_metric_score, 4),
            "composite_score": round(self.composite_score, 4),
            "train_time_ms": round(self.train_time_ms, 2),
            "inference_latency_ms": round(self.inference_latency_ms, 3),
            "status": self.status,
            "error": self.error,
        }
