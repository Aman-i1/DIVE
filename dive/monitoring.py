"""Model Monitoring System - Privacy-Safe Request & Latency Logger.

Tracks production inference request stats, latency percentiles, prediction distributions,
and schema errors without logging sensitive raw features by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dive.utils.io import ensure_dir, save_json, load_json


@dataclass
class MonitoringSnapshot:
    """Summary snapshot of monitoring metrics over a time window."""

    total_requests: int
    schema_failures: int
    avg_latency_ms: float
    p95_latency_ms: float
    prediction_distribution: Dict[str, Any]
    missing_value_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "schema_failures": self.schema_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "prediction_distribution": self.prediction_distribution,
            "missing_value_counts": self.missing_value_counts,
        }


class ModelMonitor:
    """Logs production request statistics without exposing raw sensitive payload data."""

    def __init__(self, log_dir: Union[str, Path] = ".dive/monitoring") -> None:
        self.log_dir = ensure_dir(log_dir)
        self.requests_log: List[Dict[str, Any]] = []

    def log_inference(
        self,
        prediction_val: Any,
        latency_ms: float,
        missing_count: int = 0,
        schema_valid: bool = True,
    ) -> None:
        """Record privacy-safe request summary."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pred": str(prediction_val),
            "latency_ms": float(latency_ms),
            "missing_count": int(missing_count),
            "schema_valid": bool(schema_valid),
        }
        self.requests_log.append(entry)

        if len(self.requests_log) >= 50:
            self.flush()

    def flush(self) -> None:
        """Persist current log buffer to disk."""
        if not self.requests_log:
            return
        fname = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(self.log_dir / fname, self.requests_log)
        self.requests_log = []

    def get_summary(self) -> MonitoringSnapshot:
        """Aggregate all logged request history into a MonitoringSnapshot."""
        all_logs = []
        for f in self.log_dir.glob("log_*.json"):
            try:
                all_logs.extend(load_json(f))
            except Exception:
                pass
        all_logs.extend(self.requests_log)

        if not all_logs:
            return MonitoringSnapshot(
                total_requests=0,
                schema_failures=0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                prediction_distribution={},
                missing_value_counts={},
            )

        total = len(all_logs)
        schema_fails = sum(1 for entry in all_logs if not entry.get("schema_valid", True))
        latencies = [entry["latency_ms"] for entry in all_logs if "latency_ms" in entry]
        
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        # Prediction value counts
        preds = [str(entry.get("pred")) for entry in all_logs if "pred" in entry]
        unique_preds, counts = np.unique(preds, return_counts=True)
        pred_dist = {str(k): int(v) for k, v in zip(unique_preds, counts)}

        return MonitoringSnapshot(
            total_requests=total,
            schema_failures=schema_fails,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            prediction_distribution=pred_dist,
            missing_value_counts={},
        )
