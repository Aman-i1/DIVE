"""DIVE NLP Monitoring & Drift Detection Layer - `dive/nlp/monitoring`.

Provides real-time text distribution monitoring, vocabulary OOV shift detection, and prediction PSI analysis.
"""

from __future__ import annotations

from dive.nlp.monitoring.drift import (
    NLPDriftMonitor,
    NLPDriftReport,
    monitor_nlp_drift,
)

__all__ = [
    "NLPDriftMonitor",
    "NLPDriftReport",
    "monitor_nlp_drift",
]
