"""Dynamic Uncertainty & Latency-Aware Inference Router - `dive/inference_router.py`.

Routes inference requests dynamically:
- High confidence / low uncertainty samples -> Fast path (single lightweight model, < 5ms).
- Ambiguous / high uncertainty samples -> Complex path (full calibrated ensemble stack).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class RoutedPredictionResult:
    """Outcome of dynamically routed inference."""

    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    routing_decisions: List[str]  # 'FAST_PATH' or 'ENSEMBLE_PATH'
    pct_fast_path: float
    pct_ensemble_path: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pct_fast_path": round(self.pct_fast_path, 1),
            "pct_ensemble_path": round(self.pct_ensemble_path, 1),
            "fast_count": int(sum(1 for d in self.routing_decisions if d == "FAST_PATH")),
            "ensemble_count": int(sum(1 for d in self.routing_decisions if d == "ENSEMBLE_PATH")),
        }


class DynamicInferenceRouter:
    """Routes inference queries between fast lightweight estimators and calibrated ensembles."""

    def __init__(
        self,
        fast_estimator: Any,
        ensemble_estimator: Any,
        confidence_threshold: float = 0.85,
        problem_type: str = "classification",
    ) -> None:
        self.fast_estimator = fast_estimator
        self.ensemble_estimator = ensemble_estimator
        self.confidence_threshold = confidence_threshold
        self.problem_type = problem_type

    def predict(self, X: pd.DataFrame) -> RoutedPredictionResult:
        """Route predictions dynamically based on initial confidence estimates."""
        n_samples = len(X)
        if n_samples == 0:
            return RoutedPredictionResult(
                predictions=np.array([]),
                probabilities=None,
                routing_decisions=[],
                pct_fast_path=100.0,
                pct_ensemble_path=0.0,
            )

        # 1. Generate fast path initial predictions & confidences
        fast_preds = self.fast_estimator.predict(X)
        has_proba = hasattr(self.fast_estimator, "predict_proba")

        if self.problem_type == "classification" and has_proba:
            fast_probs = self.fast_estimator.predict_proba(X)
            # Confidence is the maximum class probability
            confidences = np.max(fast_probs, axis=1)
        else:
            fast_probs = None
            confidences = np.ones(n_samples) * 0.90  # Default moderate confidence for regression

        # 2. Determine routing decision per sample
        needs_ensemble = confidences < self.confidence_threshold
        routing_decisions = ["ENSEMBLE_PATH" if flag else "FAST_PATH" for flag in needs_ensemble]

        final_preds = np.copy(fast_preds)
        final_probs = np.copy(fast_probs) if fast_probs is not None else None

        # 3. Execute ensemble for ambiguous subset only
        if np.any(needs_ensemble):
            idx_ensemble = np.where(needs_ensemble)[0]
            X_ambiguous = X.iloc[idx_ensemble]

            ens_preds = self.ensemble_estimator.predict(X_ambiguous)
            final_preds[idx_ensemble] = ens_preds

            if final_probs is not None and hasattr(self.ensemble_estimator, "predict_proba"):
                ens_probs = self.ensemble_estimator.predict_proba(X_ambiguous)
                final_probs[idx_ensemble] = ens_probs

        pct_fast = float(np.mean([1 if d == "FAST_PATH" else 0 for d in routing_decisions]) * 100.0)
        pct_ens = 100.0 - pct_fast

        return RoutedPredictionResult(
            predictions=final_preds,
            probabilities=final_probs,
            routing_decisions=routing_decisions,
            pct_fast_path=pct_fast,
            pct_ensemble_path=pct_ens,
        )
