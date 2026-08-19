"""Production Optimized NLP Predictor Wrapper - `dive/nlp/optimization/predictor.py`.

Combines in-memory LRU prediction caching, adaptive micro-batching, and telemetry tracking
into a production-hardened wrapper conforming to NLPPredictorProtocol.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dive.nlp.inference.predictor import NLPPredictor
from dive.nlp.interfaces import NLPPredictorProtocol
from dive.nlp.optimization.batching import BatchInferenceEngine
from dive.nlp.optimization.cache import PredictionCache


class OptimizedNLPPredictor:
    """Production-ready optimized NLP Predictor implementing NLPPredictorProtocol."""

    def __init__(
        self,
        base_predictor: NLPPredictor,
        enable_cache: bool = True,
        cache_capacity: int = 10000,
        batch_size: int = 64,
    ) -> None:
        self.base_predictor = base_predictor
        self.enable_cache = enable_cache
        self.cache = PredictionCache(capacity=cache_capacity) if enable_cache else None
        self.batcher = BatchInferenceEngine(default_batch_size=batch_size)

        self._total_inferences = 0
        self._total_latency_ms = 0.0

    @property
    def model_name(self) -> str:
        return f"Optimized_{self.base_predictor.model_name}"

    @property
    def has_proba(self) -> bool:
        return self.base_predictor.has_proba

    @property
    def class_names(self) -> Optional[List[str]]:
        return self.base_predictor.class_names

    def predict(
        self, data: Union[str, Sequence[str], pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> np.ndarray:
        """Run accelerated prediction with micro-batching and LRU caching."""
        t0 = time.perf_counter()
        texts = self.base_predictor._coerce_texts(data)
        if not texts:
            return np.array([])

        # Fast single query cache lookup
        if self.cache and len(texts) == 1:
            hit = self.cache.get(texts[0])
            if hit is not None:
                self._record_telemetry(time.perf_counter() - t0)
                return np.array([hit[0]], dtype=object)

        # Batch execution
        preds = self.batcher.run_batched(self.base_predictor.pipeline.predict, texts)

        # Update cache for single or small queries
        if self.cache and len(texts) <= 50:
            for idx, text in enumerate(texts):
                self.cache.set(text, preds[idx])

        self._record_telemetry(time.perf_counter() - t0)
        return preds

    def predict_proba(
        self, data: Union[str, Sequence[str], pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> np.ndarray:
        """Run accelerated probability estimation with micro-batching and caching."""
        t0 = time.perf_counter()
        texts = self.base_predictor._coerce_texts(data)
        if not texts:
            return np.empty((0, len(self.class_names or [])))

        # Single query cache lookup
        if self.cache and len(texts) == 1:
            hit = self.cache.get(texts[0])
            if hit is not None and hit[1] is not None:
                self._record_telemetry(time.perf_counter() - t0)
                return hit[1].reshape(1, -1)

        probas = self.batcher.run_batched(self.base_predictor.pipeline.predict_proba, texts)

        # Update cache
        if self.cache and len(texts) <= 50:
            for idx, text in enumerate(texts):
                pred_val = self.class_names[np.argmax(probas[idx])] if self.class_names else None
                self.cache.set(text, pred_val, probas[idx])

        self._record_telemetry(time.perf_counter() - t0)
        return probas

    def _record_telemetry(self, elapsed_sec: float) -> None:
        self._total_inferences += 1
        self._total_latency_ms += elapsed_sec * 1000.0

    def stats(self) -> Dict[str, Any]:
        """Return operational runtime metrics."""
        avg_latency = (
            self._total_latency_ms / self._total_inferences
            if self._total_inferences > 0
            else 0.0
        )
        return {
            "model_name": self.model_name,
            "total_requests": self._total_inferences,
            "avg_latency_ms": round(avg_latency, 3),
            "cache_stats": self.cache.stats() if self.cache else {"cache_enabled": False},
        }

    def describe_input(self) -> Dict[str, Any]:
        d = self.base_predictor.describe_input()
        d["optimized"] = True
        d["cache_enabled"] = self.enable_cache
        return d


def optimize_nlp_predictor(
    predictor: NLPPredictor,
    enable_cache: bool = True,
    cache_capacity: int = 10000,
    batch_size: int = 64,
) -> OptimizedNLPPredictor:
    """Factory creating an OptimizedNLPPredictor deployment artifact."""
    return OptimizedNLPPredictor(
        base_predictor=predictor,
        enable_cache=enable_cache,
        cache_capacity=cache_capacity,
        batch_size=batch_size,
    )
