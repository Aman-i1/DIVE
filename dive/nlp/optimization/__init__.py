"""DIVE NLP Optimization Layer - `dive/nlp/optimization`.

Provides inference acceleration, micro-batching, LRU prediction caching, and ONNX runtime optimization.
"""

from __future__ import annotations

from dive.nlp.optimization.batching import BatchInferenceEngine
from dive.nlp.optimization.cache import PredictionCache
from dive.nlp.optimization.onnx import (
    ONNXNLPPredictor,
    export_nlp_to_onnx,
)
from dive.nlp.optimization.predictor import (
    OptimizedNLPPredictor,
    optimize_nlp_predictor,
)

__all__ = [
    "PredictionCache",
    "BatchInferenceEngine",
    "ONNXNLPPredictor",
    "OptimizedNLPPredictor",
    "export_nlp_to_onnx",
    "optimize_nlp_predictor",
]
