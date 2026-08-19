"""DIVE NLP Inference Layer - `dive/nlp/inference`.

Provides deployable NLP predictor artifacts, serialization, and input coercion.
"""

from __future__ import annotations

from dive.nlp.inference.predictor import (
    NLPPredictor,
    load_nlp_predictor,
    save_nlp_predictor,
)

__all__ = [
    "NLPPredictor",
    "save_nlp_predictor",
    "load_nlp_predictor",
]
