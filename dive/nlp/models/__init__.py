"""DIVE NLP Model Zoo - `dive/nlp/models`.

Provides classical baselines, embeddings, and deep learning estimators.
"""

from __future__ import annotations

from dive.nlp.models.baselines import BASELINE_MODELS, build_baseline_model

__all__ = [
    "BASELINE_MODELS",
    "build_baseline_model",
]
