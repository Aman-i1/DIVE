"""DIVE NLP Evaluation Layer - `dive/nlp/evaluation`.

Provides evaluation metrics for classification and regression NLP tasks.
"""

from __future__ import annotations

from dive.nlp.evaluation.evaluator import NLPEvaluator, evaluate_nlp_predictions

__all__ = [
    "NLPEvaluator",
    "evaluate_nlp_predictions",
]
