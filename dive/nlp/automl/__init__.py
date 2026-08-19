"""DIVE AutoNLP Engine - `dive/nlp/automl`.

Provides automated trial scheduling, multi-candidate search, and multi-objective model selection.
"""

from __future__ import annotations

from dive.nlp.automl.engine import AutoNLP, fit_nlp
from dive.nlp.automl.leaderboard import NLPLeaderboard
from dive.nlp.automl.trial import NLPTrial

__all__ = [
    "AutoNLP",
    "fit_nlp",
    "NLPLeaderboard",
    "NLPTrial",
]
