"""DIVE NLP Dataset Profiling & Diagnostic Auditing - `dive/nlp/profiling`.

Provides automated statistical profiling, length distribution analysis,
vocabulary analytics, duplicate auditing, and label contamination checks.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union
import pandas as pd

from dive.nlp.data.dataset import NLPDataset
from dive.nlp.profiling.profiler import NLPProfileReport, NLPProfiler


def profile_nlp_dataset(
    dataset: Union[NLPDataset, Sequence[str], pd.DataFrame],
    text_column: Optional[str] = None,
    target_column: Optional[str] = None,
    imbalance_threshold: float = 3.0,
    top_k_tokens: int = 20,
) -> NLPProfileReport:
    """Convenience helper to profile an NLP dataset in a single call."""
    profiler = NLPProfiler(imbalance_threshold=imbalance_threshold, top_k_tokens=top_k_tokens)
    return profiler.profile(
        dataset=dataset,
        text_column=text_column,
        target_column=target_column,
    )


__all__ = [
    "NLPProfiler",
    "NLPProfileReport",
    "profile_nlp_dataset",
]
