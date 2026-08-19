"""Adaptive Micro-Batching Inference Engine - `dive/nlp/optimization/batching.py`.

Partitions large inference payloads into optimal micro-batches to maximize CPU/GPU
throughput and prevent memory allocation spikes.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, Union

import numpy as np


class BatchInferenceEngine:
    """Adaptive micro-batching orchestrator for NLP inference."""

    def __init__(self, default_batch_size: int = 64) -> None:
        self.default_batch_size = max(1, default_batch_size)

    def run_batched(
        self,
        func: Callable[[Sequence[str]], Union[np.ndarray, List[Any]]],
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Execute a prediction function over micro-batches and concatenate outputs."""
        bs = batch_size or self.default_batch_size
        n = len(texts)
        if n == 0:
            return np.array([])

        if n <= bs:
            res = func(texts)
            return np.asarray(res)

        batches = []
        for i in range(0, n, bs):
            chunk = texts[i : i + bs]
            chunk_out = func(chunk)
            batches.append(np.asarray(chunk_out))

        # Check if 1D or 2D
        if len(batches) > 0 and batches[0].ndim == 2:
            return np.vstack(batches)
        return np.concatenate(batches)
