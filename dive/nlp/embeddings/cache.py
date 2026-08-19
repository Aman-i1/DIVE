"""Persistent & In-Memory Embedding Cache - `dive/nlp/embeddings/cache.py`.

Caches text embeddings keyed by SHA-256 content hashes to avoid redundant
neural forward passes across repeated documents, validation folds, and inference batches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dive.utils.io import ensure_dir, load_pickle, save_pickle


class EmbeddingCache:
    """Two-tier (RAM + Disk) content-addressable cache for text embeddings."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        self._memory_cache: Dict[str, np.ndarray] = {}
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            ensure_dir(self.cache_dir)

    @staticmethod
    def hash_text(text: str) -> str:
        """Compute SHA-256 digest of normalized text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding vector for text, or None if not found."""
        key = self.hash_text(text)
        if key in self._memory_cache:
            return self._memory_cache[key]

        if self.cache_dir:
            file_path = self.cache_dir / f"{key}.npy"
            if file_path.exists():
                try:
                    vec = np.load(file_path)
                    self._memory_cache[key] = vec
                    return vec
                except Exception:
                    pass
        return None

    def set(self, text: str, vector: np.ndarray) -> None:
        """Store embedding vector in memory and on disk."""
        key = self.hash_text(text)
        vec = np.asarray(vector, dtype=np.float32)
        self._memory_cache[key] = vec

        if self.cache_dir:
            file_path = self.cache_dir / f"{key}.npy"
            try:
                np.save(file_path, vec)
            except Exception:
                pass

    def batch_get(
        self, texts: Sequence[str]
    ) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """Retrieve cached vectors for a batch of texts.

        Returns
        -------
        Tuple[Dict[int, np.ndarray], List[int]]
            Dictionary mapping index -> cached vector, and list of missing indices.
        """
        cached_hits: Dict[int, np.ndarray] = {}
        missing_indices: List[int] = []

        for idx, text in enumerate(texts):
            vec = self.get(text)
            if vec is not None:
                cached_hits[idx] = vec
            else:
                missing_indices.append(idx)

        return cached_hits, missing_indices

    def batch_set(self, items: Sequence[Tuple[str, np.ndarray]]) -> None:
        """Store a sequence of (text, vector) pairs."""
        for text, vec in items:
            self.set(text, vec)

    def clear(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache.clear()

    def __len__(self) -> int:
        return len(self._memory_cache)
