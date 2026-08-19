"""In-Memory LRU Prediction Cache - `dive/nlp/optimization/cache.py`.

Provides high-speed, thread-safe LRU caching for frequent, repeated NLP inference queries,
dramatically reducing p99 latency for high-throughput production endpoints.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PredictionCache:
    """Thread-safe, high-performance Least-Recently-Used (LRU) inference cache."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = max(1, capacity)
        self._cache: OrderedDict[str, Tuple[Any, Optional[np.ndarray]]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Tuple[Any, Optional[np.ndarray]]]:
        """Retrieve cached prediction and probability tuple, or None."""
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end to represent most recently used
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def set(
        self, text: str, prediction: Any, probas: Optional[np.ndarray] = None
    ) -> None:
        """Store prediction and probability in cache with LRU eviction."""
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.capacity:
                    # Pop oldest (first) item
                    self._cache.popitem(last=False)
            self._cache[key] = (prediction, probas)

    def clear(self) -> None:
        """Clear all cached entries and reset performance counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit ratio."""
        total = self._hits + self._misses
        return float(self._hits / total) if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Return cache health and performance statistics."""
        with self._lock:
            return {
                "capacity": self.capacity,
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self.hit_rate, 4),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
