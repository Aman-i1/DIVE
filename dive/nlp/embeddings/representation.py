"""Dense Vector Embedding Representation Engine - `dive/nlp/embeddings/representation.py`.

Provides pluggable dense text embeddings utilizing Sentence Transformers,
built-in hardware acceleration detection (CUDA / MPS / CPU), and two-tier caching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from dive.nlp.embeddings.cache import EmbeddingCache
from dive.nlp.interfaces import NLPRepresentationProtocol
from dive.utils.optional import is_available, load_optional

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Detect available hardware accelerator."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class EmbeddingRepresentation:
    """Dense vector embedding representation conforming to NLPRepresentationProtocol."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache: Optional[EmbeddingCache] = None,
        use_cache: bool = True,
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or _detect_device()
        self.normalize_embeddings = normalize_embeddings
        self.use_cache = use_cache
        self.cache = cache or (EmbeddingCache() if use_cache else None)
        self.dimension = dimension

        self._model: Any = None
        self._is_fallback: bool = False
        self.fitted_ = False

    def _load_model(self) -> None:
        """Lazy load SentenceTransformer model or initialize deterministic embedding engine."""
        if self._model is not None:
            return

        if is_available("sentence_transformers"):
            try:
                st_mod = load_optional("sentence_transformers", purpose="dense text embeddings")
                SentenceTransformer = getattr(st_mod, "SentenceTransformer")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._is_fallback = False
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load SentenceTransformer '{self.model_name}': {e}. "
                    "Falling back to deterministic semantic hashing projection."
                )

        # Fallback for environments without heavy downloads / sentence-transformers
        self._is_fallback = True
        self._model = HashingVectorizer(
            n_features=self.dimension,
            norm="l2" if self.normalize_embeddings else None,
            alternate_sign=True,
            token_pattern=r"(?u)\b\w+\b",
        )

    def fit(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> "EmbeddingRepresentation":
        """Initialize and verify embedding backend."""
        self._load_model()
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of texts into a dense 2D numpy embedding matrix."""
        if not self.fitted_:
            self.fit(texts)

        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts_list = [str(t) for t in texts]

        # 1. Check cache for hit indices
        if self.use_cache and self.cache:
            cached_hits, missing_indices = self.cache.batch_get(texts_list)
        else:
            cached_hits, missing_indices = {}, list(range(len(texts_list)))

        # 2. Encode missing texts
        if missing_indices:
            missing_texts = [texts_list[i] for i in missing_indices]

            if not self._is_fallback and hasattr(self._model, "encode"):
                encoded = self._model.encode(
                    missing_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                )
            else:
                # Deterministic projection fallback
                sparse_h = self._model.transform(missing_texts)
                encoded = sparse_h.toarray().astype(np.float32)
                if self.normalize_embeddings:
                    norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    encoded = encoded / norms

            # Populate results and optionally update cache
            for idx_pos, orig_idx in enumerate(missing_indices):
                vec = encoded[idx_pos]
                cached_hits[orig_idx] = vec
                if self.use_cache and self.cache:
                    self.cache.set(texts_list[orig_idx], vec)

        # 3. Assemble full result matrix in original sequence order
        out_dim = self.dimension if self._is_fallback else (cached_hits[0].shape[0] if cached_hits else self.dimension)
        res = np.empty((len(texts_list), out_dim), dtype=np.float32)
        for i in range(len(texts_list)):
            res[i] = cached_hits[i]

        return res

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> np.ndarray:
        return self.fit(texts, y).transform(texts)

    @property
    def n_features_(self) -> int:
        return self.dimension

    @property
    def feature_names_(self) -> List[str]:
        return [f"emb_{i}" for i in range(self.dimension)]


def compute_semantic_similarity(
    query_embeddings: np.ndarray, doc_embeddings: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity matrix between query and document vectors."""
    q = np.asarray(query_embeddings, dtype=np.float32)
    d = np.asarray(doc_embeddings, dtype=np.float32)

    if q.ndim == 1:
        q = q.reshape(1, -1)
    if d.ndim == 1:
        d = d.reshape(1, -1)

    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    d_norm[d_norm == 0] = 1.0

    return (q / q_norm) @ (d / d_norm).T
