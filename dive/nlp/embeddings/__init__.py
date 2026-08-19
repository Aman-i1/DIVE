"""DIVE NLP Embeddings Layer - `dive/nlp/embeddings`.

Provides dense vector embeddings, embedding caching, and empirical benchmarking tools.
"""

from __future__ import annotations

from dive.nlp.embeddings.benchmark import benchmark_tfidf_vs_embeddings
from dive.nlp.embeddings.cache import EmbeddingCache
from dive.nlp.embeddings.representation import (
    EmbeddingRepresentation,
    compute_semantic_similarity,
)

__all__ = [
    "EmbeddingRepresentation",
    "EmbeddingCache",
    "compute_semantic_similarity",
    "benchmark_tfidf_vs_embeddings",
]
