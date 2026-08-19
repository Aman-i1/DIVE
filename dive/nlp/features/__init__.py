"""DIVE NLP Feature Representation Layer - `dive/nlp/features`.

Provides sparse and dense text feature representations:
- Word TF-IDF and N-Grams
- Character N-Grams (subword and typo robustness)
- Joint Word + Character Feature Unions
- BM25 Probabilistic Relevance Weighting
- Dense Neural Embeddings (Sentence Transformers)
- Bag-of-Words Counts
"""

from __future__ import annotations

from typing import Optional, Union

from dive.nlp.config import NLPRepresentationConfig
from dive.nlp.embeddings.representation import EmbeddingRepresentation
from dive.nlp.features.bm25 import BM25Representation
from dive.nlp.features.ngrams import (
    CharNGramRepresentation,
    WordCharUnionRepresentation,
)
from dive.nlp.features.tfidf import CountRepresentation, TFIDFRepresentation


def build_representation(
    config: Optional[NLPRepresentationConfig] = None,
    representation_type: str = "tfidf",
) -> Union[
    TFIDFRepresentation,
    CharNGramRepresentation,
    WordCharUnionRepresentation,
    BM25Representation,
    CountRepresentation,
    EmbeddingRepresentation,
]:
    """Factory creating configured text feature representation."""
    rep_type = config.representation_type if config else representation_type

    if rep_type in ("char_ngrams", "char_ngram", "char"):
        return CharNGramRepresentation()
    elif rep_type in ("word_char_union", "union", "hybrid"):
        return WordCharUnionRepresentation()
    elif rep_type in ("bm25", "okapi_bm25"):
        return BM25Representation()
    elif rep_type in ("embedding", "embeddings", "dense", "sentence_transformers"):
        model_name = config.embedding_model if config and config.embedding_model else "all-MiniLM-L6-v2"
        return EmbeddingRepresentation(model_name=model_name)
    elif rep_type in ("count", "bow"):
        return CountRepresentation()
    return TFIDFRepresentation(config=config)


__all__ = [
    "TFIDFRepresentation",
    "CharNGramRepresentation",
    "WordCharUnionRepresentation",
    "BM25Representation",
    "CountRepresentation",
    "EmbeddingRepresentation",
    "build_representation",
]
