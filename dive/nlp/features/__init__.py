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

from typing import Any, Optional, Union

from dive.nlp.config import NLPRepresentationConfig
from dive.nlp.embeddings.representation import EmbeddingRepresentation
from dive.nlp.features.bm25 import BM25Representation
from dive.nlp.features.lsa import LSARepresentation
from dive.nlp.features.ngrams import (
    CharNGramRepresentation,
    WordCharUnionRepresentation,
)
from dive.nlp.features.tfidf import CountRepresentation, TFIDFRepresentation


def build_representation(
    config: Optional[NLPRepresentationConfig] = None,
    representation_type: str = "tfidf",
    **kwargs: Any,
) -> Union[
    TFIDFRepresentation,
    CharNGramRepresentation,
    WordCharUnionRepresentation,
    BM25Representation,
    CountRepresentation,
    EmbeddingRepresentation,
    LSARepresentation,
]:
    """Factory creating configured text feature representation."""
    if isinstance(config, str):
        rep_type = config
    elif config is not None:
        rep_type = getattr(config, "representation_type", representation_type)
    else:
        rep_type = representation_type


    if rep_type in ("char_ngrams", "char_ngram", "char"):
        return CharNGramRepresentation(**kwargs)
    elif rep_type in ("word_char_union", "union", "hybrid"):
        return WordCharUnionRepresentation(**kwargs)
    elif rep_type in ("bm25", "okapi_bm25"):
        return BM25Representation(**kwargs)
    elif rep_type in ("lsa", "semantic", "latent_semantic", "svd"):
        return LSARepresentation(**kwargs)
    elif rep_type in ("embedding", "embeddings", "dense", "sentence_transformers"):
        model_name = config.embedding_model if config and config.embedding_model else kwargs.get("model_name", "all-MiniLM-L6-v2")
        return EmbeddingRepresentation(model_name=model_name)
    elif rep_type in ("count", "bow"):
        return CountRepresentation(**kwargs)
    return TFIDFRepresentation(config=config, **kwargs)



__all__ = [
    "TFIDFRepresentation",
    "CharNGramRepresentation",
    "WordCharUnionRepresentation",
    "BM25Representation",
    "CountRepresentation",
    "EmbeddingRepresentation",
    "LSARepresentation",
    "build_representation",
]
