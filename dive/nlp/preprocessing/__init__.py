"""DIVE NLP Preprocessing Pipeline - `dive/nlp/preprocessing`.

Provides modular text normalization, cleaning, and configurable preprocessing pipelines.
"""

from __future__ import annotations

from typing import Optional, Sequence

from dive.nlp.config import NLPPreprocessingConfig
from dive.nlp.preprocessing.normalizer import TextNormalizer
from dive.nlp.preprocessing.preprocessor import NLPPreprocessor


def build_nlp_preprocessor(
    config: Optional[NLPPreprocessingConfig] = None,
    lowercase: bool = True,
    remove_html: bool = False,
    remove_urls: bool = False,
    max_seq_length: Optional[int] = None,
) -> NLPPreprocessor:
    """Build and configure an NLPPreprocessor."""
    if config is not None:
        return NLPPreprocessor(config=config)
    return NLPPreprocessor(
        lowercase=lowercase,
        remove_html=remove_html,
        remove_urls=remove_urls,
        max_seq_length=max_seq_length,
    )


__all__ = [
    "TextNormalizer",
    "NLPPreprocessor",
    "build_nlp_preprocessor",
]
