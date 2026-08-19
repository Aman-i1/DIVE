"""Modular NLP Text Preprocessor - `dive/nlp/preprocessing/preprocessor.py`.

Implements NLPPreprocessorProtocol with configurable text normalization,
custom stop-word filtering, and transformer-friendly raw passthroughs.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set

from dive.nlp.config import NLPPreprocessingConfig
from dive.nlp.interfaces import NLPPreprocessorProtocol
from dive.nlp.preprocessing.normalizer import TextNormalizer


class NLPPreprocessor:
    """Configurable, non-destructive text preprocessor pipeline."""

    def __init__(
        self,
        config: Optional[NLPPreprocessingConfig] = None,
        lowercase: bool = True,
        strip_accents: Optional[str] = "unicode",
        collapse_whitespace: bool = True,
        remove_html: bool = False,
        remove_urls: bool = False,
        remove_emojis: bool = False,
        remove_punctuation: bool = False,
        max_seq_length: Optional[int] = None,
        custom_stopwords: Optional[Sequence[str]] = None,
    ) -> None:
        if config is not None:
            self.normalizer = TextNormalizer(
                lowercase=config.lowercase,
                strip_accents=config.strip_accents,
                collapse_whitespace=True,
                remove_html=config.remove_html,
                remove_urls=config.remove_urls,
                remove_emojis=config.remove_emojis,
                max_word_length=config.max_seq_length,
            )
            self.custom_stopwords: Optional[Set[str]] = (
                set(config.custom_stopwords) if config.custom_stopwords else None
            )
            self.max_seq_length = config.max_seq_length
        else:
            self.normalizer = TextNormalizer(
                lowercase=lowercase,
                strip_accents=strip_accents,
                collapse_whitespace=collapse_whitespace,
                remove_html=remove_html,
                remove_urls=remove_urls,
                remove_emojis=remove_emojis,
                remove_punctuation=remove_punctuation,
                max_word_length=max_seq_length,
            )
            self.custom_stopwords = (
                set(custom_stopwords) if custom_stopwords else None
            )
            self.max_seq_length = max_seq_length

        self.fitted_ = False

    @classmethod
    def raw(cls) -> "NLPPreprocessor":
        """Return a clean passthrough preprocessor that preserves raw text untouched."""
        return cls(
            lowercase=False,
            strip_accents=None,
            collapse_whitespace=False,
            remove_html=False,
            remove_urls=False,
            remove_emojis=False,
            remove_punctuation=False,
            max_seq_length=None,
            custom_stopwords=None,
        )

    def fit(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> "NLPPreprocessor":
        """Fit preprocessor state (stateless by default, ready for vocabulary tracking)."""
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> List[str]:
        """Transform input sequence of texts according to configured rules."""
        cleaned = self.normalizer.transform(texts)

        # Optional stopword removal if explicitly specified
        if self.custom_stopwords:
            stopwords = self.custom_stopwords
            result = []
            for doc in cleaned:
                words = [w for w in doc.split() if w.lower() not in stopwords]
                result.append(" ".join(words))
            return result

        return cleaned

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> List[str]:
        """Fit and transform texts in a single call."""
        return self.fit(texts, y).transform(texts)
