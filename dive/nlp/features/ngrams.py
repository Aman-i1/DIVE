"""Character N-Grams and Multi-Channel Feature Unions - `dive/nlp/features/ngrams.py`.

Provides character n-gram and joint word+character representation engines
for typo-tolerant, subword-aware, and morphologically rich feature extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from dive.nlp.config import NLPRepresentationConfig
from dive.nlp.features.tfidf import TFIDFRepresentation
from dive.nlp.interfaces import NLPRepresentationProtocol


class CharNGramRepresentation:
    """Character n-gram feature representation conforming to NLPRepresentationProtocol."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: Optional[int] = 20000,
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.95,
        sublinear_tf: bool = True,
        analyzer: str = "char_wb",  # char_wb respects word boundaries; or 'char'
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.analyzer = analyzer

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            analyzer=self.analyzer,
        )
        self.fitted_ = False

    def fit(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> "CharNGramRepresentation":
        """Fit character n-gram vocabulary and IDF weights."""
        if len(texts) <= 50:
            self.vectorizer.min_df = 1
        elif isinstance(self.min_df, int) and self.min_df > len(texts):
            self.vectorizer.min_df = 1

        self.vectorizer.fit(texts)
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Transform documents to character n-gram TF-IDF sparse matrix."""
        if not self.fitted_:
            raise RuntimeError("CharNGramRepresentation has not been fitted. Call .fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> sp.csr_matrix:
        """Fit and transform in a single call."""
        return self.fit(texts, y).transform(texts)

    @property
    def feature_names_(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    @property
    def n_features_(self) -> int:
        return len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") else 0


class WordCharUnionRepresentation:
    """Joint word and character multi-channel feature representation.

    Concatenates sparse word-level TF-IDF and character-level n-gram matrices
    to simultaneously capture semantic word tokens and subword morphological structure.
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: Optional[int] = 10000,
        char_ngram_range: Tuple[int, int] = (3, 5),
        char_max_features: Optional[int] = 15000,
        sublinear_tf: bool = True,
    ) -> None:
        self.word_rep = TFIDFRepresentation(
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            sublinear_tf=sublinear_tf,
        )
        self.char_rep = CharNGramRepresentation(
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            sublinear_tf=sublinear_tf,
            analyzer="char_wb",
        )
        self.fitted_ = False

    def fit(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> "WordCharUnionRepresentation":
        """Fit both word and character extractors on training texts."""
        self.word_rep.fit(texts, y)
        self.char_rep.fit(texts, y)
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Transform texts and horizontally stack sparse word and character feature matrices."""
        if not self.fitted_:
            raise RuntimeError("WordCharUnionRepresentation has not been fitted. Call .fit() first.")
        X_word = self.word_rep.transform(texts)
        X_char = self.char_rep.transform(texts)
        return sp.hstack([X_word, X_char], format="csr")

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> sp.csr_matrix:
        return self.fit(texts, y).transform(texts)

    @property
    def n_features_(self) -> int:
        return self.word_rep.n_features_ + self.char_rep.n_features_

    @property
    def feature_names_(self) -> List[str]:
        w_names = [f"word__{n}" for n in self.word_rep.feature_names_]
        c_names = [f"char__{n}" for n in self.char_rep.feature_names_]
        return w_names + c_names
