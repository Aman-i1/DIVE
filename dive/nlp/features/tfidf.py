"""TF-IDF and Count Feature Representations - `dive/nlp/features/tfidf.py`.

Provides sparse bag-of-words and TF-IDF n-gram feature extraction pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from dive.nlp.config import NLPRepresentationConfig
from dive.nlp.interfaces import NLPRepresentationProtocol


class TFIDFRepresentation:
    """TF-IDF sparse feature extractor conforming to NLPRepresentationProtocol."""

    def __init__(
        self,
        config: Optional[NLPRepresentationConfig] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 10000,
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.95,
        sublinear_tf: bool = True,
        use_idf: bool = True,
        norm: Optional[str] = "l2",
    ) -> None:
        if config is not None:
            self.ngram_range = config.ngram_range
            self.max_features = config.max_features
            self.min_df = config.min_df
            self.max_df = config.max_df
            self.sublinear_tf = config.sublinear_tf
            self.use_idf = True
            self.norm = "l2"
        else:
            self.ngram_range = ngram_range
            self.max_features = max_features
            self.min_df = min_df
            self.max_df = max_df
            self.sublinear_tf = sublinear_tf
            self.use_idf = use_idf
            self.norm = norm

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            use_idf=self.use_idf,
            norm=self.norm,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.fitted_ = False

    def fit(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> "TFIDFRepresentation":
        """Fit vocabulary and IDF weights on a collection of documents."""
        # Adjust min_df if dataset is small to prevent vocabulary pruning
        if len(texts) <= 50:
            self.vectorizer.min_df = 1
        elif isinstance(self.min_df, int) and self.min_df > len(texts):
            self.vectorizer.min_df = 1

        self.vectorizer.fit(texts)
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Transform documents to document-term TF-IDF matrix."""
        if not self.fitted_:
            raise RuntimeError("TFIDFRepresentation has not been fitted yet. Call .fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> sp.csr_matrix:
        """Fit to data, then transform it."""
        return self.fit(texts, y).transform(texts)

    @property
    def vocabulary_(self) -> Dict[str, int]:
        return self.vectorizer.vocabulary_

    @property
    def feature_names_(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    @property
    def n_features_(self) -> int:
        return len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") else 0


class CountRepresentation:
    """Bag-of-Words count representation conforming to NLPRepresentationProtocol."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = 10000,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        binary: bool = False,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary

        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            binary=self.binary,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.fitted_ = False

    def fit(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> "CountRepresentation":
        if isinstance(self.min_df, int) and self.min_df > len(texts):
            self.vectorizer.min_df = 1
        self.vectorizer.fit(texts)
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        if not self.fitted_:
            raise RuntimeError("CountRepresentation has not been fitted yet. Call .fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> sp.csr_matrix:
        return self.fit(texts, y).transform(texts)

    @property
    def feature_names_(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()
