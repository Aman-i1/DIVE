"""BM25 (Best Matching 25) Sparse Feature Representation - `dive/nlp/features/bm25.py`.

Implements probabilistic Okapi BM25 document-term sparse matrix representation
with configurable k1 (term saturation) and b (length normalization) parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from dive.nlp.interfaces import NLPRepresentationProtocol


class BM25Representation:
    """Okapi BM25 sparse representation engine conforming to NLPRepresentationProtocol."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = 10000,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.idf_: Optional[np.ndarray] = None
        self.avg_doc_len_: float = 0.0
        self.fitted_ = False

    def fit(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> "BM25Representation":
        """Fit vocabulary, calculate document frequencies, and compute BM25 IDF weights."""
        if len(texts) <= 50:
            self.vectorizer.min_df = 1
        elif isinstance(self.min_df, int) and self.min_df > len(texts):
            self.vectorizer.min_df = 1

        # Fit count matrix
        X_counts = self.vectorizer.fit_transform(texts)
        n_samples, n_features = X_counts.shape

        # Compute document lengths & average document length
        doc_lengths = np.asarray(X_counts.sum(axis=1)).flatten()
        self.avg_doc_len_ = float(np.mean(doc_lengths)) if len(doc_lengths) > 0 else 1.0
        if self.avg_doc_len_ == 0.0:
            self.avg_doc_len_ = 1.0

        # Calculate document frequency n(w) for each term
        # Binary count per document
        n_docs_with_term = np.asarray((X_counts > 0).sum(axis=0)).flatten()

        # Probabilistic BM25 IDF formula with smoothing: ln(1 + (N - n(w) + 0.5) / (n(w) + 0.5))
        idf = np.log(1.0 + (n_samples - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5))
        self.idf_ = idf.astype(np.float32)
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Transform input texts into BM25 weighted sparse matrix."""
        if not self.fitted_ or self.idf_ is None:
            raise RuntimeError("BM25Representation has not been fitted. Call .fit() first.")

        X_counts = self.vectorizer.transform(texts).astype(np.float32)
        if X_counts.nnz == 0:
            return X_counts.tocsr()

        doc_lengths = np.asarray(X_counts.sum(axis=1)).flatten()

        # Compute length normalization divisor for each document:
        # L_d = 1 - b + b * (doc_len / avg_doc_len)
        denom_len = 1.0 - self.b + self.b * (doc_lengths / self.avg_doc_len_)

        # Construct BM25 sparse matrix efficiently on CSR data
        X_csr = X_counts.tocsr()
        rows, cols = X_csr.nonzero()
        data = X_csr.data

        # doc index is rows, term index is cols
        doc_factor = denom_len[rows]
        tf = data

        # BM25 numerator: tf * (k1 + 1)
        num = tf * (self.k1 + 1.0)
        # BM25 denominator: tf + k1 * doc_factor
        denom = tf + self.k1 * doc_factor
        # term weight: (num / denom) * idf
        bm25_weights = (num / denom) * self.idf_[cols]

        return sp.csr_matrix((bm25_weights, (rows, cols)), shape=X_counts.shape)

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Sequence[Any]] = None
    ) -> sp.csr_matrix:
        return self.fit(texts, y).transform(texts)

    @property
    def feature_names_(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    @property
    def n_features_(self) -> int:
        return len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") else 0
