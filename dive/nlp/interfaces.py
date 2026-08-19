"""Protocol and interface definitions for DIVE NLP - `dive/nlp/interfaces.py`.

Establishes structural subtyping protocols for datasets, preprocessors,
representations, estimators, pipelines, profilers, and predictors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class NLPDatasetProtocol(Protocol):
    """Structural protocol for NLP text datasets."""

    @property
    def texts(self) -> Sequence[str]:
        """Sequence of text documents."""
        ...

    @property
    def labels(self) -> Optional[Sequence[Any]]:
        """Optional target labels or values."""
        ...

    @property
    def sample_ids(self) -> Optional[Sequence[str]]:
        """Optional unique sample identifiers."""
        ...

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Optional tabular metadata dataframe."""
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ...


@runtime_checkable
class NLPPreprocessorProtocol(Protocol):
    """Structural protocol for text normalization and preprocessing."""

    def fit(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> NLPPreprocessorProtocol:
        ...

    def transform(self, texts: Sequence[str]) -> Sequence[str]:
        ...

    def fit_transform(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> Sequence[str]:
        ...


@runtime_checkable
class NLPRepresentationProtocol(Protocol):
    """Structural protocol for text feature representations (TF-IDF, Embeddings, etc.)."""

    def fit(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> NLPRepresentationProtocol:
        ...

    def transform(self, texts: Sequence[str]) -> Any:
        ...

    def fit_transform(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> Any:
        ...


@runtime_checkable
class NLPEstimatorProtocol(Protocol):
    """Structural protocol for NLP learning algorithms."""

    def fit(self, X: Any, y: Sequence[Any]) -> NLPEstimatorProtocol:
        ...

    def predict(self, X: Any) -> np.ndarray:
        ...


@runtime_checkable
class NLPPipelineProtocol(Protocol):
    """Structural protocol for end-to-end NLP pipelines."""

    def fit(self, texts: Sequence[str], y: Sequence[Any]) -> NLPPipelineProtocol:
        ...

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        ...


@runtime_checkable
class NLPProfilerProtocol(Protocol):
    """Structural protocol for NLP dataset profilers."""

    def profile(self, dataset: Any) -> Dict[str, Any]:
        ...


@runtime_checkable
class NLPPredictorProtocol(Protocol):
    """Structural protocol for self-contained, deployable NLP predictors."""

    def predict(self, data: Union[str, Sequence[str], pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        ...

    def describe_input(self) -> Dict[str, Any]:
        ...
