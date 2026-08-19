"""Standalone NLP Predictor & Inference Engine - `dive/nlp/inference/predictor.py`.

Provides self-describing, deployable NLP predictors that accept raw strings,
lists of texts, pandas DataFrames, or dictionary records and produce predictions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dive.nlp.exceptions import NLPInferenceError
from dive.nlp.interfaces import NLPPredictorProtocol
from dive.nlp.pipeline import NLPPipeline
from dive.utils.io import load_pickle, save_pickle


class NLPPredictor:
    """Self-contained deployable NLP model artifact implementing NLPPredictorProtocol."""

    def __init__(
        self,
        pipeline: NLPPipeline,
        model_name: str = "NLPBaseline",
        text_column: str = "text",
        target_column: Optional[str] = "label",
        task_type: str = "text_classification",
        metrics: Optional[Dict[str, Any]] = None,
        dataset_name: str = "unknown",
        trained_at: Optional[str] = None,
    ) -> None:
        self.pipeline = pipeline
        self.model_name = model_name
        self.text_column = text_column
        self.target_column = target_column
        self.task_type = task_type
        self.metrics = metrics or {}
        self.dataset_name = dataset_name
        self.trained_at = trained_at or datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Input Coercion
    # ------------------------------------------------------------------
    def _coerce_texts(
        self, data: Union[str, Sequence[str], pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> List[str]:
        """Convert arbitrary incoming input into a uniform List[str]."""
        if isinstance(data, str):
            return [data]

        if isinstance(data, pd.DataFrame):
            if self.text_column in data.columns:
                return data[self.text_column].astype(str).tolist()
            # If specified text_column is absent but single column dataframe
            if len(data.columns) == 1:
                return data.iloc[:, 0].astype(str).tolist()
            raise NLPInferenceError(
                f"DataFrame missing expected text column '{self.text_column}'.",
                f"Available columns: {', '.join(map(str, data.columns[:10]))}",
            )

        if isinstance(data, Mapping):
            if self.text_column in data:
                return [str(data[self.text_column])]
            if len(data) == 1:
                return [str(next(iter(data.values())))]
            raise NLPInferenceError(f"Dictionary record missing key '{self.text_column}'.")

        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            items = list(data)
            if not items:
                return []
            if isinstance(items[0], str):
                return [str(s) for s in items]
            if isinstance(items[0], Mapping):
                res = []
                for row in items:
                    if self.text_column in row:
                        res.append(str(row[self.text_column]))
                    elif len(row) == 1:
                        res.append(str(next(iter(row.values()))))
                    else:
                        raise NLPInferenceError(f"Sequence row missing key '{self.text_column}'.")
                return res

        raise NLPInferenceError(
            f"Unsupported input type for NLP prediction: {type(data).__name__}. "
            "Expected str, list of str, DataFrame, or record dicts."
        )

    # ------------------------------------------------------------------
    # Prediction Methods
    # ------------------------------------------------------------------
    def predict(
        self, data: Union[str, Sequence[str], pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> np.ndarray:
        """Generate predictions for raw text input."""
        texts = self._coerce_texts(data)
        if not texts:
            return np.array([])
        return self.pipeline.predict(texts)

    def predict_proba(
        self, data: Union[str, Sequence[str], pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> np.ndarray:
        """Generate class probability distributions for raw text input."""
        texts = self._coerce_texts(data)
        if not texts:
            return np.empty((0, len(self.class_names or [])))
        return self.pipeline.predict_proba(texts)

    @property
    def class_names(self) -> Optional[List[str]]:
        """List of class names for classification models."""
        return self.pipeline.class_names

    @property
    def has_proba(self) -> bool:
        """True if the model can output probability distributions."""
        return self.task_type in ("text_classification", "classification") and hasattr(
            self.pipeline.estimator, "predict_proba"
        )

    def describe_input(self) -> Dict[str, Any]:
        """Describe expected input schema for API documentation and clients."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "text_column": self.text_column,
            "target_column": self.target_column,
            "has_probabilities": self.has_proba,
            "class_names": self.class_names,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }

    def save(self, file_path: Union[str, Path]) -> Path:
        """Save the predictor to a portable pickle file."""
        return save_nlp_predictor(self, file_path)


def save_nlp_predictor(predictor: NLPPredictor, file_path: Union[str, Path]) -> Path:
    """Save an NLPPredictor artifact to disk."""
    path = Path(file_path)
    save_pickle(predictor, path)
    return path


def load_nlp_predictor(file_path: Union[str, Path]) -> NLPPredictor:
    """Load an NLPPredictor artifact from disk."""
    obj = load_pickle(file_path)
    if not isinstance(obj, NLPPredictor):
        raise TypeError(f"Expected NLPPredictor instance, but loaded {type(obj).__name__}.")
    return obj
