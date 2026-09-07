"""Deployable model artifact and inference engine for DIVE Deep Learning.

Encapsulates a fitted :class:`~dive.dl.modalities.base.ModalityAdapter` and a
trained :class:`~dive.dl.core.trainer.DLTrainer` into a unified, portable predictor
capable of end-to-end inference from raw modality inputs to predictions and probabilities.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dive.dl.config import DLConfig, DLTask, Modality
from dive.dl.core.trainer import DLTrainer
from dive.dl.exceptions import DLInferenceError
from dive.dl.modalities.base import ModalityAdapter
from dive.utils.io import load_pickle, save_pickle


class DLPredictor:
    """Self-contained, serialized deep learning model artifact for DIVE."""

    def __init__(
        self,
        adapter: ModalityAdapter,
        trainer: DLTrainer,
        config: Optional[DLConfig] = None,
        modality: Optional[Union[str, Modality]] = None,
        task: Optional[Union[str, DLTask]] = None,
        input_column: Optional[str] = None,
        target_column: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        trained_at: Optional[str] = None,
    ) -> None:
        self.adapter = adapter
        self.trainer = trainer
        self.config = config or DLConfig()
        self.modality = (
            modality.value
            if isinstance(modality, Modality)
            else (modality or getattr(adapter, "modality", Modality.TABULAR).value)
        )
        self.task = (
            task.value
            if isinstance(task, DLTask)
            else (task or getattr(trainer, "task", DLTask.CLASSIFICATION).value)
        )
        self.input_column = input_column
        self.target_column = target_column
        self.metrics = metrics or {}
        self.trained_at = trained_at or datetime.now().isoformat()

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Class labels for classification tasks."""
        return getattr(self.trainer, "classes_", None)

    @property
    def has_proba(self) -> bool:
        """True if the underlying model outputs class probabilities."""
        return self.task == DLTask.CLASSIFICATION.value and hasattr(self.trainer, "predict_proba")

    @property
    def backend(self) -> str:
        """Name of the training engine backend (torch-mlp or sklearn-mlp)."""
        return getattr(self.trainer, "backend", "unknown")

    def _prepare_features(self, inputs: Any) -> np.ndarray:
        """Transform raw incoming inputs into a dense feature matrix."""
        coerced = self.adapter.coerce(inputs)
        return self.adapter.transform(coerced)

    def predict(self, inputs: Any) -> np.ndarray:
        """Generate point predictions for raw modality input."""
        features = self._prepare_features(inputs)
        if len(features) == 0:
            return np.array([])
        return self.trainer.predict(features)

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Generate class probability distributions for raw modality input."""
        if not self.has_proba:
            raise DLInferenceError(
                f"Probability scoring is not supported for task '{self.task}' "
                f"with backend '{self.backend}'."
            )
        features = self._prepare_features(inputs)
        if len(features) == 0:
            n_classes = len(self.classes_) if self.classes_ is not None else 0
            return np.empty((0, n_classes))
        return self.trainer.predict_proba(features)

    def describe(self) -> Dict[str, Any]:
        """Structured summary of predictor metadata and capabilities."""
        return {
            "modality": self.modality,
            "task": self.task,
            "backend": self.backend,
            "has_proba": self.has_proba,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "input_column": self.input_column,
            "target_column": self.target_column,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "adapter": self.adapter.describe(),
        }

    def save(self, file_path: Union[str, Path]) -> Path:
        """Serialize the predictor to disk."""
        return save_dl_predictor(self, file_path)


def save_dl_predictor(predictor: DLPredictor, file_path: Union[str, Path]) -> Path:
    """Save a DLPredictor instance to a portable pickle file."""
    path = Path(file_path)
    save_pickle(predictor, path)
    return path


def load_dl_predictor(file_path: Union[str, Path]) -> DLPredictor:
    """Load a DLPredictor instance from disk."""
    obj = load_pickle(file_path)
    if not isinstance(obj, DLPredictor):
        raise TypeError(f"Expected DLPredictor instance, but loaded {type(obj).__name__}.")
    return obj
