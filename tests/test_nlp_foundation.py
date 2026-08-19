"""Phase 1 Foundation Tests - DIVE NLP Domain & ML Isolation.

Verifies:
1. Clean separation between DIVE ML and DIVE NLP namespaces.
2. Full backward compatibility for existing top-level DIVE imports.
3. NLP-specific exception hierarchy derived from DiveError.
4. Declarative NLP configuration schemas, serialisation, and round-tripping.
5. Structural subtyping protocols and runtime checkable interfaces.
6. NLP model capability registry and candidate matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytest

from dive.exceptions import DiveError
from dive.nlp.config import (
    NLPConfig,
    NLPPreprocessingConfig,
    NLPRepresentationConfig,
    NLPResourceConfig,
    NLPTaskType,
    NLPValidationConfig,
)
from dive.nlp.exceptions import (
    NLPConfigError,
    NLPError,
    NLPInferenceError,
    NLPModelError,
    NLPTrainingError,
    TaskNotSupportedError,
    TextDataError,
    TokenizationError,
    VocabularyError,
)
from dive.nlp.interfaces import (
    NLPDatasetProtocol,
    NLPEstimatorProtocol,
    NLPPipelineProtocol,
    NLPPredictorProtocol,
    NLPPreprocessorProtocol,
    NLPProfilerProtocol,
    NLPRepresentationProtocol,
)
from dive.nlp.registry import NLPModelCapability, NLPRegistry


def test_import_namespaces_and_backward_compatibility() -> None:
    """Verify that dive, dive.ml, and dive.nlp provide isolated, compatible namespaces."""
    import dive
    from dive import ml, nlp

    # Backward compatibility: root imports
    assert hasattr(dive, "Dive")
    assert hasattr(dive, "create_study")
    assert hasattr(dive, "DivePredictor")
    assert hasattr(dive, "DiveDoctor")
    assert hasattr(dive, "quick_dive")

    # Domain namespaces
    assert dive.ml.Dive is dive.Dive
    assert dive.ml.create_study is dive.create_study
    assert dive.ml.DivePredictor is dive.DivePredictor

    # NLP Domain
    assert hasattr(nlp, "NLPConfig")
    assert hasattr(nlp, "NLPError")
    assert hasattr(nlp, "NLPRegistry")
    assert hasattr(nlp, "NLPDatasetProtocol")


def test_nlp_exceptions_hierarchy() -> None:
    """Verify all NLP exceptions inherit from DiveError and render hints."""
    # Inheritance check
    assert issubclass(NLPError, DiveError)
    assert issubclass(TextDataError, NLPError)
    assert issubclass(NLPConfigError, NLPError)
    assert issubclass(NLPModelError, NLPError)
    assert issubclass(NLPTrainingError, NLPError)
    assert issubclass(NLPInferenceError, NLPError)
    assert issubclass(TokenizationError, NLPError)
    assert issubclass(VocabularyError, NLPError)
    assert issubclass(TaskNotSupportedError, NLPError)

    # Message and hint rendering
    exc = TextDataError("Missing text column 'content'.", "Ensure the CSV contains a text column.")
    assert "Missing text column" in str(exc)
    assert "Ensure the CSV contains" in str(exc)
    assert exc.message == "Missing text column 'content'."
    assert exc.hint == "Ensure the CSV contains a text column."


def test_nlp_config_defaults_and_validation() -> None:
    """Verify default configurations and helper conversions."""
    cfg = NLPConfig()
    assert cfg.task == "text_classification"
    assert cfg.text_column == "text"
    assert cfg.target_column == "label"
    assert cfg.mode == "balanced"
    assert cfg.resources.time_budget_secs == 600.0
    assert cfg.resources.memory_budget_mb == 8192.0
    assert cfg.validation.strategy == "stratified_kfold"
    assert cfg.validation.cv_splits == 5
    assert cfg.preprocessing.lowercase is True
    assert cfg.representation.representation_type == "tfidf"
    assert cfg.representation.ngram_range == (1, 2)

    # Task type enum resolution
    assert NLPTaskType.from_str("text_classification") == NLPTaskType.TEXT_CLASSIFICATION
    assert NLPTaskType.from_str("classification") == NLPTaskType.TEXT_CLASSIFICATION
    assert NLPTaskType.from_str("regression") == NLPTaskType.TEXT_REGRESSION
    assert NLPTaskType.from_str("embedding") == NLPTaskType.EMBEDDINGS

    with pytest.raises(ValueError, match="Unsupported NLP task type"):
        NLPTaskType.from_str("unsupported_super_task")


def test_nlp_config_dict_and_file_roundtrip(tmp_path: Path) -> None:
    """Verify dictionary conversion and file saving/loading."""
    custom_cfg = NLPConfig(
        task="text_classification",
        text_column="review_body",
        target_column="sentiment",
        mode="quality",
        resources=NLPResourceConfig(time_budget_secs=1200.0, use_gpu=True, batch_size=128),
        validation=NLPValidationConfig(strategy="kfold", cv_splits=10, test_size=0.15),
        preprocessing=NLPPreprocessingConfig(lowercase=False, remove_urls=True),
        representation=NLPRepresentationConfig(representation_type="tfidf", max_features=5000, ngram_range=(1, 3)),
    )

    d = custom_cfg.to_dict()
    assert d["text_column"] == "review_body"
    assert d["resources"]["time_budget_secs"] == 1200.0
    assert d["resources"]["use_gpu"] is True
    assert d["validation"]["cv_splits"] == 10
    assert d["preprocessing"]["remove_urls"] is True
    assert d["representation"]["max_features"] == 5000

    # Reconstruct from dict
    reconstructed = NLPConfig.from_dict(d)
    assert reconstructed.text_column == "review_body"
    assert reconstructed.resources.time_budget_secs == 1200.0
    assert reconstructed.representation.ngram_range == (1, 3)

    # Save and load JSON
    json_path = tmp_path / "nlp_config.json"
    custom_cfg.save(json_path)
    loaded_json = NLPConfig.load(json_path)
    assert loaded_json.text_column == "review_body"
    assert loaded_json.resources.use_gpu is True

    # Save and load YAML
    yaml_path = tmp_path / "nlp_config.yaml"
    custom_cfg.save(yaml_path)
    loaded_yaml = NLPConfig.load(yaml_path)
    assert loaded_yaml.text_column == "review_body"
    assert loaded_yaml.preprocessing.remove_urls is True


def test_nlp_protocols_runtime_checkable() -> None:
    """Verify structural subtyping protocol compliance with runtime checkable interfaces."""

    class DummyPreprocessor:
        def fit(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> Any:
            return self

        def transform(self, texts: Sequence[str]) -> Sequence[str]:
            return [t.lower() for t in texts]

        def fit_transform(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> Sequence[str]:
            return self.transform(texts)

    prep = DummyPreprocessor()
    assert isinstance(prep, NLPPreprocessorProtocol)

    class DummyEstimator:
        def fit(self, X: Any, y: Sequence[Any]) -> Any:
            return self

        def predict(self, X: Any) -> np.ndarray:
            return np.array([0] * len(X))

        def predict_proba(self, X: Any) -> np.ndarray:
            return np.array([[0.5, 0.5]] * len(X))

    est = DummyEstimator()
    assert isinstance(est, NLPEstimatorProtocol)

    class DummyDataset:
        def __init__(self, texts: List[str], labels: Optional[List[Any]] = None) -> None:
            self._texts = texts
            self._labels = labels

        @property
        def texts(self) -> Sequence[str]:
            return self._texts

        @property
        def labels(self) -> Optional[Sequence[Any]]:
            return self._labels

        @property
        def sample_ids(self) -> Optional[Sequence[str]]:
            return None

        @property
        def metadata(self) -> Optional[pd.DataFrame]:
            return None

        def __len__(self) -> int:
            return len(self._texts)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            return {"text": self._texts[index], "label": self._labels[index] if self._labels else None}

    ds = DummyDataset(["hello world", "test nlp"], [0, 1])
    assert isinstance(ds, NLPDatasetProtocol)
    assert len(ds) == 2
    assert ds[0]["text"] == "hello world"


def test_nlp_registry_defaults_and_registration() -> None:
    """Verify NLP capability registry defaults and model filtering."""
    registry = NLPRegistry()

    # Verify standard defaults
    lr_cap = registry.get("LogisticRegression")
    assert lr_cap is not None
    assert "text_classification" in lr_cap.task_types
    assert lr_cap.latency_profile == "fast"
    assert lr_cap.is_installed is True

    # Filter by task
    clf_models = registry.list_models("text_classification")
    clf_names = [m.model_name for m in clf_models]
    assert "LogisticRegression" in clf_names
    assert "LinearSVC" in clf_names
    assert "MultinomialNB" in clf_names

    reg_models = registry.list_models("text_regression")
    reg_names = [m.model_name for m in reg_models]
    assert "RidgeRegression" in reg_names
    assert "LogisticRegression" not in reg_names

    # Custom registration
    custom_cap = NLPModelCapability(
        model_name="CustomFastText",
        task_types=["text_classification"],
        latency_profile="fast",
        memory_profile="low",
    )
    registry.register(custom_cap)
    assert registry.get("CustomFastText") is custom_cap
    assert custom_cap.to_dict()["model_name"] == "CustomFastText"
