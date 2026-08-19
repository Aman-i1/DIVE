"""Declarative Configuration Engine for DIVE NLP - `dive/nlp/config.py`.

Provides validated configuration dataclasses for NLP tasks, resource budgets,
validation strategies, text preprocessing, and representations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None  # Optional fallback if PyYAML is absent


class NLPTaskType(str, Enum):
    """Supported and planned NLP task types."""

    TEXT_CLASSIFICATION = "text_classification"
    TEXT_REGRESSION = "text_regression"
    EMBEDDINGS = "embeddings"
    SEQUENCE_LABELING = "sequence_labeling"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    DOCUMENT_UNDERSTANDING = "document_understanding"

    @classmethod
    def from_str(cls, value: str) -> NLPTaskType:
        normalized = value.strip().lower().replace("-", "_")
        for task in cls:
            if task.value == normalized or task.name.lower() == normalized:
                return task
        # Accept common aliases
        aliases = {
            "classification": cls.TEXT_CLASSIFICATION,
            "regression": cls.TEXT_REGRESSION,
            "embedding": cls.EMBEDDINGS,
            "ner": cls.SEQUENCE_LABELING,
            "similarity": cls.SEMANTIC_SIMILARITY,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unsupported NLP task type: '{value}'. Valid: {[t.value for t in cls]}")


@dataclass
class NLPResourceConfig:
    """Hardware and computational limits for NLP workloads."""

    time_budget_secs: float = 600.0
    memory_budget_mb: float = 8192.0
    n_threads: Optional[int] = None
    use_gpu: bool = False
    batch_size: int = 64

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_budget_secs": self.time_budget_secs,
            "memory_budget_mb": self.memory_budget_mb,
            "n_threads": self.n_threads,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
        }


@dataclass
class NLPValidationConfig:
    """Validation and cross-validation configuration for NLP."""

    strategy: str = "stratified_kfold"  # stratified_kfold, kfold, train_test_split
    cv_splits: int = 5
    test_size: float = 0.2
    stratify: bool = True
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "cv_splits": self.cv_splits,
            "test_size": self.test_size,
            "stratify": self.stratify,
            "random_state": self.random_state,
        }


@dataclass
class NLPPreprocessingConfig:
    """Text normalization and preprocessing rules."""

    lowercase: bool = True
    strip_accents: Optional[str] = "unicode"
    max_seq_length: Optional[int] = None
    remove_urls: bool = False
    remove_html: bool = False
    remove_emojis: bool = False
    custom_stopwords: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "strip_accents": self.strip_accents,
            "max_seq_length": self.max_seq_length,
            "remove_urls": self.remove_urls,
            "remove_html": self.remove_html,
            "remove_emojis": self.remove_emojis,
            "custom_stopwords": self.custom_stopwords,
        }


@dataclass
class NLPRepresentationConfig:
    """Feature representation and embedding extraction settings."""

    representation_type: str = "tfidf"  # tfidf, count, embedding, none
    ngram_range: Tuple[int, int] = (1, 2)
    max_features: Optional[int] = 10000
    min_df: Union[int, float] = 2
    max_df: Union[int, float] = 0.95
    sublinear_tf: bool = True
    embedding_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "representation_type": self.representation_type,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "embedding_model": self.embedding_model,
        }


@dataclass
class NLPConfig:
    """Complete declarative configuration schema for DIVE NLP experiments."""

    task: str = NLPTaskType.TEXT_CLASSIFICATION.value
    text_column: str = "text"
    target_column: Optional[str] = "label"
    mode: str = "balanced"  # fast, balanced, quality
    output_dir: str = "./dive_nlp_output"
    resources: NLPResourceConfig = field(default_factory=NLPResourceConfig)
    validation: NLPValidationConfig = field(default_factory=NLPValidationConfig)
    preprocessing: NLPPreprocessingConfig = field(default_factory=NLPPreprocessingConfig)
    representation: NLPRepresentationConfig = field(default_factory=NLPRepresentationConfig)
    random_seed: int = 42

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NLPConfig":
        """Construct validated NLPConfig from dictionary."""
        res_data = data.get("resources", {})
        val_data = data.get("validation", {})
        prep_data = data.get("preprocessing", {})
        rep_data = data.get("representation", {})

        ngram = rep_data.get("ngram_range", (1, 2))
        if isinstance(ngram, list):
            ngram = tuple(ngram)

        return cls(
            task=data.get("task", NLPTaskType.TEXT_CLASSIFICATION.value),
            text_column=data.get("text_column", "text"),
            target_column=data.get("target_column", "label"),
            mode=data.get("mode", "balanced"),
            output_dir=data.get("output_dir", "./dive_nlp_output"),
            resources=NLPResourceConfig(
                time_budget_secs=res_data.get("time_budget_secs", 600.0),
                memory_budget_mb=res_data.get("memory_budget_mb", 8192.0),
                n_threads=res_data.get("n_threads"),
                use_gpu=res_data.get("use_gpu", False),
                batch_size=res_data.get("batch_size", 64),
            ),
            validation=NLPValidationConfig(
                strategy=val_data.get("strategy", "stratified_kfold"),
                cv_splits=val_data.get("cv_splits", 5),
                test_size=val_data.get("test_size", 0.2),
                stratify=val_data.get("stratify", True),
                random_state=val_data.get("random_state", 42),
            ),
            preprocessing=NLPPreprocessingConfig(
                lowercase=prep_data.get("lowercase", True),
                strip_accents=prep_data.get("strip_accents", "unicode"),
                max_seq_length=prep_data.get("max_seq_length"),
                remove_urls=prep_data.get("remove_urls", False),
                remove_html=prep_data.get("remove_html", False),
                remove_emojis=prep_data.get("remove_emojis", False),
                custom_stopwords=prep_data.get("custom_stopwords"),
            ),
            representation=NLPRepresentationConfig(
                representation_type=rep_data.get("representation_type", "tfidf"),
                ngram_range=ngram,
                max_features=rep_data.get("max_features", 10000),
                min_df=rep_data.get("min_df", 2),
                max_df=rep_data.get("max_df", 0.95),
                sublinear_tf=rep_data.get("sublinear_tf", True),
                embedding_model=rep_data.get("embedding_model"),
            ),
            random_seed=data.get("random_seed", 42),
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "NLPConfig":
        """Load declarative NLP configuration from YAML or JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"NLP configuration file '{path}' does not exist.")

        content = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            if yaml is not None:
                data = yaml.safe_load(content) or {}
            else:
                data = {}
                for line in content.splitlines():
                    if ":" in line and not line.strip().startswith("#"):
                        k, v = line.split(":", 1)
                        data[k.strip()] = v.strip()
        else:
            data = json.loads(content)

        return cls.from_dict(data)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save NLP configuration to JSON or YAML file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml") and yaml is not None:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "text_column": self.text_column,
            "target_column": self.target_column,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "resources": self.resources.to_dict(),
            "validation": self.validation.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "representation": self.representation.to_dict(),
            "random_seed": self.random_seed,
        }
