"""NLP Model and Component Registry - `dive/nlp/registry.py`.

Defines capability metadata and dynamic candidate matching for NLP models and representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NLPModelCapability:
    """Capability metadata for an NLP estimator or representation."""

    model_name: str
    task_types: List[str]  # ["text_classification", "text_regression", "embeddings", etc.]
    supports_gpu: bool = False
    supports_fine_tuning: bool = False
    latency_profile: str = "fast"  # fast, medium, slow
    memory_profile: str = "low"  # low, medium, high
    min_samples_recommended: int = 10
    max_samples_recommended: int = 1_000_000
    is_installed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_types": self.task_types,
            "supports_gpu": self.supports_gpu,
            "supports_fine_tuning": self.supports_fine_tuning,
            "latency_profile": self.latency_profile,
            "memory_profile": self.memory_profile,
            "min_samples_recommended": self.min_samples_recommended,
            "max_samples_recommended": self.max_samples_recommended,
            "is_installed": self.is_installed,
        }


class NLPRegistry:
    """Central registry of NLP capabilities and candidate selector."""

    def __init__(self) -> None:
        self._capabilities: Dict[str, NLPModelCapability] = {}
        self._register_defaults()

    def register(self, cap: NLPModelCapability) -> None:
        """Register capability metadata for a model or representation."""
        self._capabilities[cap.model_name] = cap

    def get(self, model_name: str) -> Optional[NLPModelCapability]:
        """Retrieve capability metadata by model name."""
        return self._capabilities.get(model_name)

    def list_models(self, task_type: Optional[str] = None) -> List[NLPModelCapability]:
        """List registered models, optionally filtered by task type."""
        models = list(self._capabilities.values())
        if task_type:
            models = [m for m in models if task_type in m.task_types]
        return models

    def _register_defaults(self) -> None:
        """Register baseline CPU and optional deep learning capabilities."""
        from dive.utils.optional import is_available

        # Classical CPU Baselines (Scikit-Learn based)
        self.register(
            NLPModelCapability(
                model_name="LogisticRegression",
                task_types=["text_classification"],
                latency_profile="fast",
                memory_profile="low",
                max_samples_recommended=5_000_000,
                is_installed=True,
            )
        )
        self.register(
            NLPModelCapability(
                model_name="LinearSVC",
                task_types=["text_classification"],
                latency_profile="fast",
                memory_profile="low",
                max_samples_recommended=5_000_000,
                is_installed=True,
            )
        )
        self.register(
            NLPModelCapability(
                model_name="MultinomialNB",
                task_types=["text_classification"],
                latency_profile="fast",
                memory_profile="low",
                max_samples_recommended=5_000_000,
                is_installed=True,
            )
        )
        self.register(
            NLPModelCapability(
                model_name="RidgeRegression",
                task_types=["text_regression"],
                latency_profile="fast",
                memory_profile="low",
                max_samples_recommended=5_000_000,
                is_installed=True,
            )
        )

        # Optional Deep Learning / Transformer Capabilities
        self.register(
            NLPModelCapability(
                model_name="SentenceTransformers",
                task_types=["embeddings", "text_classification", "semantic_similarity"],
                supports_gpu=True,
                latency_profile="medium",
                memory_profile="medium",
                is_installed=is_available("sentence_transformers"),
            )
        )
        for tf_name in ("distilbert", "bert", "roberta", "deberta"):
            self.register(
                NLPModelCapability(
                    model_name=tf_name,
                    task_types=["text_classification", "text_regression"],
                    supports_gpu=True,
                    supports_fine_tuning=True,
                    latency_profile="medium" if tf_name == "distilbert" else "slow",
                    memory_profile="medium" if tf_name == "distilbert" else "high",
                    is_installed=is_available("transformers") and is_available("torch"),
                )
            )
        self.register(
            NLPModelCapability(
                model_name="HuggingFaceTransformer",
                task_types=["text_classification", "text_regression", "sequence_labeling"],
                supports_gpu=True,
                supports_fine_tuning=True,
                latency_profile="slow",
                memory_profile="high",
                is_installed=is_available("transformers") and is_available("torch"),
            )
        )
