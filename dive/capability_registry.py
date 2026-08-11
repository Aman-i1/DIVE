"""Model Capability Registry & Selection Engine - `dive/capability_registry.py`.

Defines explicit ModelCapability metadata for every model in DIVE's ecosystem,
enabling dataset-aware and hardware-aware candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelCapability:
    """Capability metadata for an estimator."""

    model_name: str
    task_types: List[str]  # ["classification", "regression"]
    handles_missing: bool = False
    handles_categorical: bool = False
    supports_gpu: bool = False
    supports_sparse: bool = False
    max_reasonable_rows: int = 1_000_000
    latency_profile: str = "fast"  # fast, medium, slow
    memory_profile: str = "low"  # low, medium, high
    is_installed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_types": self.task_types,
            "handles_missing": self.handles_missing,
            "handles_categorical": self.handles_categorical,
            "supports_gpu": self.supports_gpu,
            "supports_sparse": self.supports_sparse,
            "max_reasonable_rows": self.max_reasonable_rows,
            "latency_profile": self.latency_profile,
            "memory_profile": self.memory_profile,
            "is_installed": self.is_installed,
        }


class CapabilityRegistry:
    """Central registry of model capabilities and matching selector."""

    def __init__(self) -> None:
        self._capabilities: Dict[str, ModelCapability] = {}
        self._register_defaults()

    def register(self, cap: ModelCapability) -> None:
        self._capabilities[cap.model_name] = cap

    def get(self, model_name: str) -> Optional[ModelCapability]:
        return self._capabilities.get(model_name)

    def _register_defaults(self) -> None:
        from dive.utils.optional import is_available

        # Linear / Logistic Baselines
        self.register(ModelCapability("LinearRegression", ["regression"], supports_sparse=True, max_reasonable_rows=10_000_000, latency_profile="fast", memory_profile="low"))
        self.register(ModelCapability("LogisticRegression", ["classification"], supports_sparse=True, max_reasonable_rows=5_000_000, latency_profile="fast", memory_profile="low"))
        self.register(ModelCapability("Ridge", ["regression", "classification"], supports_sparse=True, max_reasonable_rows=10_000_000, latency_profile="fast", memory_profile="low"))
        self.register(ModelCapability("Lasso", ["regression"], supports_sparse=True, max_reasonable_rows=5_000_000, latency_profile="fast", memory_profile="low"))
        self.register(ModelCapability("ElasticNet", ["regression"], supports_sparse=True, max_reasonable_rows=5_000_000, latency_profile="fast", memory_profile="low"))

        # Tree Ensembles
        self.register(ModelCapability("RandomForest", ["classification", "regression"], max_reasonable_rows=500_000, latency_profile="medium", memory_profile="medium"))
        self.register(ModelCapability("ExtraTrees", ["classification", "regression"], max_reasonable_rows=500_000, latency_profile="medium", memory_profile="medium"))
        self.register(ModelCapability("HistGradientBoosting", ["classification", "regression"], handles_missing=True, max_reasonable_rows=2_000_000, latency_profile="fast", memory_profile="medium"))

        # Gradient Boosted Trees (Optional Heavyweights)
        self.register(ModelCapability("LightGBM", ["classification", "regression"], handles_missing=True, handles_categorical=True, supports_gpu=True, max_reasonable_rows=20_000_000, latency_profile="fast", memory_profile="medium", is_installed=is_available("lightgbm")))
        self.register(ModelCapability("XGBoost", ["classification", "regression"], handles_missing=True, supports_gpu=True, max_reasonable_rows=20_000_000, latency_profile="fast", memory_profile="medium", is_installed=is_available("xgboost")))
        self.register(ModelCapability("CatBoost", ["classification", "regression"], handles_missing=True, handles_categorical=True, supports_gpu=True, max_reasonable_rows=10_000_000, latency_profile="medium", memory_profile="medium", is_installed=is_available("catboost")))

    def recommend(
        self,
        problem_type: str,
        n_samples: int,
        n_features: int,
        has_missing: bool = False,
        has_categorical: bool = False,
    ) -> Dict[str, List[str]]:
        """Recommend model candidates matching dataset specs."""
        recommended: List[str] = []
        acceptable: List[str] = []
        rejected: List[str] = []

        for name, cap in self._capabilities.items():
            if not cap.is_installed:
                rejected.append(name)
                continue
            if problem_type not in cap.task_types:
                rejected.append(name)
                continue
            if n_samples > cap.max_reasonable_rows and cap.latency_profile == "slow":
                rejected.append(name)
                continue

            # Prioritize GBDTs & HistGradientBoosting for larger or complex data
            if name in ("LightGBM", "CatBoost", "XGBoost", "HistGradientBoosting"):
                recommended.append(name)
            elif name in ("RandomForest", "ExtraTrees", "Ridge", "LogisticRegression", "LinearRegression"):
                acceptable.append(name)
            else:
                acceptable.append(name)

        return {
            "recommended": recommended,
            "acceptable": acceptable,
            "rejected": rejected,
        }
