"""Declarative Project Configuration Engine - `dive/config.py`.

Parses and validates declarative project configuration files (`dive.yaml` / `dive.json`)
defining targets, validation strategies, feature engineering rules, time/memory budgets,
and deployment targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None  # Optional fallback to JSON if PyYAML is not installed


@dataclass
class DeclarativeValidationConfig:
    strategy: Optional[str] = None  # e.g., 'StratifiedKFold', 'GroupKFold', 'TimeSeriesSplit'
    cv_splits: int = 5
    group_column: Optional[str] = None
    time_column: Optional[str] = None


@dataclass
class DeclarativeResourceConfig:
    time_budget_secs: float = 600.0
    memory_budget_mb: float = 8192.0
    n_threads: Optional[int] = None
    use_gpu: bool = False


@dataclass
class DeclarativeServingConfig:
    enable_rest_api: bool = True
    port: int = 8000
    export_onnx: bool = True
    batch_chunk_size: int = 10_000


@dataclass
class DiveConfig:
    """Complete declarative configuration schema for DIVE AutoML experiments."""

    target: str
    mode: str = "balanced"  # fast, balanced, thorough, competition
    output_dir: str = "./dive_output"
    validation: DeclarativeValidationConfig = field(default_factory=DeclarativeValidationConfig)
    resources: DeclarativeResourceConfig = field(default_factory=DeclarativeResourceConfig)
    serving: DeclarativeServingConfig = field(default_factory=DeclarativeServingConfig)
    random_seed: int = 42

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiveConfig":
        """Construct validated DiveConfig from dictionary."""
        val_data = data.get("validation", {})
        res_data = data.get("resources", {})
        srv_data = data.get("serving", {})

        return cls(
            target=data.get("target", ""),
            mode=data.get("mode", "balanced"),
            output_dir=data.get("output_dir", "./dive_output"),
            validation=DeclarativeValidationConfig(
                strategy=val_data.get("strategy"),
                cv_splits=val_data.get("cv_splits", 5),
                group_column=val_data.get("group_column"),
                time_column=val_data.get("time_column"),
            ),
            resources=DeclarativeResourceConfig(
                time_budget_secs=res_data.get("time_budget_secs", 600.0),
                memory_budget_mb=res_data.get("memory_budget_mb", 8192.0),
                n_threads=res_data.get("n_threads"),
                use_gpu=res_data.get("use_gpu", False),
            ),
            serving=DeclarativeServingConfig(
                enable_rest_api=srv_data.get("enable_rest_api", True),
                port=srv_data.get("port", 8000),
                export_onnx=srv_data.get("export_onnx", True),
                batch_chunk_size=srv_data.get("batch_chunk_size", 10_000),
            ),
            random_seed=data.get("random_seed", 42),
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "DiveConfig":
        """Load declarative configuration from YAML or JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file '{path}' does not exist.")

        content = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            if yaml is not None:
                data = yaml.safe_load(content) or {}
            else:
                # Basic line-by-line fallback parser for simple key-value YAML if PyYAML is missing
                data = {}
                for line in content.splitlines():
                    if ":" in line and not line.strip().startswith("#"):
                        k, v = line.split(":", 1)
                        data[k.strip()] = v.strip()
        else:
            data = json.loads(content)

        return cls.from_dict(data)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "validation": {
                "strategy": self.validation.strategy,
                "cv_splits": self.validation.cv_splits,
                "group_column": self.validation.group_column,
                "time_column": self.validation.time_column,
            },
            "resources": {
                "time_budget_secs": self.resources.time_budget_secs,
                "memory_budget_mb": self.resources.memory_budget_mb,
                "n_threads": self.resources.n_threads,
                "use_gpu": self.resources.use_gpu,
            },
            "serving": {
                "enable_rest_api": self.serving.enable_rest_api,
                "port": self.serving.port,
                "export_onnx": self.serving.export_onnx,
                "batch_chunk_size": self.serving.batch_chunk_size,
            },
            "random_seed": self.random_seed,
        }
