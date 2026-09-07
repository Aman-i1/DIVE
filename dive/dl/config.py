"""Declarative configuration for DIVE Deep Learning - ``dive/dl/config.py``.

Deliberately shaped like :mod:`dive.nlp.config` (nested dataclasses, ``from_dict``
/ ``load`` / ``save`` / ``to_dict``) so a user who has written a ``dive.yaml`` for
the NLP domain can read a DL one without relearning anything.

The one structural difference: NLP encodes the whole job in a single ``task``
string, which would need ten enum members here (five modalities x two objectives).
:class:`Modality` and :class:`DLTask` are kept orthogonal instead, so adding a
modality does not multiply the task vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dive.dl.exceptions import DLConfigError

try:
    import yaml
except ImportError:  # PyYAML is a hard dependency, but stay defensive
    yaml = None


class Modality(str, Enum):
    """The kind of data being learned from.

    ``dive dl`` covers every modality the user asked for: tabular rows, free
    text, still images, audio/voice, and video.
    """

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

    @classmethod
    def from_str(cls, value: Any) -> "Modality":
        if isinstance(value, cls):
            return value
        val_str = getattr(value, "value", str(value))
        if "." in val_str:
            val_str = val_str.split(".")[-1]
        normalized = val_str.strip().lower().replace("-", "_")
        for modality in cls:
            if modality.value == normalized or modality.name.lower() == normalized:
                return modality
        aliases = {
            "table": cls.TABULAR,
            "csv": cls.TABULAR,
            "nlp": cls.TEXT,
            "document": cls.TEXT,
            "sentence": cls.TEXT,
            "vision": cls.IMAGE,
            "images": cls.IMAGE,
            "picture": cls.IMAGE,
            "voice": cls.AUDIO,
            "speech": cls.AUDIO,
            "sound": cls.AUDIO,
            "audio_clip": cls.AUDIO,
            "movie": cls.VIDEO,
            "clip": cls.VIDEO,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise DLConfigError(
            f"Unknown modality '{value}'.",
            f"Valid modalities: {', '.join(m.value for m in cls)}",
        )


class DLTask(str, Enum):
    """What the model predicts."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    @classmethod
    def from_str(cls, value: Any) -> "DLTask":
        if isinstance(value, cls):
            return value
        val_str = getattr(value, "value", str(value))
        if "." in val_str:
            val_str = val_str.split(".")[-1]
        normalized = val_str.strip().lower().replace("-", "_")
        for task in cls:
            if task.value == normalized or task.name.lower() == normalized:
                return task
        aliases = {
            "classify": cls.CLASSIFICATION,
            "clf": cls.CLASSIFICATION,
            "label": cls.CLASSIFICATION,
            "multiclass": cls.CLASSIFICATION,
            "binary": cls.CLASSIFICATION,
            "regress": cls.REGRESSION,
            "reg": cls.REGRESSION,
            "continuous": cls.REGRESSION,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise DLConfigError(
            f"Unknown task '{value}'.",
            f"Valid tasks: {', '.join(t.value for t in cls)}",
        )


@dataclass
class DLResourceConfig:
    """Hardware limits and the consent settings that gate heavy downloads."""

    time_budget_secs: float = 900.0
    memory_budget_mb: Optional[float] = None
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 32
    num_workers: int = 0
    #: Allow ``dive dl`` to pip-install a missing backend after asking. Setting
    #: this False (or ``DIVE_NO_INSTALL=1``) restricts the run to what is present.
    allow_install: bool = True
    #: Skip the install prompt. Only set from an explicit ``--yes`` flag; never
    #: defaulted on, because the download can be ~250 MB.
    assume_yes: bool = False
    #: Fraction of usable RAM a projected peak may reach before the run is refused.
    safety_fraction: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_budget_secs": self.time_budget_secs,
            "memory_budget_mb": self.memory_budget_mb,
            "device": self.device,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "allow_install": self.allow_install,
            "assume_yes": self.assume_yes,
            "safety_fraction": self.safety_fraction,
        }


@dataclass
class DLTrainingConfig:
    """Optimisation loop settings, shared by every modality."""

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    early_stopping_patience: int = 3
    #: Mixed precision. Silently ignored on CPU, where it is not a win.
    mixed_precision: bool = False
    gradient_clip: Optional[float] = 1.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "early_stopping_patience": self.early_stopping_patience,
            "mixed_precision": self.mixed_precision,
            "gradient_clip": self.gradient_clip,
            "seed": self.seed,
        }


@dataclass
class DLDataConfig:
    """How to find the inputs and the target inside the source."""

    #: Tabular/text: the column holding the input. Image/audio/video: the column
    #: holding a file path, when loading from a manifest rather than a folder.
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    test_size: float = 0.2
    stratify: bool = True
    random_state: int = 42
    #: Per-modality preprocessing knobs, kept as a free-form mapping so a new
    #: modality does not force a schema change here.
    #: e.g. ``{"image_size": 96, "sample_rate": 16000, "frames_per_clip": 8}``
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_column": self.input_column,
            "target_column": self.target_column,
            "test_size": self.test_size,
            "stratify": self.stratify,
            "random_state": self.random_state,
            "options": dict(self.options),
        }


@dataclass
class DLConfig:
    """Complete declarative configuration for a ``dive dl`` experiment."""

    modality: str = Modality.TABULAR.value
    task: str = DLTask.CLASSIFICATION.value
    mode: str = "balanced"  # fast, balanced, quality
    output_dir: str = "./dive_dl_output"
    resources: DLResourceConfig = field(default_factory=DLResourceConfig)
    training: DLTrainingConfig = field(default_factory=DLTrainingConfig)
    data: DLDataConfig = field(default_factory=DLDataConfig)
    random_seed: int = 42

    def __post_init__(self) -> None:
        # Normalise through the enums so an invalid value fails here, at
        # construction, rather than deep inside a training loop.
        self.modality = Modality.from_str(self.modality).value
        self.task = DLTask.from_str(self.task).value
        if self.mode not in ("fast", "balanced", "quality"):
            raise DLConfigError(
                f"Unknown mode '{self.mode}'.",
                "Valid modes: fast, balanced, quality",
            )

    @property
    def modality_enum(self) -> Modality:
        return Modality.from_str(self.modality)

    @property
    def task_enum(self) -> DLTask:
        return DLTask.from_str(self.task)

    def apply_mode(self) -> "DLConfig":
        """Scale the training budget to ``mode``, returning ``self``.

        ``fast`` is what a first look should cost; ``quality`` is what a final
        model should cost. Only touches values the user has not overridden in a
        config file, which is why it is opt-in rather than run in ``__post_init__``.
        """
        presets = {
            "fast": (3, 1),
            "balanced": (10, 3),
            "quality": (30, 6),
        }
        epochs, patience = presets[self.mode]
        self.training.epochs = epochs
        self.training.early_stopping_patience = patience
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DLConfig":
        res = data.get("resources", {}) or {}
        train = data.get("training", {}) or {}
        source = data.get("data", {}) or {}

        hidden = train.get("hidden_sizes", [256, 128])
        if isinstance(hidden, (int, float)):
            hidden = [int(hidden)]

        return cls(
            modality=data.get("modality", Modality.TABULAR.value),
            task=data.get("task", DLTask.CLASSIFICATION.value),
            mode=data.get("mode", "balanced"),
            output_dir=data.get("output_dir", "./dive_dl_output"),
            resources=DLResourceConfig(
                time_budget_secs=res.get("time_budget_secs", 900.0),
                memory_budget_mb=res.get("memory_budget_mb"),
                device=res.get("device", "auto"),
                batch_size=res.get("batch_size", 32),
                num_workers=res.get("num_workers", 0),
                allow_install=res.get("allow_install", True),
                assume_yes=res.get("assume_yes", False),
                safety_fraction=res.get("safety_fraction", 0.85),
            ),
            training=DLTrainingConfig(
                epochs=train.get("epochs", 10),
                learning_rate=train.get("learning_rate", 1e-3),
                weight_decay=train.get("weight_decay", 0.0),
                hidden_sizes=[int(size) for size in hidden],
                dropout=train.get("dropout", 0.1),
                early_stopping_patience=train.get("early_stopping_patience", 3),
                mixed_precision=train.get("mixed_precision", False),
                gradient_clip=train.get("gradient_clip", 1.0),
                seed=train.get("seed", 42),
            ),
            data=DLDataConfig(
                input_column=source.get("input_column"),
                target_column=source.get("target_column"),
                test_size=source.get("test_size", 0.2),
                stratify=source.get("stratify", True),
                random_state=source.get("random_state", 42),
                options=source.get("options", {}) or {},
            ),
            random_seed=data.get("random_seed", 42),
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "DLConfig":
        """Load a DL configuration from YAML or JSON."""
        path = Path(file_path)
        if not path.exists():
            raise DLConfigError(f"DL configuration file '{path}' does not exist.")
        content = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise DLConfigError(
                    f"Cannot read '{path}' because PyYAML is not installed.",
                    "Use a .json config, or: pip install PyYAML",
                )
            data = yaml.safe_load(content) or {}
        else:
            data = json.loads(content)
        return cls.from_dict(data)

    def save(self, file_path: Union[str, Path]) -> Path:
        """Write this configuration to JSON or YAML."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml") and yaml is not None:
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "task": self.task,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "resources": self.resources.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "random_seed": self.random_seed,
        }
