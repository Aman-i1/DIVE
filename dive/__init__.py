"""dive - automated machine learning for tabular data, from the terminal.

Public API::

    from dive import Dive, quick_dive

    dive = Dive(target="diagnosis", mode="fast")
    dive.fit(dataframe)
    predictions = dive.predict(new_dataframe)

The command-line entry point is :func:`dive.cli.main`, installed as ``dive``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from dive.core import Dive, Evaluator, build_preprocessor, quick_dive
from dive.data_intelligence import DataIntelligence
from dive.exceptions import (
    DiveError,
    ConfigError,
    DataError,
    ModelError,
    SchemaError,
    TargetError,
    TrainingError,
    ValidationError,
)
from dive.feature_engineering import FeatureEngineer
from dive.model_zoo import ModelZoo

__all__ = [
    "__version__",
    "Dive",
    "DataIntelligence",
    "Evaluator",
    "FeatureEngineer",
    "ModelZoo",
    "build_preprocessor",
    "quick_dive",
    "DiveError",
    "ConfigError",
    "DataError",
    "ModelError",
    "SchemaError",
    "TargetError",
    "TrainingError",
    "ValidationError",
]
