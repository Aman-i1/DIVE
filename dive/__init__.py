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

from dive.advisor import ModelAdvisor, ValidationAdvisor
from dive.calibration import ProbabilityCalibrator
from dive.core import Dive, Evaluator, build_preprocessor, quick_dive
from dive.data_intelligence import DataIntelligence
from dive.doctor import DiveDoctor, ProductionReadinessScore
from dive.drift import DriftDetector
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
from dive.experiments import ExperimentTracker
from dive.explainability import ExplainabilityEngine
from dive.failure_analysis import ModelFailureAnalyzer
from dive.info import DatasetInspector, DatasetInfoReport
from dive.leakage import AdvancedLeakageDetector
from dive.model_zoo import ModelZoo
from dive.predictor import DivePredictor, load_predictor
from dive.registry import ModelRegistry, PromotionGate
from dive.resources import ResourceManager

__all__ = [
    "__version__",
    "Dive",
    "DataIntelligence",
    "DatasetInspector",
    "DatasetInfoReport",
    "DiveDoctor",
    "ProductionReadinessScore",
    "ValidationAdvisor",
    "ModelAdvisor",
    "AdvancedLeakageDetector",
    "ResourceManager",
    "ModelFailureAnalyzer",
    "ProbabilityCalibrator",
    "ExplainabilityEngine",
    "ExperimentTracker",
    "ModelRegistry",
    "PromotionGate",
    "DriftDetector",
    "DivePredictor",
    "Evaluator",
    "FeatureEngineer",
    "ModelZoo",
    "build_preprocessor",
    "load_predictor",
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


