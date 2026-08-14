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
from dive.adversarial_validation import (
    AdversarialValidationReport,
    AdversarialValidator,
)
from dive.artifact_store import ArtifactStore, StoredArtifact
from dive.audit import AuditCertificate, ComplianceAuditor
from dive.autopilot import AutopilotOrchestrator, AutopilotResult
from dive.batch_inference import BatchInferenceEngine, BatchInferenceStats
from dive.calibration import ProbabilityCalibrator
from dive.champion_challenger import (
    ChampionChallengerEvaluator,
    PromotionVerdict,
)
from dive.config import DiveConfig
from dive.contamination import ContaminationDetector, ContaminationReport
from dive.core import Dive, Evaluator, build_preprocessor, quick_dive
from dive.data_intelligence import DataIntelligence
from dive.data_quality import (
    DataQualityEngine,
    DataQualityReport,
    InferredRelationalRule,
)
from dive.decisions import DecisionLogger, DecisionRecord
from dive.doctor import DiveDoctor, ProductionReadinessScore
from dive.drift import DriftDetector
from dive.failure_segments import (
    FailureSegment,
    FailureSegmentAnalyzer,
    FailureSegmentsReport,
)
from dive.model_stress import ModelStressTester, StressTestReport
from dive.observability import (
    DriftMetricResult,
    ObservabilityEngine,
    ObservabilityReport,
)
from dive.lineage import LineageGraph, LineageNode
from dive.prediction_contract import (
    PredictionContract,
    PredictionContractEngine,
)
from dive.reproducibility import (
    ReproducibilityBundleExporter,
    ReproducibilityBundleMetadata,
)
from dive.senior_review import SeniorReviewEngine, SeniorReviewReport
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
from dive.gate import DeploymentGate, GateVerdict
from dive.info import DatasetInspector, DatasetInfoReport
from dive.leakage import AdvancedLeakageDetector
from dive.capability_registry import CapabilityRegistry, ModelCapability
from dive.feature_availability import FeatureAvailabilityModel, FeatureMetadata
from dive.feature_selection import FeaturePruner
from dive.meta_learning import (
    DatasetFingerprint,
    MetaLearningEngine,
    MetaWarmStartPriors,
)
from dive.ensemble_diversity import DiversityMatrix, ModelDiversityEvaluator
from dive.inference_router import DynamicInferenceRouter, RoutedPredictionResult
from dive.model_zoo import ModelZoo
from dive.ood_detector import OODDetector, OODResult
from dive.registry import ModelRegistry, PromotionGate
from dive.search_scheduler import ASHASearchScheduler, Rung, Trial
from dive.security import SecurityAuditResult, SecurityAuditor
from dive.stacking_calibrated import (
    CalibratedStackingEnsemble,
    EnsembleWeightsResult,
)
from dive.study import Study, create_study
from dive.temporal_features import LeakageSafeTemporalEngine
from dive.trust import (
    PerturbationRobustnessResult,
    TrustEngine,
    TrustReport,
)
from dive.uncertainty import (
    ConformalIntervalResult,
    ConformalPredictor,
    ConformalSetResult,
    UncertaintyDecomposition,
)
from dive.validation_engine import (
    ValidationIntelligenceEngine,
    ValidationPlan,
    ValidationRiskScore,
)

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
    "ComplianceAuditor",
    "AuditCertificate",
    "ONNXExporter",
    "DeploymentGate",
    "GateVerdict",
    "DivePredictor",
    "Evaluator",
    "FeatureEngineer",
    "ModelZoo",
    "build_preprocessor",
    "load_predictor",
    "quick_dive",
    "Study",
    "create_study",
    "DecisionRecord",
    "DecisionLogger",
    "StudyConfig",
    "StudyOrchestrator",
    "ValidationIntelligenceEngine",
    "ValidationPlan",
    "ValidationRiskScore",
    "FeatureAvailabilityModel",
    "FeatureMetadata",
    "LeakageSafeTemporalEngine",
    "FeaturePruner",
    "ModelCapability",
    "CapabilityRegistry",
    "ASHASearchScheduler",
    "Trial",
    "Rung",
    "DatasetFingerprint",
    "MetaLearningEngine",
    "MetaWarmStartPriors",
    "ConformalPredictor",
    "ConformalIntervalResult",
    "ConformalSetResult",
    "UncertaintyDecomposition",
    "OODDetector",
    "OODResult",
    "TrustEngine",
    "TrustReport",
    "PerturbationRobustnessResult",
    "DiversityMatrix",
    "ModelDiversityEvaluator",
    "CalibratedStackingEnsemble",
    "EnsembleWeightsResult",
    "DynamicInferenceRouter",
    "RoutedPredictionResult",
    "ArtifactStore",
    "StoredArtifact",
    "LineageGraph",
    "LineageNode",
    "ReproducibilityBundleExporter",
    "ReproducibilityBundleMetadata",
    "BatchInferenceEngine",
    "BatchInferenceStats",
    "ObservabilityEngine",
    "ObservabilityReport",
    "DriftMetricResult",
    "ChampionChallengerEvaluator",
    "PredictionContract",
    "PredictionContractEngine",
    "DataQualityEngine",
    "DataQualityReport",
    "InferredRelationalRule",
    "ContaminationDetector",
    "ContaminationReport",
    "AdversarialValidator",
    "AdversarialValidationReport",
    "ModelStressTester",
    "StressTestReport",
    "FailureSegmentAnalyzer",
    "FailureSegmentsReport",
    "FailureSegment",
    "SeniorReviewEngine",
    "SeniorReviewReport",
    "AutopilotOrchestrator",
    "AutopilotResult",
    "DiveConfig",
    "SecurityAuditor",
    "SecurityAuditResult",
    "DiveError",
    "ConfigError",
    "DataError",
    "ModelError",
    "SchemaError",
    "TargetError",
    "TrainingError",
    "ValidationError",
]



