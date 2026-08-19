"""DIVE ML - Tabular Machine Learning Capability Domain.

Provides automated machine learning, data intelligence, validation,
calibrated ensembling, and MLOps for structured and tabular data.
"""

from __future__ import annotations

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
from dive.capability_registry import CapabilityRegistry, ModelCapability
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
from dive.ensemble_diversity import DiversityMatrix, ModelDiversityEvaluator
from dive.exceptions import (
    ConfigError,
    DataError,
    DiveError,
    ModelError,
    SchemaError,
    TargetError,
    TrainingError,
    ValidationError,
)
from dive.experiments import ExperimentTracker
from dive.explainability import ExplainabilityEngine
from dive.failure_analysis import ModelFailureAnalyzer
from dive.failure_segments import (
    FailureSegment,
    FailureSegmentAnalyzer,
    FailureSegmentsReport,
)
from dive.feature_availability import FeatureAvailabilityModel, FeatureMetadata
from dive.feature_engineering import FeatureEngineer
from dive.feature_selection import FeaturePruner
from dive.gate import DeploymentGate, GateVerdict
from dive.inference_router import DynamicInferenceRouter, RoutedPredictionResult
from dive.info import DatasetInfoReport, DatasetInspector
from dive.leakage import AdvancedLeakageDetector
from dive.lineage import LineageGraph, LineageNode
from dive.meta_learning import (
    DatasetFingerprint,
    MetaLearningEngine,
    MetaWarmStartPriors,
)
from dive.model_stress import ModelStressTester, StressTestReport
from dive.model_zoo import ModelZoo
from dive.observability import (
    DriftMetricResult,
    ObservabilityEngine,
    ObservabilityReport,
)
from dive.onnx_export import ONNXExporter
from dive.orchestration import StudyConfig, StudyOrchestrator
from dive.prediction_contract import (
    PredictionContract,
    PredictionContractEngine,
)
from dive.predictor import DivePredictor, load_predictor
from dive.registry import ModelRegistry, PromotionGate
from dive.resources import ResourceManager
from dive.reproducibility import (
    ReproducibilityBundleExporter,
    ReproducibilityBundleMetadata,
)
from dive.search_scheduler import ASHASearchScheduler, Rung, Trial
from dive.security import SecurityAuditResult, SecurityAuditor
from dive.senior_review import SeniorReviewEngine, SeniorReviewReport
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
from dive.tuning import OptunaOptimizer
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
    "Dive",
    "DataIntelligence",
    "DatasetInspector",
    "DatasetInfoReport",
    "DiveDoctor",
    "ProductionReadinessScore",
    "ValidationAdvisor",
    "ModelAdvisor",
    "AdvancedLeakageDetector",
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
    "ResourceManager",
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
