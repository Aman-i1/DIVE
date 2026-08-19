"""DIVE NLP - Natural Language Processing Capability Domain.

Provides autonomous text modeling, profiling, classical baselines, embeddings,
and transformer pipelines for natural language workloads.
"""

from __future__ import annotations

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
from dive.nlp.data import (
    DatasetSplitter,
    NLPDataset,
    NLPSample,
    load_nlp_dataset,
)
from dive.nlp.profiling import (
    NLPProfileReport,
    NLPProfiler,
    profile_nlp_dataset,
)
from dive.nlp.preprocessing import (
    NLPPreprocessor,
    TextNormalizer,
    build_nlp_preprocessor,
)
from dive.nlp.features import (
    BM25Representation,
    CharNGramRepresentation,
    CountRepresentation,
    EmbeddingRepresentation,
    TFIDFRepresentation,
    WordCharUnionRepresentation,
    build_representation,
)
from dive.nlp.embeddings import (
    EmbeddingCache,
    benchmark_tfidf_vs_embeddings,
    compute_semantic_similarity,
)
from dive.nlp.evaluation import (
    NLPEvaluator,
    evaluate_nlp_predictions,
)
from dive.nlp.models import (
    BASELINE_MODELS,
    build_baseline_model,
)
from dive.nlp.transformers import (
    TRANSFORMER_MODELS,
    TransformerClassifier,
    TransformerConfig,
    TransformerRegressor,
    train_transformer,
)
from dive.nlp.automl import (
    AutoNLP,
    NLPLeaderboard,
    NLPTrial,
    fit_nlp,
)
from dive.nlp.optimization import (
    BatchInferenceEngine,
    ONNXNLPPredictor,
    OptimizedNLPPredictor,
    PredictionCache,
    export_nlp_to_onnx,
    optimize_nlp_predictor,
)
from dive.nlp.serving import (
    create_nlp_serving_app,
    serve_nlp_model,
)
from dive.nlp.monitoring import (
    NLPDriftMonitor,
    NLPDriftReport,
    monitor_nlp_drift,
)
from dive.nlp.pipeline import NLPPipeline
from dive.nlp.inference import (
    NLPPredictor,
    load_nlp_predictor,
    save_nlp_predictor,
)
from dive.nlp.training import train_baseline
from dive.nlp.registry import (
    NLPModelCapability,
    NLPRegistry,
)

__all__ = [
    # Autonomous Exploration & Selection
    "AutoNLP",
    "fit_nlp",
    "NLPLeaderboard",
    "NLPTrial",
    # Monitoring & Drift Detection
    "NLPDriftMonitor",
    "NLPDriftReport",
    "monitor_nlp_drift",
    # Serving & REST Endpoints
    "create_nlp_serving_app",
    "serve_nlp_model",
    # Optimization & Deployment
    "OptimizedNLPPredictor",
    "PredictionCache",
    "BatchInferenceEngine",
    "ONNXNLPPredictor",
    "export_nlp_to_onnx",
    "optimize_nlp_predictor",
    # Training & Prediction
    "train_baseline",
    "train_transformer",
    "NLPPredictor",
    "NLPPipeline",
    "save_nlp_predictor",
    "load_nlp_predictor",
    # Features & Models
    "TFIDFRepresentation",
    "CharNGramRepresentation",
    "WordCharUnionRepresentation",
    "BM25Representation",
    "CountRepresentation",
    "EmbeddingRepresentation",
    "build_representation",
    "BASELINE_MODELS",
    "build_baseline_model",
    # Transformers
    "TransformerConfig",
    "TRANSFORMER_MODELS",
    "TransformerClassifier",
    "TransformerRegressor",
    # Embeddings
    "EmbeddingCache",
    "compute_semantic_similarity",
    "benchmark_tfidf_vs_embeddings",
    # Evaluation
    "NLPEvaluator",
    "evaluate_nlp_predictions",
    # Data
    "NLPDataset",
    "NLPSample",
    "DatasetSplitter",
    "load_nlp_dataset",
    # Profiling
    "NLPProfiler",
    "NLPProfileReport",
    "profile_nlp_dataset",
    # Preprocessing
    "TextNormalizer",
    "NLPPreprocessor",
    "build_nlp_preprocessor",
    # Config
    "NLPConfig",
    "NLPResourceConfig",
    "NLPValidationConfig",
    "NLPPreprocessingConfig",
    "NLPRepresentationConfig",
    "NLPTaskType",
    # Exceptions
    "NLPError",
    "TextDataError",
    "NLPConfigError",
    "NLPModelError",
    "NLPTrainingError",
    "NLPInferenceError",
    "TokenizationError",
    "VocabularyError",
    "TaskNotSupportedError",
    # Interfaces
    "NLPDatasetProtocol",
    "NLPPreprocessorProtocol",
    "NLPRepresentationProtocol",
    "NLPEstimatorProtocol",
    "NLPPipelineProtocol",
    "NLPProfilerProtocol",
    "NLPPredictorProtocol",
    # Registry
    "NLPRegistry",
    "NLPModelCapability",
]
