"""Classical NLP Baseline Training Routine - `dive/nlp/training.py`.

Orchestrates data preparation, preprocessing, TF-IDF representation,
model training, holdout evaluation, and predictor artifact bundling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pandas as pd

from dive.nlp.config import NLPConfig, NLPPreprocessingConfig, NLPRepresentationConfig
from dive.nlp.data.dataset import NLPDataset
from dive.nlp.evaluation.evaluator import NLPEvaluator
from dive.nlp.exceptions import TextDataError
from dive.nlp.features import build_representation
from dive.nlp.inference.predictor import NLPPredictor
from dive.nlp.models.baselines import build_baseline_model
from dive.nlp.pipeline import NLPPipeline
from dive.nlp.preprocessing.preprocessor import NLPPreprocessor


def train_baseline(
    data: Union[NLPDataset, str, Path, pd.DataFrame, Sequence[Dict[str, Any]]],
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
    model_name: str = "LogisticRegression",
    representation_type: str = "tfidf",
    representation: Optional[Any] = None,
    task_type: str = "text_classification",
    config: Optional[NLPConfig] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[NLPPredictor, Dict[str, Any]]:
    """Train a fast, calibrated CPU baseline model on text data.

    Parameters
    ----------
    data : Union[NLPDataset, str, Path, pd.DataFrame, Sequence[Dict]]
        Input dataset or file path (CSV, JSON, Parquet).
    target_column : Optional[str]
        Target label column name.
    text_column : Optional[str]
        Text feature column name (auto-detected if None).
    model_name : str
        Name of baseline estimator ('LogisticRegression', 'LinearSVC', 'MultinomialNB', 'RidgeRegression').
    representation_type : str
        Feature representation type ('tfidf', 'char_ngrams', 'word_char_union', 'bm25', 'count').
    representation : Optional[Any]
        Optional custom representation engine instance.
    task_type : str
        Task type ('text_classification' or 'text_regression').
    config : Optional[NLPConfig]
        Optional declarative configuration.
    test_size : float
        Holdout test set fraction.
    random_state : int
        Seed for deterministic splitting and training.

    Returns
    -------
    Tuple[NLPPredictor, Dict[str, Any]]
        The fitted predictor bundle and holdout evaluation metrics.
    """
    # 1. Dataset Resolution
    if isinstance(data, NLPDataset):
        ds = data
    elif isinstance(data, (str, Path)):
        ds = NLPDataset.from_file(
            file_path=data,
            text_column=text_column or (config.text_column if config else None),
            target_column=target_column or (config.target_column if config else None),
        )
    elif isinstance(data, pd.DataFrame):
        ds = NLPDataset.from_dataframe(
            df=data,
            text_column=text_column or (config.text_column if config else None),
            target_column=target_column or (config.target_column if config else None),
        )
    elif isinstance(data, Sequence) and len(data) > 0 and isinstance(data[0], dict):
        ds = NLPDataset.from_records(
            records=data,  # type: ignore
            text_key=text_column or "text",
            target_key=target_column or "label",
        )
    else:
        raise TextDataError(f"Unsupported data input type: {type(data).__name__}")

    if not ds.has_labels:
        raise TextDataError("Dataset has no target labels. Cannot train supervised baseline.")

    actual_task = config.task if config else task_type
    actual_text_col = text_column or (config.text_column if config else "text")
    actual_target_col = target_column or (config.target_column if config else "label")

    # 2. Deterministic Train / Holdout Test Split
    train_ds, test_ds = ds.split(
        test_size=test_size,
        stratify=(actual_task in ("text_classification", "classification")),
        random_state=random_state,
    )

    # 3. Preprocessor & Representation
    prep_cfg = config.preprocessing if config else None
    preprocessor = NLPPreprocessor(config=prep_cfg)

    rep_cfg = config.representation if config else None
    if representation is not None:
        rep_engine = representation
    else:
        rep_engine = build_representation(config=rep_cfg, representation_type=representation_type)

    # 4. Model Construction & Fitting
    estimator = build_baseline_model(
        model_name=model_name,
        problem_type=actual_task,
        random_state=random_state,
    )

    pipeline = NLPPipeline(
        estimator=estimator,
        preprocessor=preprocessor,
        representation=rep_engine,
        task_type=actual_task,
        model_name=model_name,
    )
    pipeline.fit(train_ds.texts, train_ds.labels)  # type: ignore

    # 5. Holdout Evaluation
    y_test_pred = pipeline.predict(test_ds.texts)
    y_test_proba = None
    if actual_task in ("text_classification", "classification") and hasattr(estimator, "predict_proba"):
        try:
            y_test_proba = pipeline.predict_proba(test_ds.texts)
        except Exception:
            y_test_proba = None

    evaluator = NLPEvaluator(task_type=actual_task)
    metrics = evaluator.evaluate(
        y_true=test_ds.labels,  # type: ignore
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        class_names=pipeline.class_names,
    )

    # 6. Standalone Predictor Bundle
    predictor = NLPPredictor(
        pipeline=pipeline,
        model_name=model_name,
        text_column=actual_text_col,
        target_column=actual_target_col,
        task_type=actual_task,
        metrics=metrics,
        dataset_name=ds.name,
    )

    return predictor, metrics
