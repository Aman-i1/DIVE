"""Transformer Fine-Tuning & Pipeline Orchestration - `dive/nlp/transformers/training.py`.

Orchestrates Transformer dataset preparation, fine-tuning, holdout evaluation,
and self-contained NLPPredictor packaging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pandas as pd

from dive.nlp.data.dataset import NLPDataset
from dive.nlp.evaluation.evaluator import NLPEvaluator
from dive.nlp.exceptions import TextDataError
from dive.nlp.inference.predictor import NLPPredictor
from dive.nlp.pipeline import NLPPipeline
from dive.nlp.preprocessing.preprocessor import NLPPreprocessor
from dive.nlp.transformers.config import TransformerConfig
from dive.nlp.transformers.estimator import (
    TransformerClassifier,
    TransformerRegressor,
)


class _IdentityRepresentation:
    """Non-modifying representation passthrough for Transformer tokenizers."""

    def __init__(self) -> None:
        self.fitted_ = True

    def fit(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> "_IdentityRepresentation":
        return self

    def transform(self, texts: Sequence[str]) -> Sequence[str]:
        return texts

    def fit_transform(self, texts: Sequence[str], y: Optional[Sequence[Any]] = None) -> Sequence[str]:
        return texts


def train_transformer(
    data: Union[NLPDataset, str, Path, pd.DataFrame, Sequence[Dict[str, Any]]],
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
    model_name: str = "distilbert",
    task_type: str = "text_classification",
    config: Optional[TransformerConfig] = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_length: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[NLPPredictor, Dict[str, Any]]:
    """Fine-tune a pretrained Transformer model on text data.

    Parameters
    ----------
    data : Union[NLPDataset, str, Path, pd.DataFrame, Sequence[Dict]]
        Input dataset or file path.
    target_column : Optional[str]
        Target label column name.
    text_column : Optional[str]
        Text feature column name.
    model_name : str
        Transformer architecture name ('distilbert', 'bert', 'roberta', 'deberta').
    task_type : str
        Task type ('text_classification' or 'text_regression').
    config : Optional[TransformerConfig]
        Transformer hyperparameter configuration.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        AdamW learning rate.
    max_seq_length : int
        Maximum sequence length for tokenization.
    test_size : float
        Holdout test set fraction.
    random_state : int
        Seed for deterministic splitting.

    Returns
    -------
    Tuple[NLPPredictor, Dict[str, Any]]
        The fitted predictor bundle and holdout evaluation metrics.
    """
    # 1. Dataset Resolution
    if isinstance(data, NLPDataset):
        ds = data
    elif isinstance(data, (str, Path)):
        ds = NLPDataset.from_file(file_path=data, text_column=text_column, target_column=target_column)
    elif isinstance(data, pd.DataFrame):
        ds = NLPDataset.from_dataframe(df=data, text_column=text_column, target_column=target_column)
    elif isinstance(data, Sequence) and len(data) > 0 and isinstance(data[0], dict):
        ds = NLPDataset.from_records(records=data, text_key=text_column or "text", target_key=target_column or "label")  # type: ignore
    else:
        raise TextDataError(f"Unsupported data input type: {type(data).__name__}")

    if not ds.has_labels:
        raise TextDataError("Dataset has no target labels. Cannot train Transformer model.")

    actual_text_col = text_column or "text"
    actual_target_col = target_column or "label"

    # 2. Holdout split
    train_ds, test_ds = ds.split(
        test_size=test_size,
        stratify=(task_type in ("text_classification", "classification")),
        random_state=random_state,
    )

    # 3. Model Configuration
    tf_cfg = config or TransformerConfig(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        seed=random_state,
    )

    # 4. Construct Estimator
    if task_type in ("text_regression", "regression"):
        estimator = TransformerRegressor(config=tf_cfg)
    else:
        estimator = TransformerClassifier(config=tf_cfg)

    # 5. Fit Estimator with raw preprocessor passthrough
    pipeline = NLPPipeline(
        estimator=estimator,
        preprocessor=NLPPreprocessor.raw(),
        representation=_IdentityRepresentation(),  # type: ignore
        task_type=task_type,
        model_name=model_name,
    )
    pipeline.fit(train_ds.texts, train_ds.labels)  # type: ignore

    # 6. Holdout Evaluation
    y_test_pred = pipeline.predict(test_ds.texts)
    y_test_proba = None
    if task_type in ("text_classification", "classification") and hasattr(estimator, "predict_proba"):
        try:
            y_test_proba = pipeline.predict_proba(test_ds.texts)
        except Exception:
            y_test_proba = None

    evaluator = NLPEvaluator(task_type=task_type)
    metrics = evaluator.evaluate(
        y_true=test_ds.labels,  # type: ignore
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        class_names=pipeline.class_names,
    )

    # 7. Predictor Bundle
    predictor = NLPPredictor(
        pipeline=pipeline,
        model_name=f"Transformer_{model_name}",
        text_column=actual_text_col,
        target_column=actual_target_col,
        task_type=task_type,
        metrics=metrics,
        dataset_name=ds.name,
    )

    return predictor, metrics
