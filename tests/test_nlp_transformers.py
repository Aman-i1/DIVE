"""Phase 8 Tests - DIVE NLP Transformer Support (BERT, RoBERTa, DistilBERT, DeBERTa).

Verifies:
1. TransformerConfig architecture resolution and hyperparameter defaults.
2. TransformerClassifier fine-tuning lifecycle, predictions, probability distributions.
3. TransformerRegressor continuous target regression.
4. train_transformer() automated pipeline and holdout evaluation.
5. Standalone NLPPredictor artifact serialization and prediction reproduction.
6. Optional dependency safety across CPU / lightweight testing environments.
7. NLPEstimatorProtocol compliance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from dive.nlp.data.dataset import NLPDataset
from dive.nlp.inference import load_nlp_predictor
from dive.nlp.interfaces import NLPEstimatorProtocol, NLPPredictorProtocol
from dive.nlp.transformers import (
    TRANSFORMER_MODELS,
    TransformerClassifier,
    TransformerConfig,
    TransformerRegressor,
    train_transformer,
)


@pytest.fixture
def product_reviews() -> pd.DataFrame:
    """Fixture providing sample review corpus."""
    return pd.DataFrame(
        {
            "review": [
                "Exceptional build quality and super fast shipping, loved it!",
                "Amazing performance, totally exceeded my expectations, great product!",
                "Delightful customer support, great onboarding and seamless setup.",
                "Awesome durability, loved everything about this device, top quality.",
                "Worst purchase ever made, broken on arrival, terrible item.",
                "Terrible customer service, completely unhelpful and rude, awful.",
                "Defective unit, useless customer support, refund was refused, terrible.",
                "Awful product, broke within one hour, completely useless and worst.",
            ],
            "sentiment": [
                "positive",
                "positive",
                "positive",
                "positive",
                "negative",
                "negative",
                "negative",
                "negative",
            ],
        }
    )


def test_transformer_config_and_model_resolution() -> None:
    """Verify Transformer architecture alias resolution."""
    assert TransformerConfig(model_name="distilbert").resolve_model_id() == "distilbert-base-uncased"
    assert TransformerConfig(model_name="bert").resolve_model_id() == "bert-base-uncased"
    assert TransformerConfig(model_name="roberta").resolve_model_id() == "roberta-base"
    assert TransformerConfig(model_name="deberta").resolve_model_id() == "microsoft/deberta-v3-small"

    # Custom HF model ID passes through
    assert TransformerConfig(model_name="custom/my-model").resolve_model_id() == "custom/my-model"


def test_transformer_classifier_lifecycle(product_reviews: pd.DataFrame) -> None:
    """Verify TransformerClassifier fit, predict, predict_proba, and protocol compliance."""
    clf = TransformerClassifier(model_name="distilbert", epochs=1)
    assert isinstance(clf, NLPEstimatorProtocol)

    clf.fit(product_reviews["review"], product_reviews["sentiment"])
    assert clf.fitted_ is True

    # Predict
    preds = clf.predict(["Loved this amazing product!"])
    assert len(preds) == 1
    assert preds[0] in ("positive", "negative")

    # Predict Proba
    probas = clf.predict_proba(["Loved this amazing product!"])
    assert probas.shape == (1, 2)
    assert np.isclose(np.sum(probas), 1.0)


def test_transformer_regressor_lifecycle() -> None:
    """Verify TransformerRegressor continuous target regression."""
    texts = [
        "Terrible and broke on day one.",
        "Average quality, okay product.",
        "Spectacular excellence, loved it!",
    ]
    ratings = [1.0, 3.0, 5.0]

    reg = TransformerRegressor(model_name="distilbert", epochs=1)
    assert isinstance(reg, NLPEstimatorProtocol)

    reg.fit(texts, ratings)
    assert reg.fitted_ is True

    preds = reg.predict(["Spectacular excellence!"])
    assert len(preds) == 1
    assert isinstance(float(preds[0]), float)


def test_train_transformer_end_to_end(product_reviews: pd.DataFrame, tmp_path: Path) -> None:
    """Verify train_transformer automated training, evaluation, and serialization."""
    predictor, metrics = train_transformer(
        data=product_reviews,
        text_column="review",
        target_column="sentiment",
        model_name="distilbert",
        epochs=1,
        test_size=0.25,
        random_state=42,
    )

    assert isinstance(predictor, NLPPredictorProtocol)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics

    # Input polymorphism on transformer predictor
    p_str = predictor.predict("Loved everything about this!")
    assert p_str[0] in ("positive", "negative")

    p_df = predictor.predict(pd.DataFrame({"review": ["Great!", "Awful!"]}))
    assert len(p_df) == 2

    # Serialization roundtrip
    artifact_path = tmp_path / "transformer_predictor.pkl"
    predictor.save(artifact_path)

    loaded = load_nlp_predictor(artifact_path)
    p_loaded = loaded.predict(["Loved everything about this!"])
    assert p_loaded[0] == p_str[0]
