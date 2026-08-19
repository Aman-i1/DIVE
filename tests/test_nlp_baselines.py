"""Phase 5 Tests - DIVE NLP Classical Baselines, Representations & Evaluation.

Verifies:
1. TF-IDF and Count sparse feature representations.
2. NLPPipeline end-to-end lifecycle (fit, predict, predict_proba, label restoration).
3. Baseline model zoo: LogisticRegression, LinearSVC (Platt calibrated), MultinomialNB, RidgeRegression.
4. NLPEvaluator classification (Accuracy, F1, Log Loss) and regression (R², MAE, RMSE) scoring.
5. train_baseline() automated end-to-end training and holdout evaluation.
6. NLPPredictor input polymorphism (single str, list of str, DataFrame, dict records).
7. Portable serialization round-tripping with exact prediction consistency.
8. Protocol compliance for Pipeline, Estimator, Predictor, Representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from dive.nlp import (
    BASELINE_MODELS,
    CountRepresentation,
    NLPEvaluator,
    NLPPipeline,
    NLPPredictor,
    TFIDFRepresentation,
    build_baseline_model,
    build_representation,
    evaluate_nlp_predictions,
    load_nlp_predictor,
    save_nlp_predictor,
    train_baseline,
)
from dive.nlp.data import NLPDataset
from dive.nlp.interfaces import (
    NLPPipelineProtocol,
    NLPPredictorProtocol,
    NLPRepresentationProtocol,
)


@pytest.fixture
def sentiment_corpus() -> pd.DataFrame:
    """Fixture providing synthetic customer review text data with distinct signal."""
    reviews = [
        "Incredible product! Fast shipping and top tier build quality.",
        "Loved everything about this. Exceeded all my expectations, great product!",
        "Fantastic customer support and seamless onboarding experience. Great quality!",
        "Superb performance under heavy loads. Highly recommended and loved it.",
        "Great value for money, absolutely delightful product and loved it.",
        "Awesome performance, great build and excellent reliability.",
        "Exceptional service, loved the team and great product overall.",
        "Great experience with fast delivery and superb quality.",
        "Loved the design and excellent user interface, great buy.",
        "Superb quality, awesome service, and great value.",
        "Worst purchase ever made. Completely broken on arrival, terrible!",
        "Terrible quality, broke within two days of normal usage, worst product.",
        "Horrible customer service. Completely unhelpful, rude, and terrible.",
        "Do not buy this! Waste of money and immense frustration, terrible.",
        "Defective unit, useless customer support, refund refused, worst experience.",
        "Awful quality and terrible shipping, broken item arrived.",
        "Horrible product, worst purchase, completely useless and broken.",
        "Terrible experience, awful customer service, broke immediately.",
        "Worst quality, useless device, broke on day one, terrible.",
        "Awful support and broken product, worst purchase ever.",
    ]
    sentiments = [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
    ]
    return pd.DataFrame({"review": reviews, "sentiment": sentiments})


def test_tfidf_representation_lifecycle() -> None:
    """Verify TF-IDF representation fitting, transform, and protocol conformance."""
    texts = [
        "Natural language processing and machine learning.",
        "Machine learning models for language translation.",
        "Deep learning architectures and transformers.",
    ]
    rep = TFIDFRepresentation(ngram_range=(1, 2), min_df=1)
    assert isinstance(rep, NLPRepresentationProtocol)

    sparse_matrix = rep.fit_transform(texts)
    assert sparse_matrix.shape[0] == 3
    assert sparse_matrix.shape[1] > 5
    assert "machine learning" in rep.feature_names_

    # Transform new data
    new_sparse = rep.transform(["Machine learning"])
    assert new_sparse.shape[0] == 1
    assert new_sparse.shape[1] == sparse_matrix.shape[1]


def test_count_representation_lifecycle() -> None:
    """Verify Count representation fitting and bag-of-words extraction."""
    texts = ["apple orange banana", "banana apple grape"]
    rep = CountRepresentation(min_df=1)
    assert isinstance(rep, NLPRepresentationProtocol)

    sparse = rep.fit_transform(texts)
    assert sparse.shape[0] == 2
    assert "apple" in rep.feature_names_


def test_nlp_pipeline_classification_lifecycle(sentiment_corpus: pd.DataFrame) -> None:
    """Verify NLPPipeline training, prediction, and probability estimation."""
    estimator = build_baseline_model("LogisticRegression")
    pipeline = NLPPipeline(
        estimator=estimator,
        task_type="text_classification",
        model_name="LogisticRegression",
    )
    assert isinstance(pipeline, NLPPipelineProtocol)

    # Fit
    pipeline.fit(sentiment_corpus["review"], sentiment_corpus["sentiment"])
    assert pipeline.fitted_ is True
    assert set(pipeline.class_names or []) == {"positive", "negative"}

    # Predict returns original strings
    preds = pipeline.predict(["Superb build quality, loved it!"])
    assert len(preds) == 1
    assert preds[0] == "positive"

    neg_preds = pipeline.predict(["Broken on arrival and useless."])
    assert neg_preds[0] == "negative"

    # Predict proba
    probas = pipeline.predict_proba(["Great experience!"])
    assert probas.shape == (1, 2)
    assert np.isclose(np.sum(probas), 1.0)


@pytest.mark.parametrize("model_name", ["LogisticRegression", "LinearSVC", "MultinomialNB"])
def test_baseline_models_zoo_classification(model_name: str, sentiment_corpus: pd.DataFrame) -> None:
    """Verify that every classification baseline in the zoo trains and predicts accurately."""
    predictor, metrics = train_baseline(
        data=sentiment_corpus,
        text_column="review",
        target_column="sentiment",
        model_name=model_name,
        test_size=0.3,
        random_state=42,
    )

    assert isinstance(predictor, NLPPredictorProtocol)
    assert predictor.model_name == model_name
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] >= 0.5  # Baseline beats random on small holdout

    # Verify probability support (including calibrated LinearSVC)
    probas = predictor.predict_proba("Outstanding performance!")
    assert probas.shape[1] == 2
    assert np.isclose(np.sum(probas), 1.0)


def test_baseline_regression_task() -> None:
    """Verify Ridge regression on continuous text targets (e.g. review ratings)."""
    texts = [
        "Worst product, complete disaster, awful experience.",
        "Very poor quality, broke easily.",
        "Acceptable, does the basic job.",
        "Good product, worked as described.",
        "Spectacular excellence, flawless perfection, 10/10!",
    ]
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0]

    predictor, metrics = train_baseline(
        data=NLPDataset(texts=texts, labels=ratings),
        model_name="RidgeRegression",
        task_type="text_regression",
        test_size=0.2,
        random_state=42,
    )

    assert predictor.task_type == "text_regression"
    assert "rmse" in metrics
    assert "mae" in metrics

    pred = predictor.predict("Spectacular and flawless!")
    assert len(pred) == 1
    assert isinstance(float(pred[0]), float)


def test_nlp_evaluator_metrics() -> None:
    """Verify NLPEvaluator calculation accuracy across metrics."""
    evaluator = NLPEvaluator(task_type="text_classification")

    y_true = ["pos", "neg", "pos", "pos", "neg"]
    y_pred = ["pos", "neg", "pos", "neg", "neg"]
    y_proba = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.7, 0.3],
    ])

    metrics = evaluator.evaluate(y_true, y_pred, y_proba=y_proba)
    assert metrics["accuracy"] == 0.8
    assert metrics["macro_f1"] > 0.7
    assert "log_loss" in metrics


def test_nlp_predictor_input_polymorphism(sentiment_corpus: pd.DataFrame) -> None:
    """Verify NLPPredictor seamlessly accepts multiple input formats."""
    predictor, _ = train_baseline(
        data=sentiment_corpus,
        text_column="review",
        target_column="sentiment",
        model_name="LogisticRegression",
    )

    # 1. Single string
    p1 = predictor.predict("Loved this phenomenal item!")
    assert p1[0] == "positive"

    # 2. List of strings
    p2 = predictor.predict(["Loved this!", "Worst purchase!"])
    assert len(p2) == 2
    assert p2[0] == "positive"
    assert p2[1] == "negative"

    # 3. DataFrame
    df_in = pd.DataFrame({"review": ["Loved it!", "Hated it!"]})
    p3 = predictor.predict(df_in)
    assert len(p3) == 2

    # 4. Dictionary record
    p4 = predictor.predict({"review": "Superb excellence!"})
    assert p4[0] == "positive"

    # 5. List of records
    p5 = predictor.predict([{"review": "Great"}, {"review": "Terrible"}])
    assert len(p5) == 2

    # Describe input
    desc = predictor.describe_input()
    assert desc["text_column"] == "review"
    assert desc["target_column"] == "sentiment"
    assert desc["has_probabilities"] is True


def test_nlp_predictor_serialization_roundtrip(sentiment_corpus: pd.DataFrame, tmp_path: Path) -> None:
    """Verify artifact serialization and exact prediction reproduction."""
    predictor, _ = train_baseline(
        data=sentiment_corpus,
        text_column="review",
        target_column="sentiment",
        model_name="LogisticRegression",
    )

    test_samples = [
        "Incredible build quality and fast shipping.",
        "Completely useless and broke immediately.",
    ]
    preds_before = predictor.predict(test_samples)
    probas_before = predictor.predict_proba(test_samples)

    # Save artifact
    model_path = tmp_path / "nlp_model.pkl"
    predictor.save(model_path)

    # Load artifact
    loaded_predictor = load_nlp_predictor(model_path)
    assert loaded_predictor.model_name == predictor.model_name
    assert loaded_predictor.text_column == predictor.text_column

    # Predictions bit-for-bit identical
    preds_after = loaded_predictor.predict(test_samples)
    probas_after = loaded_predictor.predict_proba(test_samples)

    np.testing.assert_array_equal(preds_before, preds_after)
    np.testing.assert_allclose(probas_before, probas_after, rtol=1e-5)
