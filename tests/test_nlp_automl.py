"""Phase 9 Tests - DIVE AutoNLP Automated Model Selection & Exploration.

Verifies:
1. NLPTrial metadata and multi-objective scoring.
2. NLPLeaderboard ranking, champion selection, DataFrame export, and ASCII rendering.
3. AutoNLP candidate space exploration across representations and models.
4. Multi-objective trade-off modes: 'accuracy', 'latency', 'balanced'.
5. AutoNLP regression exploration.
6. Top-level fit_nlp() entrypoint.
7. Champion NLPPredictor packaging and inference.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dive.nlp import (
    AutoNLP,
    NLPLeaderboard,
    NLPTrial,
    fit_nlp,
)
from dive.nlp.interfaces import NLPPredictorProtocol


@pytest.fixture
def sentiment_corpus() -> pd.DataFrame:
    """Fixture providing sample review corpus."""
    return pd.DataFrame(
        {
            "text": [
                "Exceptional build quality and super fast shipping, loved it!",
                "Amazing performance, totally exceeded my expectations, great product!",
                "Delightful customer support, great onboarding and seamless setup.",
                "Awesome durability, loved everything about this device, top quality.",
                "Great experience with fast delivery and superb quality overall.",
                "Loved the design and excellent user interface, great buy.",
                "Worst purchase ever made, broken on arrival, terrible item.",
                "Terrible customer service, completely unhelpful and rude, awful.",
                "Defective unit, useless customer support, refund was refused, terrible.",
                "Awful product, broke within one hour, completely useless and worst.",
                "Horrible product, worst purchase, completely useless and broken.",
                "Terrible experience, awful customer service, broke immediately.",
            ],
            "label": [
                "pos",
                "pos",
                "pos",
                "pos",
                "pos",
                "pos",
                "neg",
                "neg",
                "neg",
                "neg",
                "neg",
                "neg",
            ],
        }
    )


def test_nlp_trial_and_leaderboard() -> None:
    """Verify NLPTrial properties and NLPLeaderboard ranking and rendering."""
    board = NLPLeaderboard(primary_metric="macro_f1")

    t1 = NLPTrial(
        trial_id=1,
        representation_type="tfidf",
        model_name="LogisticRegression",
        primary_metric_score=0.85,
        composite_score=0.82,
        inference_latency_ms=0.5,
        status="SUCCESS",
    )
    t2 = NLPTrial(
        trial_id=2,
        representation_type="word_char_union",
        model_name="LinearSVC",
        primary_metric_score=0.92,
        composite_score=0.90,
        inference_latency_ms=0.8,
        status="SUCCESS",
    )
    t3 = NLPTrial(
        trial_id=3,
        representation_type="bm25",
        model_name="MultinomialNB",
        status="FAILED",
        error_msg="Sample error",
    )

    board.add_trial(t1)
    board.add_trial(t2)
    board.add_trial(t3)

    # 1. Champion selection
    assert board.champion_trial is not None
    assert board.champion_trial.model_name == "LinearSVC"
    assert board.champion_trial.primary_metric_score == 0.92

    # 2. DataFrame conversion
    df = board.to_dataframe()
    assert len(df) == 2  # Only successful trials in leaderboard ranking
    assert df.iloc[0]["Model"] == "LinearSVC"

    # 3. ASCII rendering
    rendered = board.render()
    assert "DIVE AUTONLP MODEL SELECTION LEADERBOARD" in rendered
    assert "Champion Model: LinearSVC" in rendered


def test_autonlp_exploration_classification(sentiment_corpus: pd.DataFrame) -> None:
    """Verify AutoNLP autonomous trial exploration, leaderboard creation, and champion selection."""
    engine = AutoNLP(
        max_trials=4,
        optimize_for="balanced",
        include_embeddings=True,
        random_state=42,
    )

    champion, leaderboard = engine.fit(
        data=sentiment_corpus,
        text_column="text",
        target_column="label",
    )

    assert isinstance(champion, NLPPredictorProtocol)
    assert leaderboard is not None
    assert len(leaderboard.successful_trials) >= 2

    # Verify champion predictor functions
    preds = champion.predict(["Loved the build quality and fast shipping!"])
    assert len(preds) == 1
    assert preds[0] in ("pos", "neg")

    if champion.has_proba:
        probas = champion.predict_proba(["Loved the build quality!"])
        assert probas.shape[1] == 2


def test_autonlp_multi_objective_modes(sentiment_corpus: pd.DataFrame) -> None:
    """Verify AutoNLP behavior under different optimization criteria."""
    # 1. Accuracy focused
    engine_acc = AutoNLP(max_trials=3, optimize_for="accuracy", random_state=42)
    _, board_acc = engine_acc.fit(sentiment_corpus, text_column="text", target_column="label")
    for t in board_acc.successful_trials:
        assert t.composite_score == t.primary_metric_score

    # 2. Latency focused
    engine_lat = AutoNLP(max_trials=3, optimize_for="latency", random_state=42)
    _, board_lat = engine_lat.fit(sentiment_corpus, text_column="text", target_column="label")
    for t in board_lat.successful_trials:
        assert t.composite_score != t.primary_metric_score


def test_fit_nlp_top_level_entrypoint(sentiment_corpus: pd.DataFrame) -> None:
    """Verify top-level functional fit_nlp helper."""
    predictor, leaderboard = fit_nlp(
        data=sentiment_corpus,
        text_column="text",
        target_column="label",
        max_trials=3,
        random_state=42,
    )

    assert isinstance(predictor, NLPPredictorProtocol)
    assert len(leaderboard.successful_trials) > 0


def test_autonlp_regression_exploration() -> None:
    """Verify AutoNLP search on text regression tasks."""
    df = pd.DataFrame(
        {
            "review": [
                "Completely broken and useless, worst item ever.",
                "Terrible experience, broke quickly.",
                "Average product, does the basic job.",
                "Good quality, satisfied with purchase.",
                "Outstanding excellence, absolute perfection 10/10!",
            ],
            "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    engine = AutoNLP(max_trials=3, random_state=42)
    predictor, leaderboard = engine.fit(
        data=df,
        text_column="review",
        target_column="rating",
        task_type="text_regression",
    )

    assert predictor.task_type == "text_regression"
    assert leaderboard.champion_trial is not None

    preds = predictor.predict(["Absolute perfection, loved it!"])
    assert len(preds) == 1
    assert isinstance(float(preds[0]), float)
