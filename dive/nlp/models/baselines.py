"""Classical CPU Baseline NLP Models - `dive/nlp/models/baselines.py`.

Provides fast, high-accuracy baseline estimators for text tasks:
- Logistic Regression (L2-regularized with balanced class weighting)
- Linear Support Vector Machine (LinearSVC with Platt probability calibration)
- Multinomial Naive Bayes (smooth alpha prior)
- Ridge Regression (for continuous text regression targets)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from dive.nlp.exceptions import NLPModelError


BASELINE_MODELS = (
    "LogisticRegression",
    "LinearSVC",
    "MultinomialNB",
    "RidgeRegression",
)


def build_baseline_model(
    model_name: str = "LogisticRegression",
    problem_type: str = "classification",
    random_state: int = 42,
    calibrate_svc: bool = True,
    **kwargs: Any,
) -> Any:
    """Instantiate a configured scikit-learn baseline model for NLP."""
    name = model_name.strip()

    if problem_type in ("text_regression", "regression"):
        if name in ("RidgeRegression", "Ridge", "linear"):
            alpha = kwargs.get("alpha", 1.0)
            return Ridge(alpha=alpha, random_state=random_state)
        raise NLPModelError(
            f"Unsupported baseline regression model: '{name}'.",
            "Supported regression models: RidgeRegression",
        )

    # Classification baselines
    if name in ("LogisticRegression", "LogReg", "lr"):
        max_iter = kwargs.get("max_iter", 1000)
        c_val = kwargs.get("C", 1.0)
        return LogisticRegression(
            C=c_val,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
            solver="liblinear",
        )

    if name in ("LinearSVC", "SVM", "svm"):
        c_val = kwargs.get("C", 1.0)
        base_svc = LinearSVC(
            C=c_val,
            class_weight="balanced",
            random_state=random_state,
            dual="auto",
            max_iter=2000,
        )
        if calibrate_svc:
            # Calibrate linear SVM decision scores to produce well-behaved probabilities
            return CalibratedClassifierCV(estimator=base_svc, cv=3)
        return base_svc

    if name in ("MultinomialNB", "NaiveBayes", "nb"):
        alpha = kwargs.get("alpha", 1.0)
        return MultinomialNB(alpha=alpha)

    raise NLPModelError(
        f"Unsupported baseline model name: '{name}'.",
        f"Supported models: {', '.join(BASELINE_MODELS)}",
    )
