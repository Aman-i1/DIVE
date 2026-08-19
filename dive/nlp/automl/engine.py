"""AutoNLP Autonomous Exploration & Model Selection Engine - `dive/nlp/automl/engine.py`.

Orchestrates multi-candidate trial execution across representations, classical baselines,
dense embeddings, and deep neural architectures with multi-objective trade-off balancing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from dive.nlp.automl.leaderboard import NLPLeaderboard
from dive.nlp.automl.trial import NLPTrial
from dive.nlp.config import NLPConfig
from dive.nlp.data.dataset import NLPDataset
from dive.nlp.evaluation.evaluator import NLPEvaluator
from dive.nlp.exceptions import TextDataError
from dive.nlp.features import build_representation
from dive.nlp.inference.predictor import NLPPredictor
from dive.nlp.models.baselines import build_baseline_model
from dive.nlp.pipeline import NLPPipeline
from dive.nlp.preprocessing.preprocessor import NLPPreprocessor
from dive.nlp.transformers.estimator import TransformerClassifier, TransformerRegressor
from dive.utils.optional import is_available


class AutoNLP:
    """Automated Machine Learning Engine for Natural Language Processing."""

    def __init__(
        self,
        config: Optional[NLPConfig] = None,
        metric: Optional[str] = None,
        max_trials: int = 10,
        optimize_for: str = "balanced",  # balanced, accuracy, latency
        include_embeddings: bool = True,
        include_transformers: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.config = config
        self.metric = metric
        self.max_trials = max_trials
        self.optimize_for = optimize_for
        self.include_embeddings = include_embeddings
        self.include_transformers = include_transformers
        self.test_size = test_size
        self.random_state = random_state

        self.leaderboard: Optional[NLPLeaderboard] = None
        self.champion_predictor: Optional[NLPPredictor] = None

    def _generate_candidate_space(self, task_type: str) -> List[Tuple[str, str]]:
        """Generate candidate (representation_type, model_name) pairings."""
        if task_type in ("text_regression", "regression"):
            candidates = [
                ("tfidf", "RidgeRegression"),
                ("char_ngrams", "RidgeRegression"),
                ("word_char_union", "RidgeRegression"),
                ("bm25", "RidgeRegression"),
            ]
            if self.include_embeddings:
                candidates.append(("embedding", "RidgeRegression"))
            return candidates[: self.max_trials]

        # Classification candidate search space
        candidates = [
            ("tfidf", "LogisticRegression"),
            ("tfidf", "LinearSVC"),
            ("tfidf", "MultinomialNB"),
            ("char_ngrams", "LogisticRegression"),
            ("char_ngrams", "LinearSVC"),
            ("word_char_union", "LogisticRegression"),
            ("word_char_union", "LinearSVC"),
            ("bm25", "LogisticRegression"),
            ("bm25", "MultinomialNB"),
        ]
        if self.include_embeddings:
            candidates.append(("embedding", "LogisticRegression"))
            candidates.append(("embedding", "LinearSVC"))

        if self.include_transformers and is_available("transformers") and is_available("torch"):
            candidates.append(("transformer_raw", "distilbert"))

        return candidates[: self.max_trials]

    def fit(
        self,
        data: Union[NLPDataset, str, Path, pd.DataFrame, Sequence[Dict[str, Any]]],
        target_column: Optional[str] = None,
        text_column: Optional[str] = None,
        task_type: str = "text_classification",
    ) -> Tuple[NLPPredictor, NLPLeaderboard]:
        """Explore candidate space, evaluate trials, and return winning champion predictor."""
        # 1. Dataset Resolution
        if isinstance(data, NLPDataset):
            ds = data
        elif isinstance(data, (str, Path)):
            ds = NLPDataset.from_file(
                file_path=data,
                text_column=text_column or (self.config.text_column if self.config else None),
                target_column=target_column or (self.config.target_column if self.config else None),
            )
        elif isinstance(data, pd.DataFrame):
            ds = NLPDataset.from_dataframe(
                df=data,
                text_column=text_column or (self.config.text_column if self.config else None),
                target_column=target_column or (self.config.target_column if self.config else None),
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
            raise TextDataError("Dataset contains no target labels. Cannot run AutoNLP supervised search.")

        actual_task = self.config.task if self.config else task_type
        actual_text_col = text_column or (self.config.text_column if self.config else "text")
        actual_target_col = target_column or (self.config.target_column if self.config else "label")

        primary_metric = self.metric or ("macro_f1" if actual_task == "text_classification" else "rmse")
        self.leaderboard = NLPLeaderboard(primary_metric=primary_metric)

        # 2. Holdout Validation Split
        train_ds, test_ds = ds.split(
            test_size=self.test_size,
            stratify=(actual_task in ("text_classification", "classification")),
            random_state=self.random_state,
        )

        candidates = self._generate_candidate_space(actual_task)
        evaluator = NLPEvaluator(task_type=actual_task)

        # 3. Candidate Exploration Loop
        for trial_idx, (rep_type, model_name) in enumerate(candidates, start=1):
            trial = NLPTrial(
                trial_id=trial_idx,
                representation_type=rep_type,
                model_name=model_name,
                primary_metric_name=primary_metric,
            )

            try:
                t0_train = time.perf_counter()

                # Build representation and model
                if rep_type == "transformer_raw":
                    preprocessor = NLPPreprocessor.raw()
                    rep_engine = None
                    estimator = TransformerClassifier(model_name=model_name, epochs=2)
                else:
                    preprocessor = NLPPreprocessor(config=self.config.preprocessing if self.config else None)
                    rep_engine = build_representation(
                        config=self.config.representation if self.config else None,
                        representation_type=rep_type,
                    )
                    estimator = build_baseline_model(
                        model_name=model_name,
                        problem_type=actual_task,
                        random_state=self.random_state,
                    )

                pipe = NLPPipeline(
                    estimator=estimator,
                    preprocessor=preprocessor,
                    representation=rep_engine,
                    task_type=actual_task,
                    model_name=model_name,
                )
                pipe.fit(train_ds.texts, train_ds.labels)  # type: ignore
                trial.train_time_ms = (time.perf_counter() - t0_train) * 1000.0

                # Evaluate Holdout Predictions
                y_pred = pipe.predict(test_ds.texts)
                y_proba = None
                if actual_task in ("text_classification", "classification") and hasattr(pipe, "predict_proba"):
                    try:
                        y_proba = pipe.predict_proba(test_ds.texts)
                    except Exception:
                        pass

                metrics = evaluator.evaluate(
                    y_true=test_ds.labels,  # type: ignore
                    y_pred=y_pred,
                    y_proba=y_proba,
                    class_names=pipe.class_names,
                )
                trial.metrics = metrics
                metric_val = float(metrics.get(primary_metric, metrics.get("accuracy", 0.0)))
                trial.primary_metric_score = metric_val

                # Micro-latency benchmark
                bench_samples = test_ds.texts[: min(10, len(test_ds.texts))]
                t0_inf = time.perf_counter()
                for _ in range(20):
                    _ = pipe.predict(bench_samples)
                trial.inference_latency_ms = ((time.perf_counter() - t0_inf) / (20 * len(bench_samples))) * 1000.0

                # Multi-Objective Composite Scoring
                # Speed factor S between 0.0 and 1.0 (faster latency yields higher S)
                speed_factor = 1.0 / (1.0 + trial.inference_latency_ms / 5.0)

                # For regression, lower RMSE is better -> invert for composite
                if actual_task in ("text_regression", "regression"):
                    normalized_acc = 1.0 / (1.0 + metric_val)
                else:
                    normalized_acc = metric_val

                if self.optimize_for == "accuracy":
                    trial.composite_score = normalized_acc
                elif self.optimize_for == "latency":
                    trial.composite_score = 0.35 * normalized_acc + 0.65 * speed_factor
                else:  # balanced
                    trial.composite_score = 0.75 * normalized_acc + 0.25 * speed_factor

                trial.status = "SUCCESS"

            except Exception as e:
                trial.status = "FAILED"
                trial.error_msg = str(e)

            self.leaderboard.add_trial(trial)

        # 4. Champion Retraining on Full Dataset
        champion = self.leaderboard.champion_trial
        if champion is None:
            raise RuntimeError("AutoNLP search failed to produce any successful candidate pipeline.")

        # Fit winning champion configuration on the complete dataset
        if champion.representation_type == "transformer_raw":
            champ_prep = NLPPreprocessor.raw()
            champ_rep = None
            champ_estimator = TransformerClassifier(model_name=champion.model_name, epochs=3)
        else:
            champ_prep = NLPPreprocessor(config=self.config.preprocessing if self.config else None)
            champ_rep = build_representation(
                config=self.config.representation if self.config else None,
                representation_type=champion.representation_type,
            )
            champ_estimator = build_baseline_model(
                model_name=champion.model_name,
                problem_type=actual_task,
                random_state=self.random_state,
            )

        champ_pipeline = NLPPipeline(
            estimator=champ_estimator,
            preprocessor=champ_prep,
            representation=champ_rep,
            task_type=actual_task,
            model_name=champion.model_name,
        )
        champ_pipeline.fit(ds.texts, ds.labels)  # type: ignore

        self.champion_predictor = NLPPredictor(
            pipeline=champ_pipeline,
            model_name=f"AutoNLP_{champion.model_name}",
            text_column=actual_text_col,
            target_column=actual_target_col,
            task_type=actual_task,
            metrics=champion.metrics,
            dataset_name=ds.name,
        )

        return self.champion_predictor, self.leaderboard


def fit_nlp(
    data: Union[NLPDataset, str, Path, pd.DataFrame, Sequence[Dict[str, Any]]],
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
    task_type: str = "text_classification",
    metric: Optional[str] = None,
    max_trials: int = 10,
    optimize_for: str = "balanced",
    random_state: int = 42,
) -> Tuple[NLPPredictor, NLPLeaderboard]:
    """Autonomous top-level user entrypoint for AutoNLP."""
    engine = AutoNLP(
        metric=metric,
        max_trials=max_trials,
        optimize_for=optimize_for,
        random_state=random_state,
    )
    return engine.fit(
        data=data,
        target_column=target_column,
        text_column=text_column,
        task_type=task_type,
    )
