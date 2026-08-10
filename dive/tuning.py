"""Stage 5 - Optuna hyperparameter search.

Ported from the ``OptunaOptimizer`` class in ``Automatic Machine Learning.ipynb``.
The per-model search spaces, the TPE sampler, the median pruner, and the wall-clock
budget behaviour are unchanged.

Optuna is optional. When it is not installed, :meth:`OptunaOptimizer.tune` returns
the untuned pipeline and a ``None`` score, so callers need no special-casing.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Tuple

from sklearn.model_selection import cross_val_score

from dive.model_zoo import _BoostingWrapper
from dive.utils.optional import load_optional

# Trials are also bounded by time; this caps the count on very fast models.
DEFAULT_N_TRIALS = 50


class OptunaOptimizer:
    """Search hyperparameters for one pipeline at a time, under a time budget."""

    def __init__(
        self,
        problem_type: str,
        cv: Any,
        scoring: str,
        time_budget_per_model: float = 180,
        n_trials: int = DEFAULT_N_TRIALS,
        random_state: int = 42,
    ) -> None:
        self.problem_type = problem_type
        self.cv = cv
        self.scoring = scoring
        self.time_budget = time_budget_per_model
        self.n_trials = n_trials
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _param_space(self, trial: Any, name: str) -> Dict[str, Any]:
        """Return the search space for a model, keyed by pipeline param path."""
        if "RandomForest" in name or "ExtraTrees" in name:
            return {
                "model__n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "model__max_depth": trial.suggest_int("max_depth", 3, 25),
                "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "model__max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5, 0.7]
                ),
            }
        if "HistGBM" in name:
            return {
                "model__max_iter": trial.suggest_int("max_iter", 100, 800),
                "model__max_depth": trial.suggest_int("max_depth", 2, 12),
                "model__learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.3, log=True
                ),
                "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
                "model__l2_regularization": trial.suggest_float(
                    "l2_regularization", 1e-6, 10, log=True
                ),
                "model__max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            }
        if "LogisticRegression" in name:
            return {"model__C": trial.suggest_float("C", 1e-3, 100, log=True)}
        if "RidgeClassifier" in name:
            return {"model__alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)}
        if name == "Ridge":
            return {"model__alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)}
        if "Lasso" in name:
            return {"model__alpha": trial.suggest_float("alpha", 1e-4, 10, log=True)}
        if "ElasticNet" in name:
            return {
                "model__alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
                "model__l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
        if "MLP" in name:
            return {
                "model__alpha": trial.suggest_float("alpha", 1e-5, 0.1, log=True),
                "model__learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 0.01, log=True
                ),
            }
        if "AdaBoost" in name:
            return {
                "model__n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "model__learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 2.0, log=True
                ),
            }
        if "KNN" in name:
            return {"model__n_neighbors": trial.suggest_int("n_neighbors", 3, 30)}
        return {}

    # ------------------------------------------------------------------
    def tune(
        self, pipeline: Any, name: str, X: Any, y: Any
    ) -> Tuple[Any, Optional[float]]:
        """Return ``(best_pipeline, best_cv_score)``.

        Returns the input pipeline and ``None`` when Optuna is unavailable, when
        the model has no search space, or when every trial failed - the caller
        then keeps the untuned fit.
        """
        optuna = load_optional("optuna")
        if optuna is None or isinstance(pipeline, _BoostingWrapper):
            return pipeline, None

        started = time.time()

        def objective(trial: Any) -> float:
            if time.time() - started > self.time_budget:
                raise optuna.TrialPruned()
            params = self._param_space(trial, name)
            if not params:
                raise optuna.TrialPruned()
            try:
                candidate = copy.deepcopy(pipeline)
                candidate.set_params(**params)
                scores = cross_val_score(
                    candidate,
                    X,
                    y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=-1,
                    error_score="raise",
                )
                return float(scores.mean())
            except Exception:
                raise optuna.TrialPruned()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(
            objective,
            n_trials=min(self.n_trials, max(5, int(self.time_budget / 3))),
            timeout=self.time_budget,
            catch=(Exception,),
            show_progress_bar=False,
        )

        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            return pipeline, None

        best = copy.deepcopy(pipeline)
        best.set_params(**{f"model__{k}": v for k, v in study.best_params.items()})
        return best, float(study.best_value)
