"""Stage 4 - the model zoo.

Ported from the ``ModelZoo`` class in ``Automatic Machine Learning.ipynb``.
Composition rules are unchanged, including GPU auto-detection, version-aware
XGBoost construction, and the conditional inclusion of KNN / AdaBoost.

The mode dimension now has a *distinct third profile* for ``fast``: a small
subset of models with downsized capacities, no boosters, and no KNN. ``fast``
exists to finish quickly on large datasets or as a CI smoke test; ``balanced``
and ``competition`` keep the full competition-grade zoo.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from dive.utils.optional import is_available, load_optional, version_tuple

# 2.x removed `early_stopping_rounds` from fit() in favour of constructor callbacks.
XGB_MIN_VERSION_FOR_CALLBACKS = (2, 0)

_KNN_CAPACITY_CAP = 50_000
# Model names that are fitted through _BoostingWrapper with an eval-set.
BOOSTING_MODELS = ("XGBoost", "LightGBM", "CatBoost")


def detect_gpu() -> bool:
    """NVIDIA GPU availability, cached for the process lifetime."""
    return _GPU_CACHE


def _cached_gpu_detection() -> bool:
    from dive.utils.optional import detect_gpu as _detect

    return _detect()


_GPU_CACHE: bool = _cached_gpu_detection()


# ----------------------------------------------------------------------
class _BoostingWrapper(BaseEstimator):
    """Adapter that fits a boosting model on transformed features.

    Boosters are fitted on pre-transformed data (numeric scaling + one-hot)
    with ``eval_set`` early stopping, so they live behind a wrapper that owns
    the transformation pipeline and exposes the usual sklearn interface.
    """

    def __init__(self, pre_pipe: Any = None, model: Any = None,
                 problem_type: str = "classification") -> None:
        self.pre_pipe = pre_pipe
        self.model = model
        self.problem_type = problem_type

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "_BoostingWrapper":
        X_t = self.pre_pipe.fit_transform(X, y)
        self.model.fit(X_t, y, **kwargs)
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(self.pre_pipe.transform(X))

    def predict_proba(self, X: Any) -> Any:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(self.pre_pipe.transform(X))
        raise AttributeError(
            f"{type(self.model).__name__} has no predict_proba."
        )

    def transform(self, X: Any) -> Any:
        return self.pre_pipe.transform(X)


class _RidgeClassifierWrapper(BaseEstimator, ClassifierMixin):
    """RidgeClassifier with a softmax probability transform.

    sklearn's RidgeClassifier has no ``predict_proba``, which blocks ROC-AUC and
    log-loss and disqualifies it from stacking. This converts the decision
    function into probabilities via a softmax over the margin.

    Inheriting from ``BaseEstimator``/``ClassifierMixin`` is required, not
    cosmetic: scikit-learn >= 1.6 resolves estimator capabilities through
    ``__sklearn_tags__``, which only exists on ``BaseEstimator``. Without it,
    ``Pipeline.fit`` raises and this model silently drops out of every run.
    """

    def __init__(self, alpha: float = 1.0, class_weight: Any = None,
                 random_state: Any = None) -> None:
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "_RidgeClassifierWrapper":
        from sklearn.linear_model import RidgeClassifier

        self.clf_ = RidgeClassifier(
            alpha=self.alpha, class_weight=self.class_weight
        )
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X: Any) -> Any:
        return self.clf_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        decision = self.clf_.decision_function(X)
        if decision.ndim == 1:
            decision = np.c_[-decision, decision]
        decision = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(decision)
        return exp / exp.sum(axis=1, keepdims=True)


def _xgb_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise XGBoost constructor kwargs for the installed major version.

    XGBoost 2.0 moved early stopping from ``fit()`` to a constructor callback and
    introduced the unified ``device`` parameter. On 1.x, ``device`` is unknown and
    would raise, so it is translated away.
    """
    early_stopping_rounds = kwargs.pop("early_stopping_rounds", 50)
    if version_tuple("xgboost") >= XGB_MIN_VERSION_FOR_CALLBACKS:
        from xgboost.callback import EarlyStopping

        kwargs["callbacks"] = [
            EarlyStopping(rounds=early_stopping_rounds, save_best=True)
        ]
    else:
        device = kwargs.pop("device", "cpu")
        if device == "gpu":
            kwargs["tree_method"] = "gpu_hist"
    return kwargs


def _make_xgb_classifier(**kwargs: Any) -> Any:
    """XGBClassifier that works on both XGBoost 1.x and 2.x+."""
    from xgboost import XGBClassifier

    return XGBClassifier(**_xgb_kwargs(kwargs))


def _make_xgb_regressor(**kwargs: Any) -> Any:
    """XGBRegressor that works on both XGBoost 1.x and 2.x+."""
    from xgboost import XGBRegressor

    return XGBRegressor(**_xgb_kwargs(kwargs))


def xgb_fit_kwargs(X_val: Any, y_val: Any, early_stopping_rounds: int = 50) -> Dict[str, Any]:
    """Return version-correct ``fit()`` kwargs for XGBoost early stopping."""
    if version_tuple("xgboost") >= XGB_MIN_VERSION_FOR_CALLBACKS:
        # Early stopping already lives on the constructor callback.
        return {"eval_set": [(X_val, y_val)], "verbose": False}
    return {
        "eval_set": [(X_val, y_val)],
        "early_stopping_rounds": early_stopping_rounds,
        "verbose": False,
    }


def strip_early_stopping(model: Any) -> None:
    """Disable early stopping on a model that will be fitted without an eval set.

    Needed by the stacking engine: out-of-fold fitting has no held-out eval set,
    and an XGBoost EarlyStopping callback raises without one. Clearing only
    ``early_stopping_rounds`` (as the notebook did) misses the 2.x callback path,
    which silently dropped every booster out of the stack.
    """
    for attr in ("early_stopping_rounds", "n_iter_no_change"):
        try:
            if hasattr(model, attr):
                setattr(model, attr, None)
        except Exception:
            pass
    # XGBoost >= 2.0 carries early stopping as a constructor callback.
    try:
        if getattr(model, "callbacks", None):
            model.set_params(callbacks=None)
    except Exception:
        try:
            model.callbacks = None
        except Exception:
            pass
    # CatBoost uses overfitting-detector wait rounds.
    try:
        if hasattr(model, "get_params") and "od_wait" in (model.get_params() or {}):
            model.set_params(od_wait=None)
    except Exception:
        pass


# ----------------------------------------------------------------------
class ModelZoo:
    """Build the mode-appropriate dictionary of untrained models."""

    def __init__(
        self,
        problem_type: str,
        profile: Dict[str, Any],
        mode: str = "balanced",
        random_state: int = 42,
        use_balanced: bool = True,
    ) -> None:
        self.problem_type = problem_type
        self.profile = profile
        self.mode = mode
        self.random_state = random_state
        self.use_balanced = use_balanced
        self.n_samples = int(profile.get("n_samples", 0))

    # ------------------------------------------------------------------
    def get_models(self) -> Dict[str, Any]:
        if self.mode == "fast":
            return self._fast_models()
        return self._full_models()

    # ------------------------------------------------------------------
    def _fast_models(self) -> Dict[str, Any]:
        """Small, fast subset - the defining trait of ``fast`` mode.

        No boosters (they carry a several-second startup cost each), no KNN,
        no AdaBoost. RandomForest/ExtraTrees run at 100 trees instead of 300.
        """
        rs = self.random_state
        models: Dict[str, Any] = {}
        if self.problem_type == "classification":
            models["LogisticRegression"] = LogisticRegression(
                max_iter=500, random_state=rs
            )
            models["RandomForest"] = RandomForestClassifier(
                n_estimators=100, random_state=rs, class_weight=None, n_jobs=-1
            )
            models["HistGBM"] = HistGradientBoostingClassifier(
                max_iter=150,
                random_state=rs,
                early_stopping=True,
                n_iter_no_change=10,
            )
        else:
            models["LinearRegression"] = LinearRegression(n_jobs=-1)
            models["Ridge"] = Ridge(random_state=rs)
            models["RandomForest"] = RandomForestRegressor(
                n_estimators=100, random_state=rs, n_jobs=-1
            )
        return models

    # ------------------------------------------------------------------
    def _full_models(self) -> Dict[str, Any]:
        """The balanced/competition zoo: the notebook's composition unchanged."""
        rs = self.random_state
        pt = self.problem_type
        n = self.n_samples
        class_weight = (
            "balanced"
            if (self.use_balanced and self.profile.get("is_imbalanced"))
            else None
        )
        gpu = "gpu" if detect_gpu() else "cpu"
        models: Dict[str, Any] = {}

        if pt == "classification":
            models["LogisticRegression"] = LogisticRegression(
                max_iter=2000, random_state=rs, class_weight=class_weight
            )
            models["RidgeClassifier"] = _RidgeClassifierWrapper(
                alpha=1.0, class_weight=class_weight
            )
            models["RandomForest"] = RandomForestClassifier(
                n_estimators=300, random_state=rs, class_weight=class_weight, n_jobs=-1
            )
            models["ExtraTrees"] = ExtraTreesClassifier(
                n_estimators=300, random_state=rs, class_weight=class_weight, n_jobs=-1
            )
            hist_kwargs: Dict[str, Any] = {
                "max_iter": 500,
                "random_state": rs,
                "early_stopping": True,
                "n_iter_no_change": 20,
            }
            if class_weight and _supports_param(
                HistGradientBoostingClassifier, "class_weight"
            ):
                hist_kwargs["class_weight"] = class_weight
            models["HistGBM"] = HistGradientBoostingClassifier(**hist_kwargs)
            models["MLP"] = MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=500,
                random_state=rs, early_stopping=True,
            )

            if self.mode in ("balanced", "competition"):
                if n <= _KNN_CAPACITY_CAP:
                    models["KNN"] = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
                if is_available("xgboost"):
                    models["XGBoost"] = _make_xgb_classifier(
                        early_stopping_rounds=50,
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=rs,
                        n_jobs=-1,
                        device=gpu,
                        verbosity=0,
                        scale_pos_weight=self.profile.get("imbalance_ratio") or 1.0,
                    )
                lightgbm = load_optional("lightgbm")
                if lightgbm is not None:
                    models["LightGBM"] = lightgbm.LGBMClassifier(
                        n_estimators=500, learning_rate=0.05, num_leaves=63,
                        subsample=0.8, colsample_bytree=0.8,
                        class_weight=class_weight, random_state=rs,
                        n_jobs=-1, verbosity=-1,
                    )
                catboost = load_optional("catboost")
                if catboost is not None:
                    models["CatBoost"] = catboost.CatBoostClassifier(
                        iterations=500, learning_rate=0.05, depth=6,
                        eval_metric="Accuracy", random_seed=rs, verbose=0,
                        auto_class_weights="Balanced" if class_weight else None,
                        task_type="GPU" if detect_gpu() else "CPU",
                    )
            if self.mode == "competition":
                models["AdaBoost"] = AdaBoostClassifier(n_estimators=300, random_state=rs)

        else:
            models["LinearRegression"] = LinearRegression(n_jobs=-1)
            models["Ridge"] = Ridge(random_state=rs)
            models["Lasso"] = Lasso(random_state=rs, max_iter=5000)
            models["ElasticNet"] = ElasticNet(random_state=rs, max_iter=5000)
            models["RandomForest"] = RandomForestRegressor(
                n_estimators=300, random_state=rs, n_jobs=-1
            )
            models["ExtraTrees"] = ExtraTreesRegressor(
                n_estimators=300, random_state=rs, n_jobs=-1
            )
            models["HistGBM"] = HistGradientBoostingRegressor(
                max_iter=500, random_state=rs, early_stopping=True, n_iter_no_change=20
            )
            models["MLP"] = MLPRegressor(
                hidden_layer_sizes=(256, 128), max_iter=500,
                random_state=rs, early_stopping=True,
            )

            if self.mode in ("balanced", "competition"):
                if n <= _KNN_CAPACITY_CAP:
                    models["KNN"] = KNeighborsRegressor(n_neighbors=7, n_jobs=-1)
                if is_available("xgboost"):
                    models["XGBoost"] = _make_xgb_regressor(
                        early_stopping_rounds=50,
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="rmse",
                        random_state=rs,
                        n_jobs=-1,
                        device=gpu,
                        verbosity=0,
                    )
                lightgbm = load_optional("lightgbm")
                if lightgbm is not None:
                    models["LightGBM"] = lightgbm.LGBMRegressor(
                        n_estimators=500, learning_rate=0.05, num_leaves=63,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=rs, n_jobs=-1, verbosity=-1,
                    )
                catboost = load_optional("catboost")
                if catboost is not None:
                    models["CatBoost"] = catboost.CatBoostRegressor(
                        iterations=500, learning_rate=0.05, depth=6,
                        random_seed=rs, verbose=0,
                        task_type="GPU" if detect_gpu() else "CPU",
                    )
            if self.mode == "competition":
                models["AdaBoost"] = AdaBoostRegressor(n_estimators=300, random_state=rs)

        return models


def _supports_param(estimator_class: Any, name: str) -> bool:
    """True when ``estimator_class.__init__`` accepts the named keyword.

    ``HistGradientBoosting*`` only gained ``class_weight`` in scikit-learn 1.5,
    so passing it unconditionally breaks older-but-supported installs.
    """
    import inspect

    try:
        return name in inspect.signature(estimator_class.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-implemented __init__
        return False
