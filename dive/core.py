"""The Dive orchestrator and its evaluator.

Ported from ``Automatic Machine Learning.ipynb``. The pipeline order is
unchanged: profile -> engineer features -> split -> build preprocessor ->
train the zoo -> tune the top N -> stack the top N -> pick the best.

Changes made for CLI/library use, none of which alter modelling behaviour:

* Progress is emitted through :class:`dive.utils.logging.Console` so training
  reports each model and its elapsed-vs-budget position instead of going silent.
* ``predict`` reuses the pickled, already-fitted FeatureEngineer - it is never
  refitted at inference time.
* Fitted artifacts carry a schema record so predict-time mismatches fail loudly.
* ``save``/``load`` round-trip through :mod:`dive.utils.io`, which converts
  pickle failures into clear messages.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from dive.data_intelligence import DataIntelligence
from dive.ensembling import StackingEngine, _StackWrapper
from dive.exceptions import SchemaError, TrainingError
from dive.feature_engineering import FeatureEngineer
from dive.model_zoo import (
    BOOSTING_MODELS,
    ModelZoo,
    _BoostingWrapper,
    xgb_fit_kwargs,
)
from dive.tuning import OptunaOptimizer
from dive.utils.io import load_pickle, save_pickle
from dive.utils.logging import Console, get_console
from dive.utils.optional import is_available

MODES = ("fast", "balanced", "competition")
# Below this many seconds remaining, no new model is started.
MIN_SECONDS_PER_MODEL = 15
# PCA engages automatically above this many engineered features.
PCA_FEATURE_THRESHOLD = 50


# ----------------------------------------------------------------------
def build_preprocessor(
    X: pd.DataFrame,
    variance_threshold: bool = True,
    use_pca: bool = False,
    pca_variance: float = 0.95,
    random_state: int = 42,
) -> Pipeline:
    """Impute + scale numerics, impute + one-hot categoricals, then optionally PCA."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=20,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise TrainingError(
            "No usable feature columns remain after feature engineering.",
            "Every column was constant, ID-like, or empty. Check the input data.",
        )

    steps: List[Any] = [("ct", ColumnTransformer(transformers, remainder="drop"))]
    if variance_threshold:
        steps.append(("var_thresh", VarianceThreshold(threshold=0.0)))
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=random_state)))
    return Pipeline(steps)


# ----------------------------------------------------------------------
class Evaluator:
    """Compute the leaderboard row for one fitted model."""

    def __init__(self, problem_type: str) -> None:
        self.problem_type = problem_type

    def evaluate(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: Any,
        name: str = "model",
        n_classes: int = 2,
        skip_cv: bool = False,
    ) -> Dict[str, Any]:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        cv_mean, cv_std, cv_scores = np.nan, np.nan, []
        if not skip_cv:
            try:
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=self.scorer,
                    n_jobs=-1,
                    error_score=np.nan,
                )
                cv_scores = [float(s) for s in scores]
                cv_mean = float(np.nanmean(scores))
                cv_std = float(np.nanstd(scores))
            except Exception:
                cv_mean, cv_std, cv_scores = np.nan, np.nan, []

        if self.problem_type == "classification":
            auc, logloss = np.nan, np.nan
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    if n_classes == 2 and proba.shape[1] == 2:
                        auc = float(roc_auc_score(y_test, proba[:, 1]))
                        logloss = float(log_loss(y_test, proba))
                    elif proba.shape[1] > 2:
                        auc = float(
                            roc_auc_score(
                                y_test, proba, multi_class="ovr", average="weighted"
                            )
                        )
                        logloss = float(log_loss(y_test, proba))
            except Exception:
                # AUC/log-loss are informational; a model that cannot produce
                # calibrated probabilities still competes on accuracy.
                pass
            return {
                "Model": name,
                "Train Accuracy": float(accuracy_score(y_train, train_pred)),
                "Test Accuracy": float(accuracy_score(y_test, test_pred)),
                "CV Score (mean)": cv_mean,
                "CV Score (std)": cv_std,
                "Test F1 (weighted)": float(
                    f1_score(y_test, test_pred, average="weighted", zero_division=0)
                ),
                "Test ROC-AUC": auc,
                "Test LogLoss": logloss,
                "Balanced Accuracy": float(balanced_accuracy_score(y_test, test_pred)),
                "_cv_scores": cv_scores,
            }

        return {
            "Model": name,
            "Train R2": float(r2_score(y_train, train_pred)),
            "Test R2": float(r2_score(y_test, test_pred)),
            "CV Score (mean)": cv_mean,
            "CV Score (std)": cv_std,
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "Test MAE": float(mean_absolute_error(y_test, test_pred)),
            "Test MAPE": self._mape(y_test, test_pred),
            "_cv_scores": cv_scores,
        }

    @property
    def scorer(self) -> str:
        return "accuracy" if self.problem_type == "classification" else "r2"

    @staticmethod
    def _mape(y_true: Any, y_pred: Any, eps: float = 1e-8) -> float:
        """Mean absolute percentage error, skipping near-zero actuals."""
        actual = np.asarray(y_true, dtype=float)
        predicted = np.asarray(y_pred, dtype=float)
        mask = np.abs(actual) > eps
        if not mask.any():
            return float("nan")
        return float(
            np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        )


# ----------------------------------------------------------------------
class Dive:
    """End-to-end tabular Dive: profile, engineer, train, tune, stack, select."""

    def __init__(
        self,
        target: Optional[str] = None,
        mode: str = "balanced",
        time_budget: float = 1800,
        problem_type: Optional[str] = None,
        test_size: float = 0.2,
        cv_folds: Optional[int] = None,
        random_state: int = 42,
        time_series: bool = False,
        scoring: Optional[str] = None,
        use_pca: Any = "auto",
        tune_top_n: int = 3,
        stack_top_n: int = 5,
        outlier_clip: bool = True,
        verbose: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        self.target = target
        self.mode = mode
        self.time_budget = time_budget
        self.problem_type = problem_type
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.time_series = time_series
        self.scoring = scoring
        self.use_pca = use_pca
        self.tune_top_n = tune_top_n
        self.stack_top_n = stack_top_n
        self.outlier_clip = outlier_clip
        self.verbose = verbose
        self.console = console or get_console(verbose=verbose)

        self.profile_: Optional[Dict[str, Any]] = None
        self.feature_engineer_: Optional[FeatureEngineer] = None
        self.feature_columns_: Optional[List[str]] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.results_df_: Optional[pd.DataFrame] = None
        self.fitted_pipelines_: Dict[str, Any] = {}
        self.best_model_name_: Optional[str] = None
        self.best_estimator_: Optional[Any] = None
        self.stacked_estimator_: Optional[Any] = None
        self.skipped_models_: Dict[str, str] = {}
        self.cv_scores_: Dict[str, List[float]] = {}

        self._X_train = self._X_test = None
        self._y_train = self._y_test = None
        self._preprocessor: Optional[Pipeline] = None
        self._metadata: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._primary_score_col: Optional[str] = None
        self._cv: Any = None
        self._decide_pca = False
        self._fs_step_used = False
        self._fit_seconds: Optional[float] = None

    # -- timing ---------------------------------------------------------
    def _elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    def _budget_left(self) -> float:
        return self.time_budget - self._elapsed()

    def _log(self, message: str) -> None:
        self.console.info(message)

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Dive":
        """Run the full pipeline against ``df`` and select a best model."""
        if self.mode not in MODES:
            raise TrainingError(
                f"Unknown mode '{self.mode}'.",
                f"Valid modes are: {', '.join(MODES)}.",
            )

        self._start_time = time.time()
        df = df.copy()

        if self.target is None:
            self.target = str(df.columns[-1])
            self.console.warn(
                f"No --target given; using the last column: '{self.target}'"
            )

        before = len(df)
        df = df.dropna(subset=[self.target])
        dropped = before - len(df)
        if dropped:
            self._log(f"[1/6] Dropped {dropped} row(s) with a missing target.")

        # -- 1. profile --------------------------------------------------
        self.console.step(1, 6, "Profiling dataset")
        profiler = DataIntelligence(self.target, self.random_state)
        self.profile_ = profiler.analyze(df)
        if self.problem_type is None:
            self.problem_type = self.profile_["problem_type"]
        else:
            self.profile_["problem_type"] = self.problem_type

        self.console.kv("Problem type", self.problem_type)
        self.console.kv("Rows x features", f"{self.profile_['n_samples']} x {self.profile_['n_features']}")
        self.console.kv("Mode", self.mode)
        if self.profile_["is_imbalanced"]:
            self.console.warn(
                f"Class imbalance {self.profile_['imbalance_ratio']:.1f}:1 detected - "
                "using class_weight='balanced' where supported."
            )

        # -- 2. encode target + engineer features ------------------------
        self.console.step(2, 6, "Engineering features")
        y_raw = df[self.target]
        if self.problem_type == "classification":
            self.label_encoder_ = LabelEncoder()
            y_encoded = pd.Series(
                self.label_encoder_.fit_transform(y_raw.astype(str)),
                index=y_raw.index,
                name=self.target,
            )
        else:
            y_encoded = pd.to_numeric(y_raw, errors="coerce").astype("float32")
            if y_encoded.isna().any():
                keep = y_encoded.notna()
                self.console.warn(
                    f"Dropped {int((~keep).sum())} row(s) whose target could not be "
                    "parsed as a number."
                )
                df, y_encoded = df[keep], y_encoded[keep]

        X_raw = df.drop(columns=[self.target])

        use_advanced = self.mode in ("balanced", "competition")
        self.feature_engineer_ = FeatureEngineer(
            profile=self.profile_,
            target=self.target,
            mode=self.mode,
            random_state=self.random_state,
            use_target_encoding=use_advanced and is_available("category_encoders"),
            use_freq_encoding=use_advanced,
            outlier_clip=self.outlier_clip,
        )
        X_fe = self.feature_engineer_.fit_transform(X_raw, y_encoded)
        self.feature_columns_ = X_fe.columns.tolist()
        if not self.feature_columns_:
            raise TrainingError(
                "Feature engineering removed every column.",
                "All inputs were constant or ID-like. Provide columns with signal.",
            )
        self.console.kv("Engineered features", len(self.feature_columns_))

        # -- 3. split ----------------------------------------------------
        self.console.step(3, 6, "Splitting train / holdout")
        X_train, X_test, y_train, y_test = self._split(X_fe, y_encoded)
        self._X_train, self._X_test = X_train, X_test
        self._y_train, self._y_test = y_train, y_test
        self.console.kv("Train / holdout rows", f"{len(X_train)} / {len(X_test)}")

        self._cv = self._build_cv(len(X_train), y_train)
        if self.scoring is None:
            self.scoring = (
                "accuracy" if self.problem_type == "classification" else "r2"
            )
        self._primary_score_col = (
            "Test Accuracy" if self.problem_type == "classification" else "Test R2"
        )

        # -- 4. preprocessor + model zoo ---------------------------------
        self._decide_pca = bool(
            self.use_pca is True
            or (self.use_pca == "auto" and X_train.shape[1] > PCA_FEATURE_THRESHOLD)
        )
        self._preprocessor = build_preprocessor(
            X_train,
            variance_threshold=True,
            use_pca=self._decide_pca,
            pca_variance=0.95,
            random_state=self.random_state,
        )
        if self._decide_pca:
            self._log("      PCA enabled - retaining 95% of variance.")

        feature_select_step = None
        self._fs_step_used = False
        if self.mode == "competition" and X_train.shape[1] > 10:
            score_func = (
                mutual_info_classif
                if self.problem_type == "classification"
                else mutual_info_regression
            )
            k = min(X_train.shape[1], max(10, int(X_train.shape[1] * 0.8)))
            feature_select_step = SelectKBest(score_func, k=k)
            self._fs_step_used = True

        zoo = ModelZoo(
            self.problem_type, self.profile_, self.mode, self.random_state,
            use_balanced=True,
        )
        models = zoo.get_models()

        self.console.step(4, 6, f"Training {len(models)} models")
        results = self._train_models(
            models, feature_select_step, X_train, y_train, X_test, y_test
        )

        if not results:
            raise TrainingError(
                "Every model failed to train.",
                "The last errors are listed above. This usually means the data "
                "contains a column type no model could consume, or the dataset "
                "is too small for the requested split.",
            )

        self.results_df_ = (
            pd.DataFrame(results)
            .sort_values(self._primary_score_col, ascending=False)
            .reset_index(drop=True)
        )

        # -- 5. tuning ---------------------------------------------------
        self._tune_top_models(X_train, y_train, X_test, y_test)

        # -- 6. stacking -------------------------------------------------
        self._build_stack(X_train, y_train, X_test, y_test)

        # -- select best -------------------------------------------------
        self.best_model_name_ = str(self.results_df_.iloc[0]["Model"])
        self.best_estimator_ = self.fitted_pipelines_[self.best_model_name_]
        self._fit_seconds = self._elapsed()
        self._metadata = self._build_metadata()

        best_score = float(self.results_df_.iloc[0][self._primary_score_col])
        self.console.print("")
        self.console.success(
            f"Training complete in {self._fit_seconds:.0f}s - "
            f"best model: {self.best_model_name_} "
            f"({self._primary_score_col} = {best_score:.4f})"
        )
        return self

    # ------------------------------------------------------------------
    def _split(self, X: pd.DataFrame, y: pd.Series):
        """Chronological split for time series, stratified split otherwise."""
        if self.time_series:
            cutoff = int(len(X) * (1 - self.test_size))
            cutoff = max(1, min(cutoff, len(X) - 1))
            return X.iloc[:cutoff], X.iloc[cutoff:], y.iloc[:cutoff], y.iloc[cutoff:]

        stratify = None
        if self.problem_type == "classification" and y.value_counts().min() >= 2:
            stratify = y
        try:
            return train_test_split(
                X, y, test_size=self.test_size,
                random_state=self.random_state, stratify=stratify,
            )
        except ValueError:
            # Stratification fails when a class is too small for the split size.
            return train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

    def _build_cv(self, n_train: int, y_train: pd.Series) -> Any:
        """Choose a CV splitter with a fold count the data can actually support."""
        n_folds = self._adaptive_folds(n_train)
        if self.time_series:
            return TimeSeriesSplit(n_splits=max(2, min(n_folds, n_train - 1)))
        if self.problem_type == "classification":
            smallest_class = int(y_train.value_counts().min())
            n_folds = max(2, min(n_folds, smallest_class))
            return StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )
        n_folds = max(2, min(n_folds, n_train))
        return KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

    def _adaptive_folds(self, n: int) -> int:
        if self.cv_folds is not None:
            return max(2, int(self.cv_folds))
        if self.mode == "fast":
            return 3
        if n < 500:
            return 3
        if n < 5_000:
            return 5
        if self.mode == "competition" and n < 50_000:
            return 7
        return 5

    # ------------------------------------------------------------------
    def _train_models(
        self, models, feature_select_step, X_train, y_train, X_test, y_test
    ) -> List[Dict[str, Any]]:
        """Fit each model, evaluate it, and record progress against the budget."""
        evaluator = Evaluator(self.problem_type)
        n_classes = int(self.profile_.get("n_classes") or 2)
        results: List[Dict[str, Any]] = []
        total = len(models)

        for index, (name, model) in enumerate(models.items(), start=1):
            if self._budget_left() < MIN_SECONDS_PER_MODEL:
                remaining = list(models)[index - 1:]
                for skipped in remaining:
                    self.skipped_models_[skipped] = "time budget exhausted"
                self.console.warn(
                    f"Time budget exhausted after {self._elapsed():.0f}s - "
                    f"skipping {len(remaining)} remaining model(s): "
                    f"{', '.join(remaining)}"
                )
                break

            model_started = time.time()
            try:
                base_steps: List[Any] = [("preprocessor", self._preprocessor)]
                if feature_select_step is not None:
                    base_steps.append(("feature_select", feature_select_step))

                if name in BOOSTING_MODELS:
                    pipe = self._fit_boosting(
                        base_steps, model, name, X_train, y_train, X_test, y_test
                    )
                    # Boosters early-stop against the holdout, so a CV re-fit here
                    # would both leak and cost more than the score is worth.
                    skip_cv = True
                else:
                    pipe = Pipeline(base_steps + [("model", model)])
                    pipe.fit(X_train, y_train)
                    skip_cv = False

                row = evaluator.evaluate(
                    pipe, X_train, y_train, X_test, y_test, self._cv,
                    name=name, n_classes=n_classes, skip_cv=skip_cv,
                )
                self.cv_scores_[name] = row.pop("_cv_scores", [])
                results.append(row)
                self.fitted_pipelines_[name] = pipe

                took = time.time() - model_started
                score = row[self._primary_score_col]
                self.console.info(
                    f"  {self.console.symbol('ok')} [{index}/{total}] {name:<20} "
                    f"{self._primary_score_col}={score:.4f}  "
                    f"({took:.1f}s, {self._elapsed():.0f}s/{self.time_budget:.0f}s budget)"
                )
            except Exception as exc:
                self.skipped_models_[name] = f"{type(exc).__name__}: {exc}"
                self.console.info(
                    f"  {self.console.symbol('fail')} [{index}/{total}] {name:<20} "
                    f"failed: {type(exc).__name__}: {exc}"
                )
        return results

    def _fit_boosting(
        self, base_steps, model, name, X_train, y_train, X_test, y_test
    ) -> _BoostingWrapper:
        """Fit a booster on transformed features with library-specific early stopping."""
        pre_pipe = Pipeline(base_steps)
        pre_pipe.fit(X_train, y_train)
        X_train_t = pre_pipe.transform(X_train)
        X_test_t = pre_pipe.transform(X_test)

        fitted = copy.deepcopy(model)
        lowered = name.lower()

        if "xgboost" in lowered:
            fitted.fit(X_train_t, y_train, **xgb_fit_kwargs(X_test_t, y_test, 50))
        elif "lightgbm" in lowered:
            kwargs: Dict[str, Any] = {"eval_set": [(X_test_t, y_test)]}
            try:
                from lightgbm import early_stopping, log_evaluation

                kwargs["callbacks"] = [
                    early_stopping(stopping_rounds=50, verbose=False),
                    log_evaluation(period=-1),
                ]
            except Exception:
                pass
            fitted.fit(X_train_t, y_train, **kwargs)
        elif "catboost" in lowered:
            fitted.fit(X_train_t, y_train, eval_set=(X_test_t, y_test), use_best_model=True)
        else:  # pragma: no cover - guarded by BOOSTING_MODELS
            fitted.fit(X_train_t, y_train)

        return _BoostingWrapper(
            pre_pipe=pre_pipe, model=fitted, problem_type=self.problem_type
        )

    # ------------------------------------------------------------------
    def _tune_top_models(self, X_train, y_train, X_test, y_test) -> None:
        """Optuna-tune the top N models, keeping a tuned fit only if it wins."""
        if self.mode == "fast":
            self._log("[5/6] Tuning skipped (fast mode).")
            return
        if not is_available("optuna"):
            self.console.warn(
                "[5/6] Tuning skipped - optuna is not installed "
                "(pip install optuna)."
            )
            return
        if self._budget_left() < 60:
            self.console.warn("[5/6] Tuning skipped - less than 60s of budget remains.")
            return

        top_models = self.results_df_["Model"].tolist()[: self.tune_top_n]
        per_model_budget = max(30, int(self._budget_left() * 0.5 / max(1, len(top_models))))
        self.console.step(5, 6, f"Tuning top {len(top_models)} model(s)")

        evaluator = Evaluator(self.problem_type)
        n_classes = int(self.profile_.get("n_classes") or 2)
        optimizer = OptunaOptimizer(
            self.problem_type, self._cv, self.scoring,
            time_budget_per_model=per_model_budget, n_trials=40,
            random_state=self.random_state,
        )

        for name in top_models:
            if self._budget_left() < MIN_SECONDS_PER_MODEL:
                self.console.warn("      Budget exhausted - stopping tuning early.")
                break
            pipeline = self.fitted_pipelines_.get(name)
            if pipeline is None:
                continue
            self._log(f"      Tuning {name} (up to {per_model_budget}s)...")
            tuned, best_value = optimizer.tune(pipeline, name, X_train, y_train)
            if best_value is None:
                self._log(f"      = {name}: no search space or no completed trials.")
                continue
            try:
                tuned.fit(X_train, y_train)
                tuned_name = f"{name} (tuned)"
                row = evaluator.evaluate(
                    tuned, X_train, y_train, X_test, y_test, self._cv,
                    name=tuned_name, n_classes=n_classes,
                )
                cv_scores = row.pop("_cv_scores", [])
                current = float(
                    self.results_df_.loc[
                        self.results_df_["Model"] == name, self._primary_score_col
                    ].values[0]
                )
                if row[self._primary_score_col] > current:
                    self.console.info(
                        f"      {self.console.symbol('up')} {name}: "
                        f"{current:.4f} -> {row[self._primary_score_col]:.4f}"
                    )
                    self.cv_scores_[tuned_name] = cv_scores
                    self._append_result(row)
                    self.fitted_pipelines_[tuned_name] = tuned
                else:
                    self._log(f"      = {name}: tuning did not improve the score.")
            except Exception as exc:
                self._log(f"      {self.console.symbol('fail')} Tuning {name} failed: {exc}")

    def _build_stack(self, X_train, y_train, X_test, y_test) -> None:
        """Stack the top models when the mode and budget allow it."""
        if self.mode not in ("balanced", "competition"):
            self._log("[6/6] Stacking skipped (fast mode).")
            return
        if len(self.fitted_pipelines_) < 3:
            self._log("[6/6] Stacking skipped - fewer than 3 trained models.")
            return
        if self._budget_left() < 60:
            self.console.warn("[6/6] Stacking skipped - less than 60s of budget remains.")
            return

        self.console.step(6, 6, "Building stacked ensemble")
        top_n = min(self.stack_top_n, len(self.fitted_pipelines_))
        top_names = self.results_df_["Model"].tolist()[:top_n]
        selected = {
            name: self.fitted_pipelines_[name]
            for name in top_names
            if name in self.fitted_pipelines_
        }
        try:
            engine = StackingEngine(self.problem_type, self._cv, self.random_state)
            stacked = engine.build_stack(
                selected, X_train, y_train, X_test, y_test, top_n=top_n
            )
            if stacked is None:
                self._log("      Stacking produced fewer than 2 usable base models.")
                return
            self.stacked_estimator_ = stacked
            evaluator = Evaluator(self.problem_type)
            row = evaluator.evaluate(
                stacked, X_train, y_train, X_test, y_test, self._cv,
                name="StackedEnsemble",
                n_classes=int(self.profile_.get("n_classes") or 2),
                skip_cv=True,
            )
            row.pop("_cv_scores", None)
            self._append_result(row)
            self.fitted_pipelines_["StackedEnsemble"] = stacked
            self.console.info(
                f"      {self.console.symbol('ok')} StackedEnsemble "
                f"{self._primary_score_col}={row[self._primary_score_col]:.4f} "
                f"({len(stacked.base_estimators)} base models)"
            )
        except Exception as exc:
            self.console.warn(f"      Stacking failed: {type(exc).__name__}: {exc}")

    def _append_result(self, row: Dict[str, Any]) -> None:
        """Add a leaderboard row and re-sort by the primary score."""
        self.results_df_ = (
            pd.concat([self.results_df_, pd.DataFrame([row])], ignore_index=True)
            .sort_values(self._primary_score_col, ascending=False)
            .reset_index(drop=True)
        )

    def _build_metadata(self) -> Dict[str, Any]:
        engineer = self.feature_engineer_
        return {
            "target": self.target,
            "problem_type": self.problem_type,
            "feature_columns": self.feature_columns_,
            "input_columns": list(engineer.input_columns_) if engineer else [],
            "input_dtypes": dict(engineer.input_dtypes_) if engineer else {},
            "dropped_columns": list(engineer.drop_cols_) if engineer else [],
            "profile": self.profile_,
            "mode": self.mode,
            "random_state": self.random_state,
            "best_model": self.best_model_name_,
            "primary_score_col": self._primary_score_col,
            "label_classes": (
                [str(c) for c in self.label_encoder_.classes_]
                if self.label_encoder_ is not None
                else None
            ),
            "fit_seconds": self._fit_seconds,
            "time_budget": self.time_budget,
            "test_size": self.test_size,
            "cv_folds": getattr(self._cv, "n_splits", None),
            "cv_strategy": type(self._cv).__name__ if self._cv is not None else None,
            "scoring": self.scoring,
            "pca_applied": self._decide_pca,
            "feature_selection_applied": self._fs_step_used,
            "skipped_models": dict(self.skipped_models_),
            "feature_engineering": engineer.describe() if engineer else {},
            "dive_version": _package_version(),
        }

    # ------------------------------------------------------------------
    def leaderboard(self, n: Optional[int] = None) -> pd.DataFrame:
        """Return the leaderboard, without private bookkeeping columns."""
        if self.results_df_ is None:
            raise TrainingError("No leaderboard is available - the model is not fitted.")
        frame = self.results_df_.drop(
            columns=[c for c in self.results_df_.columns if c.startswith("_")],
            errors="ignore",
        )
        return frame.head(n) if n else frame.copy()

    def _prepare_features(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the *fitted* engineer and align columns to training order."""
        if self.feature_engineer_ is None or self.feature_columns_ is None:
            raise SchemaError(
                "This model has no fitted feature engineer.",
                "Retrain with the current version of dive.",
            )
        transformed = self.feature_engineer_.transform(new_df.copy())
        return transformed.reindex(columns=self.feature_columns_, fill_value=0)

    def predict(self, new_df: pd.DataFrame) -> np.ndarray:
        """Predict on new raw data, decoding labels back to their original values."""
        prepared = self._prepare_features(new_df)
        predictions = self.best_estimator_.predict(prepared)
        if self.problem_type == "classification" and self.label_encoder_ is not None:
            predictions = self.label_encoder_.inverse_transform(
                np.asarray(predictions).astype(int)
            )
        return np.asarray(predictions)

    def predict_proba(self, new_df: pd.DataFrame) -> np.ndarray:
        """Class probabilities, in the column order of ``label_encoder_.classes_``."""
        if self.problem_type != "classification":
            raise SchemaError(
                "predict_proba is only available for classification models.",
                f"This model solves a {self.problem_type} problem.",
            )
        prepared = self._prepare_features(new_df)
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise SchemaError(
                f"The best model ({self.best_model_name_}) cannot produce probabilities.",
                "Re-run training; a different model may support predict_proba.",
            )
        return np.asarray(self.best_estimator_.predict_proba(prepared))

    @property
    def class_names(self) -> Optional[List[str]]:
        if self.label_encoder_ is None:
            return None
        return [str(c) for c in self.label_encoder_.classes_]

    # ------------------------------------------------------------------
    def save(self, path: Any = "dive_model.pkl") -> Path:
        """Persist the fitted engineer, best estimator, leaderboard, and metadata."""
        if self.best_estimator_ is None:
            raise TrainingError("Nothing to save - the model has not been fitted.")
        payload = {
            "format_version": 1,
            "feature_engineer": self.feature_engineer_,
            "feature_columns": self.feature_columns_,
            "best_estimator": self.best_estimator_,
            "label_encoder": self.label_encoder_,
            "results_df": self.results_df_,
            "metadata": self._metadata,
            "problem_type": self.problem_type,
            "target": self.target,
            "profile": self.profile_,
            "cv_scores": self.cv_scores_,
        }
        saved = save_pickle(payload, path)
        self._log(f"      Model saved to {saved}")
        return saved

    @classmethod
    def load(cls, path: Any = "dive_model.pkl") -> "Dive":
        """Rebuild a predict-ready instance from a saved artifact."""
        payload = load_pickle(path)
        if not isinstance(payload, dict) or "best_estimator" not in payload:
            raise SchemaError(
                f"'{path}' is not an dive model file.",
                "It unpickled successfully but has none of the expected contents.",
            )

        instance = cls.__new__(cls)
        metadata = payload.get("metadata", {}) or {}

        instance.feature_engineer_ = payload.get("feature_engineer")
        instance.feature_columns_ = payload.get("feature_columns")
        instance.best_estimator_ = payload.get("best_estimator")
        instance.label_encoder_ = payload.get("label_encoder")
        instance.results_df_ = payload.get("results_df")
        instance.profile_ = payload.get("profile") or metadata.get("profile")
        instance.cv_scores_ = payload.get("cv_scores", {}) or {}
        instance._metadata = metadata
        instance.problem_type = payload.get("problem_type")
        instance.target = payload.get("target")
        instance.best_model_name_ = metadata.get("best_model")
        instance.mode = metadata.get("mode", "balanced")
        instance.random_state = metadata.get("random_state", 42)
        instance.test_size = metadata.get("test_size", 0.2)
        instance.time_budget = metadata.get("time_budget", 1800)
        instance.scoring = metadata.get("scoring")
        instance.verbose = True
        instance.console = get_console()
        instance._primary_score_col = metadata.get("primary_score_col") or (
            "Test Accuracy"
            if payload.get("problem_type") == "classification"
            else "Test R2"
        )
        instance._decide_pca = metadata.get("pca_applied", False)
        instance._fs_step_used = metadata.get("feature_selection_applied", False)
        instance._fit_seconds = metadata.get("fit_seconds")
        instance.stacked_estimator_ = None
        instance.fitted_pipelines_ = {}
        instance.skipped_models_ = metadata.get("skipped_models", {}) or {}
        instance.time_series = False
        instance.use_pca = "auto"
        instance.tune_top_n = 3
        instance.stack_top_n = 5
        instance.outlier_clip = True
        instance.cv_folds = metadata.get("cv_folds")
        instance._X_train = instance._X_test = None
        instance._y_train = instance._y_test = None
        instance._preprocessor = None
        instance._start_time = None
        instance._cv = None
        return instance

    # ------------------------------------------------------------------
    def check_schema(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare incoming columns against the training schema (no exceptions).

        Returns a dict with ``missing``/``extra``/``ok`` keys. The hard-failing
        variant used by ``dive predict`` lives in :mod:`dive.validation`.
        """
        expected = list(self._metadata.get("input_columns") or [])
        dropped = set(self._metadata.get("dropped_columns") or [])
        required = [c for c in expected if c not in dropped]
        present = {str(c) for c in new_df.columns}
        missing = [c for c in required if c not in present]
        extra = [c for c in present if c not in set(expected)]
        return {"missing": missing, "extra": extra, "ok": not missing}

    def describe_pipeline(self) -> Dict[str, Any]:
        """Return a structured description of the fitted pipeline for reports."""
        engineer = self.feature_engineer_
        best = self.best_estimator_
        inner = _inner_model(best)
        description: Dict[str, Any] = {
            "target": self.target,
            "problem_type": self.problem_type,
            "mode": self._metadata.get("mode", self.mode),
            "best_model": self.best_model_name_,
            "best_model_type": type(inner).__name__ if inner is not None else None,
            "n_features": len(self.feature_columns_ or []),
            "feature_columns": list(self.feature_columns_ or []),
            "profile": self.profile_ or {},
            "feature_engineering": (
                engineer.describe() if engineer is not None
                else self._metadata.get("feature_engineering", {})
            ),
            "pca_applied": self._decide_pca,
            "feature_selection_applied": self._fs_step_used,
            "cv_strategy": self._metadata.get("cv_strategy"),
            "cv_folds": self._metadata.get("cv_folds"),
            "scoring": self._metadata.get("scoring", self.scoring),
            "label_classes": self._metadata.get("label_classes"),
            "fit_seconds": self._metadata.get("fit_seconds"),
            "skipped_models": self._metadata.get("skipped_models", {}),
            "hyperparameters": _safe_params(inner),
            "is_stack": isinstance(best, _StackWrapper),
            "stack_base_models": (
                [name for name, _ in best.base_estimators]
                if isinstance(best, _StackWrapper)
                else []
            ),
        }
        return description

    def feature_importances(self, top_n: int = 25) -> Optional[pd.DataFrame]:
        """Return a ranked importance table, or ``None`` when unavailable.

        Importances come from the model's own ``feature_importances_``/``coef_``.
        These are indexed against the *post-preprocessing* matrix, which after
        one-hot encoding may be wider than the engineered feature list, so names
        are only attached when the widths line up.
        """
        inner = _inner_model(self.best_estimator_)
        if inner is None:
            return None
        if hasattr(inner, "feature_importances_"):
            values = np.asarray(inner.feature_importances_, dtype=float).ravel()
        elif hasattr(inner, "coef_"):
            values = np.abs(np.asarray(inner.coef_, dtype=float)).ravel()
        else:
            return None
        if values.size == 0:
            return None

        names = list(self.feature_columns_ or [])
        if len(names) != values.size:
            names = [f"feature_{i}" for i in range(values.size)]

        frame = pd.DataFrame({"feature": names, "importance": values})
        frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
        return frame.head(top_n)


# ----------------------------------------------------------------------
def _inner_model(estimator: Any) -> Optional[Any]:
    """Unwrap a pipeline/booster wrapper down to the actual estimator."""
    if estimator is None:
        return None
    if isinstance(estimator, _BoostingWrapper):
        return estimator.model
    if isinstance(estimator, _StackWrapper):
        return estimator.meta_learner
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps.get("model", estimator)
    return estimator


def _safe_params(model: Any) -> Dict[str, str]:
    """Stringified hyperparameters, skipping callables and callback objects."""
    if model is None or not hasattr(model, "get_params"):
        return {}
    try:
        params = model.get_params()
    except Exception:
        return {}
    return {
        str(key): str(value)
        for key, value in sorted(params.items())
        if not callable(value) and key != "callbacks" and value != "deprecated"
    }


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("dive")
    except Exception:
        return "0.1.0"


def quick_dive(
    df: pd.DataFrame,
    target: Optional[str] = None,
    mode: str = "balanced",
    time_budget: float = 600,
    **kwargs: Any,
) -> Dive:
    """One-call convenience wrapper mirroring the notebook's entry point."""
    dive = Dive(
        target=target, mode=mode, time_budget=time_budget, **kwargs
    )
    dive.fit(df)
    return dive
