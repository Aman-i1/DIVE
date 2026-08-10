"""Stage 6 - manual stacking of the top models.

Ported from ``StackingEngine`` / ``_StackWrapper`` in
``Automatic Machine Learning.ipynb``. The approach is unchanged: generate
out-of-fold predictions for each base model, train a meta-learner on the OOF
matrix, and average base test predictions across folds at inference time.

One correctness fix carried over from the port: early stopping is stripped via
:func:`dive.model_zoo.strip_early_stopping`, which also clears the XGBoost 2.x
constructor callback. The notebook only cleared ``early_stopping_rounds``, so on
XGBoost >= 2.0 every booster raised inside the fold loop (no eval set) and was
silently dropped from the stack by the surrounding ``except Exception: continue``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge

from dive.model_zoo import _BoostingWrapper, strip_early_stopping


class _StackWrapper(BaseEstimator):
    """A fitted stack: frozen base models feeding a trained meta-learner."""

    def __init__(
        self,
        base_estimators: List[Tuple[str, Any]],
        meta_learner: Any,
        filled_cols: List[int],
        out_width: int,
        use_proba: bool,
        problem_type: str,
        n_classes: int,
    ) -> None:
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        self.filled_cols = filled_cols
        self.out_width = out_width
        self.use_proba = use_proba
        self.problem_type = problem_type
        self.n_classes = n_classes

    def _build_meta_input(self, X: Any) -> np.ndarray:
        """Stack each base model's output into the meta-learner's feature matrix."""
        parts = []
        for _, estimator in self.base_estimators:
            if isinstance(estimator, _BoostingWrapper):
                X_t = estimator.pre_pipe.transform(X)
                raw = estimator.model
                if self.use_proba and hasattr(raw, "predict_proba"):
                    parts.append(raw.predict_proba(X_t))
                else:
                    parts.append(np.asarray(raw.predict(X_t)).reshape(-1, 1))
            else:
                if self.use_proba and hasattr(estimator, "predict_proba"):
                    parts.append(estimator.predict_proba(X))
                else:
                    parts.append(np.asarray(estimator.predict(X)).reshape(-1, 1))
        return np.hstack(parts)

    def predict(self, X: Any) -> np.ndarray:
        return self.meta_learner.predict(self._build_meta_input(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.problem_type != "classification":
            raise ValueError("predict_proba is available for classification only.")
        if hasattr(self.meta_learner, "predict_proba"):
            return self.meta_learner.predict_proba(self._build_meta_input(X))
        raise AttributeError("The meta-learner has no predict_proba.")

    def fit(self, X: Any, y: Any) -> "_StackWrapper":
        """No-op: base models and meta-learner are already fitted.

        Present so the object satisfies the estimator interface used by the
        evaluator; cross-validating a stack is explicitly skipped upstream.
        """
        return self


class StackingEngine:
    """Build an out-of-fold stacked ensemble from already-fitted pipelines."""

    def __init__(self, problem_type: str, cv: Any, random_state: int = 42) -> None:
        self.problem_type = problem_type
        self.cv = cv
        self.random_state = random_state

    def build_stack(
        self,
        fitted_pipelines: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        top_n: int = 5,
    ) -> Optional[_StackWrapper]:
        """Return a fitted stack, or ``None`` when fewer than 2 bases survive."""
        items = list(fitted_pipelines.items())[:top_n]
        if len(items) < 2:
            return None

        n_train = len(X_train)
        n_classes = int(len(np.unique(y_train)))
        is_classification = self.problem_type == "classification"
        use_proba = is_classification
        out_width = n_classes if use_proba else 1

        oof_matrix = np.zeros((n_train, len(items) * out_width))
        surviving: List[Tuple[str, Any, int, int]] = []

        X_arr = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

        for index, (name, estimator) in enumerate(items):
            col_start, col_end = index * out_width, (index + 1) * out_width
            oof_preds = np.zeros((n_train, out_width))
            folds_done = 0
            try:
                for train_idx, val_idx in self.cv.split(X_arr, y_arr):
                    X_fold_train = self._take(X_train, train_idx)
                    y_fold_train = self._take(y_train, train_idx)
                    X_fold_val = self._take(X_train, val_idx)

                    cloned = copy.deepcopy(estimator)

                    if isinstance(cloned, _BoostingWrapper):
                        raw = copy.deepcopy(cloned.model)
                        strip_early_stopping(raw)
                        X_ft = cloned.pre_pipe.transform(X_fold_train)
                        X_fv = cloned.pre_pipe.transform(X_fold_val)
                        raw.fit(X_ft, y_fold_train)
                        val_pred = self._emit(raw, X_fv, use_proba, out_width)
                    else:
                        cloned.fit(X_fold_train, y_fold_train)
                        val_pred = self._emit(cloned, X_fold_val, use_proba, out_width)

                    oof_preds[val_idx] = val_pred
                    folds_done += 1

                if folds_done == 0:
                    continue
                oof_matrix[:, col_start:col_end] = oof_preds
                surviving.append((name, estimator, col_start, col_end))
            except Exception:
                # A base model that cannot produce OOF predictions is dropped;
                # the stack is still valid as long as two others succeed.
                continue

        if len(surviving) < 2:
            return None

        filled_cols: List[int] = []
        for _, _, col_start, col_end in surviving:
            filled_cols.extend(range(col_start, col_end))

        oof_matrix = oof_matrix[:, filled_cols]

        meta = (
            LogisticRegression(
                max_iter=2000, C=1.0, random_state=self.random_state
            )
            if is_classification
            else Ridge(random_state=self.random_state)
        )
        meta.fit(oof_matrix, y_train)

        return _StackWrapper(
            base_estimators=[(name, est) for name, est, _, _ in surviving],
            meta_learner=meta,
            filled_cols=filled_cols,
            out_width=out_width,
            use_proba=use_proba,
            problem_type=self.problem_type,
            n_classes=n_classes,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _take(data: Any, indices: np.ndarray) -> Any:
        """Positional row selection that works for DataFrames, Series and arrays."""
        return data.iloc[indices] if hasattr(data, "iloc") else data[indices]

    @staticmethod
    def _emit(model: Any, X: Any, use_proba: bool, out_width: int) -> np.ndarray:
        """Produce a fixed-width prediction block for the OOF matrix.

        A CV fold can be missing a class entirely, in which case ``predict_proba``
        returns fewer columns than the full-data width. The columns are remapped
        by ``model.classes_`` so probabilities always land in the right slot
        rather than being shifted - a silent corruption of the meta-features.
        """
        if use_proba and hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            if proba.shape[1] == out_width:
                return proba
            block = np.zeros((proba.shape[0], out_width))
            classes = getattr(model, "classes_", None)
            if classes is None:
                block[:, : proba.shape[1]] = proba[:, :out_width]
                return block
            for position, class_label in enumerate(classes):
                try:
                    column = int(class_label)
                except (TypeError, ValueError):
                    column = position
                if 0 <= column < out_width:
                    block[:, column] = proba[:, position]
            return block
        return np.asarray(model.predict(X)).reshape(-1, 1)
