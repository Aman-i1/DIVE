"""Standalone, self-describing predictors - one per trained model.

``dive train`` writes ``model.pkl`` (the whole :class:`~dive.core.Dive` run) and,
alongside it, one :class:`DivePredictor` per model in the zoo. A predictor is the
smallest thing that can turn a row of *raw* data into a prediction:

    import pickle, pandas as pd
    predictor = pickle.load(open("iris_data__XGBoost.pkl", "rb"))
    predictor.predict(pd.read_csv("new_rows.csv"))

The distinction that motivates this module: a fitted estimator alone is not
usable. It was trained on the output of a :class:`~dive.feature_engineering.FeatureEngineer`
- columns dropped, dates expanded into calendar parts, rare categories bucketed,
categoricals frequency- and target-encoded, labels integer-encoded. Handing it
raw CSV rows produces either a crash or, worse, silently wrong numbers. A
predictor bundles the fitted engineer, the exact training column order, and the
label encoder with the estimator, so raw input is the *only* thing it accepts.

Each predictor also carries :attr:`input_schema`, a description of the raw frame
it was trained on - column names, dtypes, which columns are required, the
categories seen per categorical column, and an example row - so a caller can
discover the expected input structure without the original CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Rows accepted by DivePredictor.predict: a frame, one dict, or many dicts.
RawInput = Union[pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]

# Categorical columns with more distinct values than this are summarised by
# count rather than listed, to keep the pickle and the schema JSON small.
_MAX_LISTED_CATEGORIES = 50


class SchemaMismatch(ValueError):
    """Raised when incoming raw data does not match the training schema."""


def build_input_schema(
    raw_df: pd.DataFrame,
    target: str,
    dropped_columns: Sequence[str],
) -> Dict[str, Any]:
    """Describe the raw training frame so callers can reconstruct valid input.

    ``raw_df`` is the frame as the user supplied it, *before* any feature
    engineering, minus nothing - the target is filtered out here rather than by
    the caller so the recorded example row cannot accidentally leak a label.
    """
    features = [str(c) for c in raw_df.columns if str(c) != str(target)]
    dropped = {str(c) for c in dropped_columns}

    columns: List[Dict[str, Any]] = []
    for name in features:
        series = raw_df[name]
        entry: Dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "kind": _kind_of(series),
            # A column dropped during training (constant or ID-like) may be
            # omitted at predict time: it cannot change the prediction.
            "required": name not in dropped,
            "used_by_model": name not in dropped,
            "nullable": bool(series.isna().any()),
            "example": _json_safe(series.dropna().iloc[0]) if series.notna().any() else None,
        }
        if entry["kind"] == "categorical":
            uniques = series.dropna().unique()
            entry["n_categories"] = int(len(uniques))
            if len(uniques) <= _MAX_LISTED_CATEGORIES:
                entry["categories"] = sorted(_json_safe(v) for v in uniques)
        elif entry["kind"] == "numeric":
            entry["min"] = _json_safe(series.min())
            entry["max"] = _json_safe(series.max())
        columns.append(entry)

    example_row = {}
    if len(raw_df):
        first = raw_df.iloc[0]
        example_row = {name: _json_safe(first[name]) for name in features}

    return {
        "target": str(target),
        "n_features": len(features),
        "columns": columns,
        "required_columns": [c["name"] for c in columns if c["required"]],
        "optional_columns": [c["name"] for c in columns if not c["required"]],
        "column_order": features,
        "example_row": example_row,
    }


def _kind_of(series: pd.Series) -> str:
    """Coarse column family used for schema description and dtype warnings."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars into plain Python for JSON serialisation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


class DivePredictor:
    """One trained model plus everything needed to feed it raw data.

    Pickle-round-trips as a unit. Unpickling requires ``dive`` to be installed,
    because the fitted feature engineer and any booster wrappers are dive types.
    """

    def __init__(
        self,
        model_name: str,
        estimator: Any,
        feature_engineer: Any,
        feature_columns: Sequence[str],
        label_encoder: Any,
        target: str,
        problem_type: str,
        input_schema: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        dataset_name: str = "",
        dive_version: str = "",
        trained_at: str = "",
        label_lookup: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.estimator = estimator
        self.feature_engineer = feature_engineer
        self.feature_columns = list(feature_columns)
        self.label_encoder = label_encoder
        # Maps the stringified label back to the value (and dtype) the user
        # supplied, so an integer target does not come back as "0"/"1".
        self.label_lookup = dict(label_lookup or {})
        self.target = target
        self.problem_type = problem_type
        self.input_schema = input_schema
        self.metrics = dict(metrics or {})
        self.dataset_name = dataset_name
        self.dive_version = dive_version
        self.trained_at = trained_at

    # -- input handling -------------------------------------------------
    @property
    def required_columns(self) -> List[str]:
        """Raw columns that must be present in incoming data."""
        return list(self.input_schema.get("required_columns", []))

    @property
    def class_names(self) -> Optional[List[str]]:
        if self.label_encoder is None:
            return None
        return [str(c) for c in self.label_encoder.classes_]

    @property
    def has_proba(self) -> bool:
        """True when :meth:`predict_proba` is available on this predictor."""
        return self.problem_type == "classification" and hasattr(
            self.estimator, "predict_proba"
        )

    def _coerce(self, data: RawInput) -> pd.DataFrame:
        """Accept a DataFrame, a single row dict, a list of row dicts, or numpy ndarray."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, np.ndarray):
            if self.feature_columns and data.ndim > 1 and data.shape[1] == len(self.feature_columns):
                return pd.DataFrame(data, columns=self.feature_columns)
            return pd.DataFrame(data)
        if isinstance(data, Mapping):
            return pd.DataFrame([dict(data)])
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            rows = list(data)
            if rows and all(isinstance(row, Mapping) for row in rows):
                return pd.DataFrame([dict(row) for row in rows])
        raise SchemaMismatch(
            "predict() expects a pandas DataFrame, a dict of one row, a list "
            f"of row dicts, or a numpy ndarray - got {type(data).__name__}. "
            "Call .describe_input() to see the expected structure."
        )

    def _prepare(self, data: RawInput) -> pd.DataFrame:
        """Validate raw input, then apply the fitted training-time transforms."""
        frame = self._coerce(data)
        frame.columns = [str(c).strip() for c in frame.columns]
        for c in frame.columns:
            if frame[c].dtype == object:
                num_s = pd.to_numeric(frame[c], errors="coerce")
                if num_s.notna().sum() / max(len(frame), 1) > 0.80:
                    frame[c] = num_s

        # The target may be present (scoring a labelled holdout); it is never an
        # input to the model, so drop it rather than letting it look like noise.
        if self.target in frame.columns:
            frame = frame.drop(columns=[self.target])

        missing = [c for c in self.required_columns if c not in frame.columns]
        if missing:
            raise SchemaMismatch(
                f"{len(missing)} required column(s) missing: {', '.join(missing)}.\n"
                f"This predictor expects: {', '.join(self.required_columns)}.\n"
                "Call .describe_input() for dtypes and an example row."
            )

        if self.feature_engineer is not None:
            engineered = self.feature_engineer.transform(frame)
        else:
            engineered = frame

        if self.feature_columns:
            return engineered.reindex(columns=self.feature_columns, fill_value=0)
        return engineered

    # -- prediction -----------------------------------------------------
    def predict(self, data: RawInput) -> np.ndarray:
        """Predict from raw, un-encoded input.

        Classification labels come back in their original form (the strings or
        values from the training file), not as encoded integers.
        """
        predictions = self.estimator.predict(self._prepare(data))
        if self.problem_type in ("classification", "binary_classification", "multiclass") and self.label_encoder is not None:
            from dive.core import decode_labels

            predictions = decode_labels(
                predictions, self.label_encoder, getattr(self, "label_lookup", None)
            )
        return np.asarray(predictions)

    def predict_proba(self, data: RawInput) -> pd.DataFrame:
        """Class probabilities as a DataFrame with original class names."""
        if self.problem_type not in ("classification", "binary_classification", "multiclass"):
            raise SchemaMismatch(
                f"predict_proba is classification-only; this predictor solves a "
                f"{self.problem_type} problem."
            )
        if not hasattr(self.estimator, "predict_proba"):
            raise SchemaMismatch(
                f"{self.model_name} cannot produce probabilities."
            )
        proba = np.asarray(self.estimator.predict_proba(self._prepare(data)))
        names = self.class_names or [f"class_{i}" for i in range(proba.shape[1])]
        if len(names) != proba.shape[1]:
            names = [f"class_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, columns=names)

    def __call__(self, data: RawInput) -> np.ndarray:
        return self.predict(data)

    # -- introspection --------------------------------------------------
    def describe_input(self) -> str:
        """Return a human-readable description of the expected input."""
        lines = [
            f"{self.model_name} - trained on '{self.dataset_name}' to predict '{self.target}'",
            f"Problem type: {self.problem_type}",
            "",
            f"Expects a DataFrame (or dict) with {len(self.required_columns)} required column(s):",
        ]
        for column in self.input_schema.get("columns", []):
            flag = "required" if column["required"] else "optional (unused)"
            detail = f"  {column['name']:<28} {column['dtype']:<12} {flag}"
            if column.get("categories"):
                shown = ", ".join(map(str, column["categories"][:8]))
                more = " ..." if len(column["categories"]) > 8 else ""
                detail += f"\n      categories: {shown}{more}"
            elif column.get("n_categories"):
                detail += f"\n      {column['n_categories']} distinct categories"
            elif column.get("min") is not None:
                detail += f"\n      range: {column['min']} .. {column['max']}"
            lines.append(detail)

        example = self.input_schema.get("example_row") or {}
        if example:
            lines.extend(["", "Example call:", "", "  predictor.predict({"])
            for key, value in example.items():
                lines.append(f"      {key!r}: {value!r},")
            lines.append("  })")
        if self.class_names:
            lines.extend(["", f"Returns one of: {', '.join(self.class_names)}"])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary, written beside the pickles as JSON."""
        return {
            "model_name": self.model_name,
            "dataset": self.dataset_name,
            "target": self.target,
            "problem_type": self.problem_type,
            "classes": self.class_names,
            "metrics": self.metrics,
            "dive_version": self.dive_version,
            "trained_at": self.trained_at,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        return (
            f"DivePredictor(model={self.model_name!r}, target={self.target!r}, "
            f"problem_type={self.problem_type!r}, "
            f"n_required_columns={len(self.required_columns)})"
        )


def load_predictor(path: Any) -> DivePredictor:
    """Load a predictor pickle written by ``dive train`` or wrap a full-run ``model.pkl``."""
    from dive.utils.io import load_pickle

    resolved_path = Path(str(path))
    obj = load_pickle(resolved_path)
    if isinstance(obj, DivePredictor):
        return obj

    # If the user supplied the full-run model.pkl (dict format)
    if isinstance(obj, dict) and "best_estimator" in obj:
        # 1. First check if adjacent models/ directory holds the exported DivePredictor
        models_dir = resolved_path.parent / "models"
        metadata = obj.get("metadata") or {}
        best_name = metadata.get("best_model")
        if models_dir.exists() and best_name:
            for pkl_file in models_dir.glob("*.pkl"):
                if f"__{best_name}.pkl" in pkl_file.name or best_name in pkl_file.name:
                    try:
                        cand = load_pickle(pkl_file)
                        if isinstance(cand, DivePredictor):
                            return cand
                    except Exception:
                        pass

        # 2. Reconstruct DivePredictor from the saved full-run payload
        from dive.core import Dive
        dive_inst = Dive.load(resolved_path)
        feat_cols = list(dive_inst.feature_columns_ or [])
        columns_meta = []
        for col in feat_cols:
            columns_meta.append({
                "name": str(col),
                "dtype": "float64",
                "kind": "numeric",
                "required": True,
                "used_by_model": True,
                "nullable": False,
                "example": None,
            })
        schema = {
            "target": str(dive_inst.target),
            "n_features": len(feat_cols),
            "columns": columns_meta,
            "required_columns": feat_cols,
            "optional_columns": [],
            "column_order": feat_cols,
            "example_row": {},
        }
        return DivePredictor(
            model_name=dive_inst.best_model_name_ or "Champion",
            estimator=dive_inst.best_estimator_,
            feature_engineer=dive_inst.feature_engineer_,
            feature_columns=feat_cols,
            label_encoder=dive_inst.label_encoder_,
            label_lookup=dict(getattr(dive_inst, "label_lookup_", {}) or {}),
            target=str(dive_inst.target),
            problem_type=str(dive_inst.problem_type),
            input_schema=schema,
            metrics={},
            dataset_name=resolved_path.stem,
            dive_version="0.1.0",
            trained_at="",
        )

    # If obj is a Dive instance
    from dive.core import Dive
    if isinstance(obj, Dive):
        feat_cols = list(obj.feature_columns_ or [])
        columns_meta = [
            {
                "name": str(col),
                "dtype": "float64",
                "kind": "numeric",
                "required": True,
                "used_by_model": True,
                "nullable": False,
                "example": None,
            }
            for col in feat_cols
        ]
        schema = {
            "target": str(obj.target),
            "n_features": len(feat_cols),
            "columns": columns_meta,
            "required_columns": feat_cols,
            "optional_columns": [],
            "column_order": feat_cols,
            "example_row": {},
        }
        return DivePredictor(
            model_name=obj.best_model_name_ or "Champion",
            estimator=obj.best_estimator_,
            feature_engineer=obj.feature_engineer_,
            feature_columns=feat_cols,
            label_encoder=obj.label_encoder_,
            label_lookup=dict(getattr(obj, "label_lookup_", {}) or {}),
            target=str(obj.target),
            problem_type=str(obj.problem_type),
            input_schema=schema,
            metrics={},
            dataset_name=resolved_path.stem,
            dive_version="0.1.0",
            trained_at="",
        )

    # If obj is a scikit-learn Pipeline or any estimator (e.g. from reproducibility bundle)
    if hasattr(obj, "predict"):
        import json
        meta_file = resolved_path.parent / "metadata.json"
        meta_dict = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta_dict = json.load(mf)
            except Exception:
                pass

        target_name = meta_dict.get("target") or "target"
        problem_type = meta_dict.get("problem_type")
        if not problem_type:
            has_proba = hasattr(obj, "predict_proba") or (
                hasattr(obj, "steps") and hasattr(obj.steps[-1][1], "predict_proba")
            )
            problem_type = "classification" if has_proba else "regression"

        if hasattr(obj, "steps"):
            model_name = obj.steps[-1][0] or type(obj.steps[-1][1]).__name__
        else:
            model_name = type(obj).__name__

        feat_cols = []
        if hasattr(obj, "feature_names_in_"):
            feat_cols = list(obj.feature_names_in_)
        elif hasattr(obj, "steps") and hasattr(obj.steps[0][1], "feature_names_in_"):
            feat_cols = list(obj.steps[0][1].feature_names_in_)

        columns_meta = [
            {
                "name": str(col),
                "dtype": "float64",
                "kind": "numeric",
                "required": True,
                "used_by_model": True,
                "nullable": False,
                "example": None,
            }
            for col in feat_cols
        ]
        schema = {
            "target": str(target_name),
            "n_features": len(feat_cols),
            "columns": columns_meta,
            "required_columns": feat_cols,
            "optional_columns": [],
            "column_order": feat_cols,
            "example_row": {},
        }
        return DivePredictor(
            model_name=model_name,
            estimator=obj,
            feature_engineer=None,
            feature_columns=feat_cols,
            label_encoder=None,
            label_lookup={},
            target=str(target_name),
            problem_type=str(problem_type),
            input_schema=schema,
            metrics={},
            dataset_name=resolved_path.stem,
            dive_version="0.1.0",
            trained_at="",
        )

    raise TypeError(
        f"{resolved_path.name} does not contain a DivePredictor, Pipeline, or valid Dive model artifact "
        f"(found {type(obj).__name__})."
    )
