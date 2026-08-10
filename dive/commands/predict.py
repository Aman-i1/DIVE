"""Implementation of ``dive predict``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dive.core import Dive
from dive.exceptions import DataError, SchemaError
from dive.utils.io import load_dataframe, save_dataframe
from dive.utils.logging import Console

# Predictions with values beyond this range are almost certainly a schema mix-up.
_MIN_PLAUSIBLE_ABS = 1e14
_MAX_PLAUSIBLE_ABS = 1e18


def run_predict(
    console: Console,
    model_path: str,
    data_path: str,
    output_path: str,
    with_proba: bool = False,
    include_input: bool = False,
) -> Dict[str, Any]:
    """Load a model, verify the incoming schema, score rows, write a CSV."""
    dive = Dive.load(model_path)

    frame = load_dataframe(data_path)
    console.rule("dive predict")
    console.kv("Model", Path(str(model_path)).name)
    console.kv("Incoming rows", frame.shape[0])
    console.kv("Incoming columns", frame.shape[1])

    if dive.feature_engineer_ is None or not dive.feature_columns_:
        raise SchemaError(
            "The saved model carries no fitted feature engineer.",
            "Retrain with the current version of dive.",
        )

    _check_schema(dive, frame, console)

    if include_input and dive.target in frame.columns:
        frame = frame.drop(columns=[dive.target])

    predictions = dive.predict(frame)

    if with_proba:
        if dive.problem_type != "classification":
            raise SchemaError(
                "--proba requires a classification model.",
                f"This model solves a {dive.problem_type} problem.",
            )
        probabilities = dive.predict_proba(frame)
        class_names = dive.class_names or [
            f"class_{i}" for i in range(probabilities.shape[1])
        ]
        if len(class_names) != probabilities.shape[1]:
            raise SchemaError(
                "Class count mismatch while decoding probabilities.",
                "This can happen when the model and the data disagree on classes.",
            )

    _sanity_check_predictions(predictions, frame)

    if include_input:
        original = load_dataframe(data_path)
        if dive.target in original.columns:
            original = original.drop(columns=[dive.target])
        output = original.copy()
    else:
        output = pd.DataFrame(index=frame.index)

    output.insert(0, "prediction", predictions)

    if with_proba:
        for name, column in zip(class_names, probabilities.T):
            output[f"prob_{name}"] = column

    save_dataframe(output, output_path)
    console.kv("Predictions written", Path(str(output_path)).name)
    console.kv("Best model", dive.best_model_name_)
    console.print("")
    console.table(output.head(10))
    console.print("")
    console.success(f"Scored {len(output)} row(s) -> {output_path}")
    return {"output": Path(str(output_path)), "model": Path(str(model_path))}


def _check_schema(dive: Dive, frame: pd.DataFrame, console: Console) -> None:
    """Fail fast on schema mismatch rather than silently misaligning columns.

    Missing columns that were *dropped* during training (constants/ID-like) are
    allowed to be absent; anything else must be present. Extra columns are
    ignored with a notice - they cannot corrupt the prediction.
    """
    engineer = dive.feature_engineer_
    missing = engineer.missing_input_columns(frame)

    expected_dtypes = dive._metadata.get("input_dtypes", {})
    dtype_warnings: list = []

    if missing:
        hint_lines = [
            f"The model was trained with {len(engineer.input_columns_)} column(s): "
            f"{', '.join(map(str, engineer.input_columns_[:20]))}"
            + (" ..." if len(engineer.input_columns_) > 20 else "")
        ]
        raise SchemaError(
            f"Schema mismatch: {len(missing)} required feature column(s) missing "
            f"from the incoming data: {', '.join(map(str, missing))}",
            "\n".join(hint_lines),
        )

    # dtype mismatches are usually fixable by the pipeline (imputers, one-hot
    # encoder), so warn rather than fail - unless a numeric column became text.
    for column, trained_dtype in expected_dtypes.items():
        if column not in frame.columns:
            continue
        incoming_dtype = str(frame[column].dtype)
        trained_kind = _kind(trained_dtype)
        incoming_kind = _kind(incoming_dtype)
        if trained_kind != incoming_kind:
            dtype_warnings.append(f"  - {column}: trained as {trained_kind}, got {incoming_kind}")

    extra = [str(c) for c in frame.columns if str(c) not in set(engineer.input_columns_)]
    if extra:
        console.info(
            f"  {console.symbol('warn')} Extra column(s) present and ignored: "
            f"{', '.join(extra[:10])}"
        )
    if dtype_warnings:
        console.warn(
            "Column dtype mismatches vs. training (usually handled by the "
            "pipeline, but verify the values are sensible):\n" + "\n".join(dtype_warnings)
        )
    console.success("Schema check passed.")


def _kind(dtype: str) -> str:
    """Coarse dtype family used for mismatch detection."""
    dtype = str(dtype).lower()
    if any(token in dtype for token in ("int", "float")):
        return "numeric"
    if "bool" in dtype:
        return "boolean"
    if "datetime" in dtype:
        return "datetime"
    return "text"


def _sanity_check_predictions(predictions: np.ndarray, frame: pd.DataFrame) -> None:
    """Flag predictions whose magnitude implies a misaligned column.

    Catches the classic failure: data columns are in a different order (or
    shifted by one) than the training schema, so the model scores unrelated
    numbers. Only triggered for regression, where magnitudes are meaningful.
    """
    if predictions.dtype.kind not in "fi":
        return
    values = np.asarray(predictions, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    magnitude = float(np.nanmax(np.abs(finite)))
    if magnitude < _MIN_PLAUSIBLE_ABS:
        return
    raise SchemaError(
        f"Prediction values are implausibly large (max |value| = {magnitude:.3e}).",
        "The incoming columns are probably misaligned with the training schema. "
        "Verify the column order and dtypes of the data file.",
    )
