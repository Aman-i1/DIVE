"""Implementation of ``dive predict``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dive.core import Dive
from dive.exceptions import DataError, SchemaError
from dive.predictor import DivePredictor, SchemaMismatch
from dive.utils.io import load_dataframe, load_pickle, resolve_path, save_dataframe
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
    """Load a model, verify the incoming schema, score rows, write a CSV.

    Accepts either artifact ``dive train`` writes: the full-run ``model.pkl``
    or one of the per-model predictors under ``models/``. Both expose the same
    predict surface, so the only difference is where the schema check lives.
    """
    artifact = _load_artifact(model_path)

    frame = load_dataframe(data_path)
    console.rule("dive predict")
    console.kv("Model", resolve_path(model_path).name)
    console.kv("Incoming rows", frame.shape[0])
    console.kv("Incoming columns", frame.shape[1])

    if isinstance(artifact, DivePredictor):
        return _predict_with_predictor(
            console, artifact, frame, data_path, output_path,
            with_proba=with_proba, include_input=include_input,
            model_path=model_path,
        )

    dive = artifact
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


def _load_artifact(model_path: str) -> Any:
    """Return a ``Dive`` or a ``DivePredictor``, whichever the file holds."""
    payload = load_pickle(model_path)
    if isinstance(payload, DivePredictor):
        return payload
    return Dive.load(model_path)


def _predict_with_predictor(
    console: Console,
    predictor: DivePredictor,
    frame: pd.DataFrame,
    data_path: str,
    output_path: str,
    with_proba: bool,
    include_input: bool,
    model_path: str,
) -> Dict[str, Any]:
    """Score rows with a single exported model.

    The predictor validates the raw schema itself and raises ``SchemaMismatch``;
    it is translated here so the CLI reports it as an ordinary dive error with a
    hint rather than an unexpected internal failure.
    """
    try:
        predictions = predictor.predict(frame)
        probabilities = (
            predictor.predict_proba(frame)
            if with_proba and predictor.has_proba
            else None
        )
    except SchemaMismatch as exc:
        raise SchemaError(str(exc).split("\n")[0], predictor.describe_input()) from exc

    if with_proba and probabilities is None:
        raise SchemaError(
            f"--proba is not available for {predictor.model_name}.",
            f"This predictor solves a {predictor.problem_type} problem."
            if predictor.problem_type != "classification"
            else "The underlying model cannot produce probabilities.",
        )

    _sanity_check_predictions(predictions, frame)

    if include_input:
        output = frame.drop(
            columns=[predictor.target], errors="ignore"
        ).reset_index(drop=True)
    else:
        output = pd.DataFrame(index=range(len(predictions)))

    output.insert(0, "prediction", predictions)
    if probabilities is not None:
        for name in probabilities.columns:
            output[f"prob_{name}"] = probabilities[name].to_numpy()

    save_dataframe(output, output_path)
    console.kv("Predictions written", Path(str(output_path)).name)
    console.kv("Model", predictor.model_name)
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


def run_interactive_predict(
    console: Console,
    model_path: str,
    json_input: Optional[str] = None,
) -> None:
    """Interactive prediction machine - prompt user column by column or parse dict input."""
    import json
    artifact = _load_artifact(model_path)
    console.banner("DIVE INTERACTIVE PREDICTION MACHINE", f"Loaded prediction engine: {Path(model_path).name}")

    if isinstance(artifact, DivePredictor):
        predictor = artifact
    else:
        best_est = artifact.best_estimator_
        feat_eng = artifact.feature_engineer_
        feat_cols = artifact.feature_columns_
        target = artifact.target
        prob_type = artifact.problem_type
        label_enc = getattr(artifact, "label_encoder_", None)
        schema = artifact._metadata.get("schema", {})
        predictor = DivePredictor(
            model_name=artifact.best_model_name_,
            estimator=best_est,
            feature_engineer=feat_eng,
            feature_columns=feat_cols,
            label_encoder=label_enc,
            target=target,
            problem_type=prob_type,
            input_schema=schema,
        )

    schema = predictor.input_schema
    req_cols = predictor.required_columns or [c["name"] for c in schema.get("columns", []) if c.get("required")]

    row_data: Dict[str, Any] = {}

    if json_input:
        try:
            row_data = json.loads(json_input)
            console.success("Parsed input JSON dictionary.")
        except Exception as exc:
            raise SchemaError(f"Invalid JSON string in --input: {exc}")
    else:
        console.print("  Enter feature values below (press Enter to accept default example):")
        console.print("")
        cols_meta = {c["name"]: c for c in schema.get("columns", [])}

        for i, col in enumerate(req_cols, 1):
            meta = cols_meta.get(col, {})
            kind = meta.get("kind", "numeric")
            example = meta.get("example")
            cats = meta.get("categories")

            ex_prompt = f" [default: {example}]" if example is not None else ""
            cat_prompt = f" (choices: {cats[:4]})" if cats else ""

            prompt_str = f"  [{i}/{len(req_cols)}] Enter {col} ({kind}{cat_prompt}){ex_prompt}: "

            try:
                val = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("")
                console.warn("Input cancelled by user.")
                return

            if not val and example is not None:
                val = str(example)

            if kind == "numeric" and val:
                try:
                    val = float(val) if "." in val else int(val)
                except ValueError:
                    pass

            row_data[col] = val

    console.print("")
    console.rule("Prediction Output")

    pred = predictor.predict(row_data)[0]
    console.kv("Target Column", predictor.target)
    console.kv("Predicted Value / Class", str(pred))

    if predictor.has_proba:
        proba_df = predictor.predict_proba(row_data)
        top_prob = float(proba_df.iloc[0].max())
        top_cls = str(proba_df.iloc[0].idxmax())
        console.kv("Top Confidence Score", f"{top_prob * 100:.1f}% ({top_cls})")
        console.print("")
        console.print("  Probability Breakdown:")
        for cls_name, prob_val in proba_df.iloc[0].items():
            prob_val = float(prob_val)
            bar_len = int(prob_val * 25)
            bar_str = "█" * bar_len + "░" * (25 - bar_len)
            console.print(f"    • {cls_name:<16} : [{bar_str}] {prob_val * 100:.1f}%")

    console.print("")
    console.success("Prediction Machine execution complete.")

