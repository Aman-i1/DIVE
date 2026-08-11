"""Implementation of ``dive train``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dive.exceptions import ConfigError, DataError
from dive.utils.io import (
    ensure_dir,
    load_dataframe,
    resolve_path,
    save_dataframe,
    save_pickle,
    validate_target,
    write_text,
)
from dive.utils.logging import Console

# Files written into --output. Names are stable so scripts can depend on them.
MODEL_FILENAME = "model.pkl"
LEADERBOARD_FILENAME = "leaderboard.csv"
METADATA_FILENAME = "metadata.json"
REPORT_FILENAME = "report.html"
VALIDATION_FILENAME = "validation.json"
PLOTS_DIRNAME = "plots"
MODELS_DIRNAME = "models"
SCHEMA_FILENAME = "input_schema.json"
USAGE_FILENAME = "how_to_predict.py"

# Characters that are illegal in a Windows filename, plus whitespace.
_UNSAFE_FILENAME_CHARS = r'<>:"/\|?*'


def _slugify(text: str, fallback: str = "dataset") -> str:
    """Turn a dataset or model name into a safe, readable filename component."""
    cleaned = []
    for char in str(text).strip():
        if char in _UNSAFE_FILENAME_CHARS or char.isspace():
            cleaned.append("_")
        elif char.isalnum() or char in "-_.":
            cleaned.append(char)
    slug = "".join(cleaned).strip("._")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback


def _write_predictors(
    dive: Any,
    destination: Path,
    dataset_name: str,
    console: Console,
) -> Dict[str, Path]:
    """Write one raw-input predictor per trained model into ``models/``.

    Filenames are ``<dataset>__<model>.pkl`` so that artifacts from different
    datasets stay distinguishable once they are moved out of the output folder.
    A shared ``input_schema.json`` and a runnable ``how_to_predict.py`` land
    beside them, since the schema is identical across models.
    """
    from dive.utils.logging import Style

    try:
        predictors = dive.build_predictors(dataset_name=dataset_name)
    except Exception as exc:
        # Never fail a finished training run over an export problem: model.pkl
        # is already on disk and remains fully usable.
        console.warn(f"Could not export per-model predictors: {exc}")
        return {}

    if not predictors:
        return {}

    models_dir = ensure_dir(destination / MODELS_DIRNAME)
    dataset_slug = _slugify(dataset_name)
    paths: Dict[str, Path] = {}

    console.print("")
    console.rule("Per-model predictors")
    for name, predictor in predictors.items():
        filename = f"{dataset_slug}__{_slugify(name, fallback='model')}.pkl"
        try:
            paths[name] = save_pickle(predictor, models_dir / filename)
        except Exception as exc:
            console.warn(f"Could not write {filename}: {exc}")
            continue
        primary = ""
        for key in ("Accuracy", "R2", "F1", "RMSE"):
            if key in predictor.metrics and predictor.metrics[key] is not None:
                primary = f"{key}={predictor.metrics[key]:.4f}"
                break
        console.info(
            f"  {console.status_symbol('ok')} "
            f"{console.paint(f'{name:<20}', Style.MAGENTA)} "
            f"{console.paint(filename, Style.MUTED)}  {primary}"
        )

    if not paths:
        return {}

    sample = next(iter(predictors.values()))
    write_text(
        models_dir / SCHEMA_FILENAME,
        json.dumps(
            {
                "dataset": dataset_name,
                "target": sample.target,
                "problem_type": sample.problem_type,
                "classes": sample.class_names,
                "models": {
                    name: {"file": path.name, "metrics": predictors[name].metrics}
                    for name, path in paths.items()
                },
                "input_schema": sample.input_schema,
            },
            indent=2,
            default=str,
        ),
    )
    write_text(models_dir / USAGE_FILENAME, _usage_snippet(sample, paths))

    console.print("")
    console.info(
        f"  {len(paths)} predictor(s) in {models_dir}{os.sep} - each takes raw "
        f"{Path(str(dataset_name)).name} rows; see {USAGE_FILENAME}"
    )
    return paths


def _usage_snippet(predictor: Any, paths: Dict[str, Path]) -> str:
    """Generate a runnable script showing how to load and call a predictor."""
    example = predictor.input_schema.get("example_row") or {}
    first_file = next(iter(paths.values())).name
    rows = "\n".join(f"        {key!r}: {value!r}," for key, value in example.items())
    available = "\n".join(f"#   {name}: {path.name}" for name, path in paths.items())
    return f'''"""Predict with a model exported by `dive train`.

Each .pkl in this folder holds a DivePredictor: the fitted estimator bundled
with the feature engineering it was trained on. Pass it raw rows in the same
shape as {predictor.dataset_name!r} - the encoding is applied for you.

Available models:
{available}

Requires `dive` to be installed (pip install dive-ml).
"""
import pickle

import pandas as pd

with open("{first_file}", "rb") as handle:
    predictor = pickle.load(handle)

# What this model expects as input:
print(predictor.describe_input())

# Predict from a single raw row, exactly as it appears in the source file.
print(predictor.predict({{
{rows}
}}))

# Or from a whole raw file - the target column may be present or absent.
# frame = pd.read_csv("new_rows.csv")
# frame["predicted_{predictor.target}"] = predictor.predict(frame)
# frame.to_csv("predictions.csv", index=False)
'''


def run_train(
    console: Console,
    data_path: str,
    target: Optional[str],
    mode: str,
    time_budget: float,
    output_dir: str,
    test_size: float,
    cv_folds: Optional[int],
    random_state: int,
    time_series: bool,
    make_plots: bool,
    make_report: bool,
    run_validation: bool,
) -> Dict[str, Any]:
    """Validate inputs, train, and write every artifact into ``output_dir``."""
    if not data_path:
        raise ConfigError(
            "No data file given.",
            "Pass --data <path>, or set 'data:' in the file given to --config.",
        )
    _check_options(mode, time_budget, test_size, cv_folds)

    console.rule("dive train")
    # Resolve once: the raw string may carry the quotes a user pasted around a
    # path with spaces, and those must not leak into displayed names or the
    # dataset slug used for predictor filenames.
    resolved_data = resolve_path(data_path, must_exist=True, kind="data file")
    frame = load_dataframe(resolved_data)
    console.kv("Data file", resolved_data.name)
    console.kv("Rows x columns", f"{frame.shape[0]} x {frame.shape[1]}")

    if target is None:
        target = str(frame.columns[-1])
        console.warn(f"No --target given; using the last column: '{target}'")
    validate_target(frame, target)
    console.kv("Target", target)

    destination = ensure_dir(output_dir)
    console.kv("Output directory", destination)
    console.print("")

    # -- crosscheck suite before training ------------------------------
    validation_report = None
    if run_validation:
        from dive.validation import run_validation_suite

        console.rule("Pre-flight validation")
        validation_report = run_validation_suite(
            frame,
            target=target,
            test_size=test_size,
            random_state=random_state,
            time_series=time_series,
        )
        console.print(validation_report.render(console))
        write_text(
            destination / VALIDATION_FILENAME,
            json.dumps(validation_report.to_dict(), indent=2, default=str),
        )
        if validation_report.has_failures:
            console.warn(
                "Validation reported FAIL-level findings (see above). Training "
                "continues, but treat the resulting scores with suspicion."
            )
        console.print("")

    # -- train ----------------------------------------------------------
    from dive.core import Dive

    console.rule("Training")
    dive = Dive(
        target=target,
        mode=mode,
        time_budget=time_budget,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=random_state,
        time_series=time_series,
        console=console,
    )
    dive.fit(frame)

    # -- post-training checks -------------------------------------------
    from dive.validation import validate_trained_model

    validation_report = validate_trained_model(dive, validation_report)
    stability = validation_report.get("cv_stability")
    if stability is not None and stability.status != "SKIP":
        console.print("")
        console.print(f"  Stability: {stability.summary}")

    # -- artifacts ------------------------------------------------------
    console.print("")
    console.rule("Writing artifacts")
    written: Dict[str, Any] = {}

    model_path = dive.save(destination / MODEL_FILENAME)
    written["model"] = model_path
    console.kv("Model", model_path.name)

    leaderboard = dive.leaderboard()
    leaderboard_path = save_dataframe(leaderboard, destination / LEADERBOARD_FILENAME)
    written["leaderboard"] = leaderboard_path
    console.kv("Leaderboard", leaderboard_path.name)

    metadata = dict(dive._metadata)
    if validation_report is not None:
        metadata["validation"] = validation_report.to_dict()
        write_text(
            destination / VALIDATION_FILENAME,
            json.dumps(validation_report.to_dict(), indent=2, default=str),
        )
    metadata_path = write_text(
        destination / METADATA_FILENAME, json.dumps(metadata, indent=2, default=str)
    )
    written["metadata"] = metadata_path
    console.kv("Metadata", metadata_path.name)

    dataset_name = resolved_data.stem
    predictor_paths = _write_predictors(
        dive, destination, dataset_name=dataset_name, console=console
    )
    written["predictors"] = predictor_paths

    if make_plots:
        from dive.reporting import generate_plots

        plot_dir = ensure_dir(destination / PLOTS_DIRNAME)
        plots = generate_plots(dive, plot_dir, console=console)
        written["plots"] = plots
        console.kv("Plots", f"{len(plots)} PNG file(s) in {PLOTS_DIRNAME}/")

    if make_report:
        from dive.reporting import build_html_report

        report_path = build_html_report(
            dive,
            destination / REPORT_FILENAME,
            validation=validation_report,
            plots_dir=destination / PLOTS_DIRNAME if make_plots else None,
        )
        written["report"] = report_path
        console.kv("Report", report_path.name)

    console.print("")
    console.rule("Leaderboard")
    console.table(leaderboard, max_rows=15, highlight_first=True)
    console.print("")
    console.success(f"All artifacts written to: {destination}")
    console.print(
        f"  Next: dive predict --model {model_path} --data <new_rows.csv>"
    )
    return written


def _check_options(
    mode: str, time_budget: float, test_size: float, cv_folds: Optional[int]
) -> None:
    """Reject invalid option combinations before any data is read."""
    from dive.core import MODES

    if mode not in MODES:
        raise ConfigError(
            f"Unknown mode '{mode}'.", f"Valid modes are: {', '.join(MODES)}."
        )
    if time_budget <= 0:
        raise ConfigError(
            f"--time-budget must be greater than 0 (got {time_budget}).",
            "Try --time-budget 300 for a five-minute run.",
        )
    if not 0.05 <= test_size <= 0.5:
        raise ConfigError(
            f"--test-size must be between 0.05 and 0.5 (got {test_size}).",
            "This is the fraction of rows held out for evaluation.",
        )
    if cv_folds is not None and cv_folds < 2:
        raise ConfigError(
            f"--cv-folds must be at least 2 (got {cv_folds}).",
            "Cross-validation needs at least two folds.",
        )
