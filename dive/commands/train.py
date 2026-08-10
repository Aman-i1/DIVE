"""Implementation of ``dive train``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from dive.exceptions import ConfigError, DataError
from dive.utils.io import (
    ensure_dir,
    load_dataframe,
    save_dataframe,
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
    frame = load_dataframe(data_path)
    console.kv("Data file", Path(str(data_path)).name)
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
    console.table(leaderboard, max_rows=15)
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
