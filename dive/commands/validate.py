"""Implementation of ``dive validate``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dive.utils.io import load_dataframe, validate_target, write_text
from dive.utils.logging import Console
from dive.validation import FAIL, WARN, run_validation_suite


def run_validate(
    console: Console,
    data_path: str,
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    time_series: bool = False,
    output_path: Optional[str] = None,
    strict: bool = False,
) -> int:
    """Run the crosscheck suite with no training and print a pass/warn/fail report.

    Returns the process exit code: 0 when clean (or only warnings), 1 when a
    check failed, or when ``--strict`` promotes warnings to failures.
    """
    frame = load_dataframe(data_path)

    console.rule("dive validate")
    console.kv("Data file", Path(str(data_path)).name)
    console.kv("Rows x columns", f"{frame.shape[0]} x {frame.shape[1]}")

    if target is not None:
        validate_target(frame, target)
        console.kv("Target", target)
    else:
        console.warn(
            "No --target given - only structural checks will run. "
            "Pass --target <column> to check for leakage, imbalance, and drift."
        )
    console.print("")

    report = run_validation_suite(
        frame,
        target=target,
        test_size=test_size,
        random_state=random_state,
        time_series=time_series,
    )

    console.print(report.render(console))
    console.print("")

    if output_path:
        import json

        written = write_text(
            output_path, json.dumps(report.to_dict(), indent=2, default=str)
        )
        console.kv("Report written", written)

    if report.has_failures:
        console.error(
            "Validation FAILED. Fix the issues above before trusting any model "
            "trained on this data."
        )
        return 1
    if report.has_warnings:
        console.warn(
            "Validation passed with warnings. Training will work, but read the "
            "notes above when interpreting the scores."
        )
        return 1 if strict else 0

    console.success("All checks passed - this dataset is ready to train on.")
    return 0
