"""CLI Command logic for `dive drift`."""

from __future__ import annotations

from typing import Optional

from dive.drift import DriftDetector
from dive.utils.io import load_dataframe, save_json
from dive.utils.logging import Console


def run_drift(
    console: Console,
    reference_path: str,
    current_path: str,
    output_path: Optional[str] = None,
) -> None:
    console.rule("DIVE Production Data Drift Analysis")
    console.info(f"Loading reference baseline data from {reference_path}...")
    ref_df = load_dataframe(reference_path)

    console.info(f"Loading current production data from {current_path}...")
    curr_df = load_dataframe(current_path)

    detector = DriftDetector()
    report = detector.analyze_drift(ref_df, curr_df)

    console.print("")
    console.print(report.render())
    console.print("")

    if output_path:
        save_json(output_path, report.to_dict())
        console.success(f"Wrote drift report JSON to {output_path}")
