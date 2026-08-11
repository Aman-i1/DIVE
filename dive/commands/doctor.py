"""CLI Command logic for `dive doctor`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dive.doctor import DiveDoctor
from dive.utils.io import load_dataframe, save_json
from dive.utils.logging import Console


def run_doctor(
    console: Console,
    data_path: str,
    target: str,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """Run ML Doctor audit on data_path."""
    console.rule("DIVE ML Doctor Diagnostic Audit")
    console.info(f"Loading data from {data_path}...")
    df = load_dataframe(data_path)

    console.info(f"Analyzing dataset readiness relative to target '{target}'...")
    doctor = DiveDoctor(
        target=target,
        group_column=group_column,
        time_column=time_column,
    )
    report = doctor.analyze(df)

    console.print("")
    console.print(report.render_text())
    console.print("")

    if output_path:
        save_json(output_path, report.to_dict())
        console.success(f"Wrote diagnostic report JSON to {output_path}")
