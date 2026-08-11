"""CLI Command logic for `dive info`."""

from __future__ import annotations

from typing import Optional

from dive.info import DatasetInspector
from dive.utils.io import load_dataframe, save_json
from dive.utils.logging import Console


def run_info(
    console: Console,
    data_path: str,
    output_path: Optional[str] = None,
) -> None:
    """Run targetless dataset inspection on data_path."""
    console.banner("DIVE DATASET INSPECTOR", f"Profiling unknown dataset: {data_path}")

    with console.spinner(f"Analyzing dataset structure for {data_path}..."):
        df = load_dataframe(data_path)
        inspector = DatasetInspector()
        report = inspector.inspect(df)

    console.print("")
    console.print(report.render())
    console.print("")

    if output_path:
        save_json(output_path, report.to_dict())
        console.success(f"Wrote dataset info report JSON to {output_path}")
