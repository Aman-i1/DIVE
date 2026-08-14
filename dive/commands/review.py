"""Senior ML Review CLI Subcommand - `dive review`."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Optional

import click
import pandas as pd

from dive.autopilot import AutopilotOrchestrator
from dive.utils.logging import Console, get_console


@click.command(name="review", help="Perform comprehensive Senior ML Reliability review on an experiment or dataset.")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", "-t", required=True, help="Target column name.")
@click.option("--entity", "-e", default=None, help="Entity identifier column.")
@click.option("--time-column", default=None, help="Timestamp column.")
@click.option("--output", "-o", default="./review_report.json", help="Path to save output JSON review.")
def review_command(
    data_path: str,
    target: str,
    entity: Optional[str],
    time_column: Optional[str],
    output: str,
) -> None:
    """Execute Senior ML Review on dataset."""
    console = get_console()
    try:
        if data_path.endswith((".parquet", ".pq")):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
    except Exception as exc:
        console.error(f"Failed to read dataset: {exc}")
        sys.exit(1)

    orchestrator = AutopilotOrchestrator(
        target=target,
        mode="fast",
        time_budget="60s",
        entity_column=entity,
        time_column=time_column,
        console=console,
    )
    result = orchestrator.run(df)
    result.senior_review.save(output)
    console.success(f"Senior Review report saved to: {output}")
