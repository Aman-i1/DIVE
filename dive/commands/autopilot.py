"""Autonomous Senior ML Autopilot CLI Subcommand - `dive autopilot`."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import click
import pandas as pd

from dive.autopilot import AutopilotOrchestrator
from dive.utils.logging import Console, get_console


@click.command(name="autopilot", help="Execute complete 20-step Senior ML Review + Reliability + AutoML Autopilot.")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", "-t", required=True, help="Target column name to predict.")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fast", "balanced", "thorough", "competition"]),
    default="balanced",
    help="Execution search intensity mode.",
)
@click.option("--budget", "-b", default="300s", help="Time budget (e.g. '30s', '10m', '1h').")
@click.option("--entity", "-e", "entity_column", default=None, help="Entity/group identifier column.")
@click.option("--time-column", default=None, help="Timestamp/date column.")
@click.option("--output", "-o", default="./dive_autopilot_out", help="Output directory for artifacts & reviews.")
def autopilot_command(
    data_path: str,
    target: str,
    mode: str,
    budget: str,
    entity_column: Optional[str],
    time_column: Optional[str],
    output: str,
) -> None:
    """Run Senior ML Autopilot workflow."""
    console = get_console()
    from dive.utils.io import load_dataframe
    try:
        df = load_dataframe(data_path)
    except Exception as exc:
        console.error(f"Failed to read dataset: {exc}")
        sys.exit(1)

    orchestrator = AutopilotOrchestrator(
        target=target,
        mode=mode,
        time_budget=budget,
        entity_column=entity_column,
        time_column=time_column,
        output_dir=output,
        console=console,
    )
    result = orchestrator.run(df)

    if result.senior_review.final_decision == "BLOCKED":
        console.error("Autopilot review resulted in BLOCKED status due to critical reliability risks.")
        sys.exit(1)
    else:
        console.success(f"Autopilot finished with verdict: [{result.senior_review.final_decision}]")
