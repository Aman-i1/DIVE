"""Prediction Contract CLI Subcommand - `dive contract`."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import click
import pandas as pd

from dive.prediction_contract import PredictionContractEngine
from dive.utils.logging import Console, get_console


@click.command(name="contract", help="Establish and persist a formal Prediction Contract for a dataset.")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", "-t", required=True, help="Target column name to predict.")
@click.option("--entity", "-e", default=None, help="Entity/group column.")
@click.option("--time-column", default=None, help="Timestamp/date column.")
@click.option("--horizon", default=None, help="Prediction horizon (e.g. '30d', '1h').")
@click.option("--output", "-o", default="./contract.json", help="Path to save output JSON contract.")
def contract_command(
    data_path: str,
    target: str,
    entity: Optional[str],
    time_column: Optional[str],
    horizon: Optional[str],
    output: str,
) -> None:
    """Establish and inspect formal prediction contract."""
    console = get_console()
    from dive.utils.io import load_dataframe
    try:
        df = load_dataframe(data_path)
    except Exception as exc:
        console.error(f"Failed to read dataset: {exc}")
        sys.exit(1)

    engine = PredictionContractEngine()
    contract = engine.infer_contract(
        df=df,
        target=target,
        entity=entity,
        time_column=time_column,
        horizon=horizon,
    )

    console.print("\n" + contract.render() + "\n")
    contract.save(output)
    console.success(f"Prediction contract saved to: {output}")
