"""Autonomous AutoML CLI Subcommand - `dive auto`.

End-to-end execution orchestrating all 20 industrial-grade ML domain engines:
Resource checks -> Data audit -> Validation Intelligence -> Meta-Learning ->
Feature Engineering -> ASHA Search -> Calibrated Stacking -> Trust Report ->
Artifact Store -> Standalone Reproducibility Bundle.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Optional

import click
import pandas as pd

from dive.config import DiveConfig
from dive.orchestration import StudyConfig, StudyOrchestrator
from dive.reproducibility import ReproducibilityBundleExporter
from dive.study import create_study
from dive.utils.logging import Console, get_console


@click.command(name="auto", help="Execute end-to-end autonomous AutoML workflow across all 20 domain engines.")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", "-t", required=False, help="Target column name to predict.")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fast", "balanced", "thorough", "competition"]),
    default="balanced",
    help="Execution search intensity mode.",
)
@click.option("--budget", "-b", default="300s", help="Time budget (e.g. '30s', '10m', '1h').")
@click.option("--output", "-o", default="./dive_output", help="Output directory for artifacts & bundles.")
@click.option("--config", "-c", "config_file", type=click.Path(exists=True), help="Optional path to dive.yaml / dive.json.")
def auto_command(
    data_path: str,
    target: Optional[str],
    mode: str,
    budget: str,
    output: str,
    config_file: Optional[str],
) -> None:
    """Run full industrial autonomous AutoML workflow."""
    console = get_console()
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse configuration if provided
    if config_file:
        cfg = DiveConfig.load(config_file)
        target = target or cfg.target
        mode = mode or cfg.mode

    if not target:
        console.error("Target column must be specified via --target <col> or in config file.")
        sys.exit(2)

    console.banner("DIVE INDUSTRIAL AUTONOMOUS ENGINE", f"Dataset: {Path(data_path).name} | Target: '{target}' | Mode: {mode}")

    # Load dataset
    try:
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith((".parquet", ".pq")):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
    except Exception as exc:
        console.error(f"Failed to load dataset: {exc}")
        sys.exit(1)

    # Execute study orchestration
    study = create_study(
        data=df,
        target=target,
        mode=mode,
        time_budget=budget,
        output_dir=out_dir,
    )
    study.fit()

    # Generate Standalone Reproducibility Bundle
    if study.best_estimator is not None:
        exporter = ReproducibilityBundleExporter(experiment_id="dive_study_auto")
        bundle_path = exporter.export_bundle(
            output_dir=out_dir,
            model=study.best_estimator,
            target=target,
            problem_type=study.problem_type or "classification",
        )
        console.info(f"Reproducibility bundle created at: {bundle_path}")

    console.success("Autonomous AutoML execution finished successfully.")
