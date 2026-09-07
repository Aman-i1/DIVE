"""CLI Command logic for `dive models`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dive.registry import ModelRegistry
from dive.utils.logging import Console
from dive.utils.report import ReportBuilder


def run_models_list(console: Console, model_name: Optional[str] = None) -> None:
    registry = ModelRegistry()
    models = registry.list_models(model_name)
    if not models:
        console.info("No models registered yet. Register models using `dive models register`.")
        return

    builder = ReportBuilder("MODEL REGISTRY CATALOG", console=console)
    builder.table(
        ["Model", "Version", "Stage", "Created At"],
        [
            [
                entry.get("model_name", ""),
                entry.get("version", ""),
                str(entry.get("stage", "")).upper(),
                entry.get("created_at", ""),
            ]
            for entry in models
        ],
    )
    console.report(builder)


def run_models_register(
    console: Console, model_path: str, model_name: str, stage: str = "candidate"
) -> None:
    registry = ModelRegistry()
    console.info(f"Registering model artifact {model_path} as '{model_name}'...")
    ver_dir = registry.register_model(
        model_name=model_name,
        model_artifact_path=model_path,
        metrics={},
        schema={},
        stage=stage,
    )
    console.success(f"Registered {model_name} into registry at {ver_dir}")


def run_models_promote(
    console: Console, model_name: str, version: str, stage: str = "production"
) -> None:
    registry = ModelRegistry()
    console.info(f"Evaluating promotion gate for {model_name} {version} -> {stage}...")
    gate_check = registry.promote_model(model_name, version, target_stage=stage)
    console.print("")
    console.print(gate_check.render())
    if gate_check.approved:
        console.success(f"Successfully promoted {model_name} {version} to {stage.upper()}")
    else:
        console.error(f"Promotion rejected for {model_name} {version}")
