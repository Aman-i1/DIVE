"""CLI Command logic for `dive models`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dive.registry import ModelRegistry
from dive.utils.logging import Console


def run_models_list(console: Console, model_name: Optional[str] = None) -> None:
    registry = ModelRegistry()
    models = registry.list_models(model_name)
    if not models:
        console.info("No models registered yet. Register models using `dive models register`.")
        return

    console.rule("Model Registry Catalog")
    lines = [f"{'Model':<16} {'Version':<10} {'Stage':<12} {'Created At':<22}"]
    lines.append("-" * 62)
    for m in models:
        lines.append(
            f"{m.get('model_name', ''):<16} "
            f"{m.get('version', ''):<10} "
            f"{m.get('stage', '').upper():<12} "
            f"{m.get('created_at', ''):<22}"
        )
    console.print("\n".join(lines))


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
