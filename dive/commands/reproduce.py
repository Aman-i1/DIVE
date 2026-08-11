"""CLI Command logic for `dive reproduce`."""

from __future__ import annotations

from dive.experiments import ExperimentTracker, ReproducibilityEngine
from dive.utils.logging import Console


def run_reproduce(console: Console, experiment_id: str, output_dir: str = "reproduce_bundle") -> None:
    console.rule(f"DIVE Reproducibility Engine - {experiment_id}")
    tracker = ExperimentTracker()
    engine = ReproducibilityEngine(tracker)

    console.info(f"Generating reproducibility manifest for {experiment_id}...")
    bundle_path = engine.export_reproduce_manifest(experiment_id, output_dir=output_dir)

    console.success(f"Reproducibility manifest bundle generated at {bundle_path}")
    console.print(f"  - {bundle_path}/experiment.json")
    console.print(f"  - {bundle_path}/environment.json")
    console.print(f"  - {bundle_path}/dataset_fingerprint.json")
