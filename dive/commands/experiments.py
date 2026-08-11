"""CLI Command logic for `dive experiments`."""

from __future__ import annotations

from typing import Any, List, Optional

from dive.experiments import ExperimentTracker
from dive.utils.logging import Console


def run_experiments_list(console: Console) -> None:
    tracker = ExperimentTracker()
    exps = tracker.list_experiments()
    if not exps:
        console.info("No experiments tracked yet. Run `dive train` to record experiments.")
        return

    console.rule("Tracked Experiments")
    lines = [f"{'ID':<12} {'Model':<16} {'Dataset Hash':<16} {'Time (s)':<10} {'RAM (MB)':<10}"]
    lines.append("-" * 68)
    for exp in exps:
        lines.append(
            f"{exp.get('experiment_id', ''):<12} "
            f"{exp.get('model_name', ''):<16} "
            f"{exp.get('dataset_hash', ''):<16} "
            f"{exp.get('training_time_seconds', 0):<10.1f} "
            f"{exp.get('peak_memory_mb', 0):<10.1f}"
        )
    console.print("\n".join(lines))


def run_experiments_show(console: Console, experiment_id: str) -> None:
    tracker = ExperimentTracker()
    exp = tracker.get_experiment(experiment_id)
    if not exp:
        console.error(f"Experiment '{experiment_id}' not found.")
        return

    console.rule(f"Experiment {experiment_id}")
    for k, v in exp.items():
        console.print(f"  {k:<24}: {v}")


def run_experiments_compare(console: Console, experiment_ids: List[str]) -> None:
    tracker = ExperimentTracker()
    df_cmp = tracker.compare_experiments(experiment_ids)
    if df_cmp.empty:
        console.error("No valid experiments found for comparison.")
        return

    console.rule("Experiment Comparison")
    console.print(df_cmp.to_string(index=False))
