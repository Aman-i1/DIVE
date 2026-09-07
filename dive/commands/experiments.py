"""CLI Command logic for `dive experiments`."""

from __future__ import annotations

from typing import Any, List, Optional

from dive.experiments import ExperimentTracker
from dive.utils.logging import Console
from dive.utils.report import ReportBuilder


def run_experiments_list(console: Console) -> None:
    tracker = ExperimentTracker()
    exps = tracker.list_experiments()
    if not exps:
        console.info("No experiments tracked yet. Run `dive train` to record experiments.")
        return

    builder = ReportBuilder("TRACKED EXPERIMENTS", console=console)
    builder.table(
        ["ID", "Model", "Dataset Hash", "Time (s)", "RAM (MB)"],
        [
            [
                exp.get("experiment_id", ""),
                exp.get("model_name", ""),
                exp.get("dataset_hash", ""),
                f"{exp.get('training_time_seconds', 0):.1f}",
                f"{exp.get('peak_memory_mb', 0):.1f}",
            ]
            for exp in exps
        ],
    )
    console.report(builder)


def run_experiments_show(console: Console, experiment_id: str) -> None:
    tracker = ExperimentTracker()
    exp = tracker.get_experiment(experiment_id)
    if not exp:
        console.error(f"Experiment '{experiment_id}' not found.")
        return

    builder = ReportBuilder(f"EXPERIMENT {experiment_id}", console=console)
    builder.kvs(exp)
    console.report(builder)


def run_experiments_compare(console: Console, experiment_ids: List[str]) -> None:
    tracker = ExperimentTracker()
    df_cmp = tracker.compare_experiments(experiment_ids)
    if df_cmp.empty:
        console.error("No valid experiments found for comparison.")
        return

    builder = ReportBuilder("EXPERIMENT COMPARISON", console=console)
    builder.dataframe(df_cmp, max_rows=len(df_cmp))
    console.report(builder)
