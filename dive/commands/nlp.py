"""CLI Subcommand Group for DIVE NLP - `dive/commands/nlp.py`.

Provides user-facing terminal commands:
- `dive nlp profile <data_path>`: Profile document text characteristics, token distributions, and label contamination.
- `dive nlp train <data_path>`: Autonomous model search, candidate leaderboard, and predictor serialization.
- `dive nlp serve <model_path>`: Launch production REST API model server.
- `dive nlp monitor <ref_path> <curr_path>`: Evaluate production distribution shift, OOV rate, and drift alerts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from dive.nlp import (
    AutoNLP,
    NLPDataset,
    NLPDriftMonitor,
    NLPProfiler,
    load_nlp_predictor,
    save_nlp_predictor,
    serve_nlp_model,
)
from dive.utils.logging import get_console


@click.group("nlp")
def nlp_command() -> None:
    """Natural Language Processing (DIVE NLP) subcommands."""
    pass


@nlp_command.command("profile")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--text-col", "-x", default=None, help="Name of the text feature column.")
@click.option("--target-col", "-y", default=None, help="Name of the target label column.")
def profile_cmd(data_path: str, text_col: Optional[str], target_col: Optional[str]) -> None:
    """Profile NLP dataset, character/token distributions, and label audits."""
    console = get_console()
    console.rule("[bold cyan]DIVE NLP Dataset Profiling[/bold cyan]")
    ds = NLPDataset.from_file(data_path, text_column=text_col, target_column=target_col)
    profiler = NLPProfiler()
    report = profiler.profile(ds)
    console.print(report.render())


@nlp_command.command("train")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--target-col", "-y", default=None, help="Target label column.")
@click.option("--text-col", "-x", default=None, help="Text feature column.")
@click.option("--output", "-o", default="nlp_champion.pkl", help="Destination path for trained champion model.")
@click.option("--trials", "-n", default=5, type=int, help="Maximum number of candidate trials to evaluate.")
@click.option(
    "--optimize-for",
    type=click.Choice(["balanced", "accuracy", "latency"]),
    default="balanced",
    help="Multi-objective optimization criterion.",
)
def train_cmd(
    data_path: str,
    target_col: Optional[str],
    text_col: Optional[str],
    output: str,
    trials: int,
    optimize_for: str,
) -> None:
    """Autonomously evaluate representations and models, select champion, and serialize predictor."""
    console = get_console()
    console.rule("[bold cyan]DIVE AutoNLP Autonomous Search[/bold cyan]")
    engine = AutoNLP(max_trials=trials, optimize_for=optimize_for)
    predictor, leaderboard = engine.fit(
        data=data_path,
        target_column=target_col,
        text_column=text_col,
    )
    console.print(leaderboard.render())
    save_nlp_predictor(predictor, output)
    console.success(f"Champion predictor saved to: {output}")


@nlp_command.command("serve")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--host", default="127.0.0.1", help="Host interface to bind REST server.")
@click.option("--port", default=8000, type=int, help="Port to listen for requests.")
def serve_cmd(model_path: str, host: str, port: int) -> None:
    """Launch production REST API server for a saved NLP predictor."""
    console = get_console()
    console.rule(f"[bold cyan]Launching DIVE NLP Model Server on {host}:{port}[/bold cyan]")
    predictor = load_nlp_predictor(model_path)
    serve_nlp_model(predictor, host=host, port=port)


@nlp_command.command("monitor")
@click.argument("ref_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("curr_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--text-col", "-x", default=None, help="Text column name.")
@click.option("--oov-threshold", default=0.15, type=float, help="OOV rate alert threshold.")
def monitor_cmd(ref_path: str, curr_path: str, text_col: Optional[str], oov_threshold: float) -> None:
    """Audit production distribution shift, length drift, and vocabulary OOV rate."""
    console = get_console()
    console.rule("[bold cyan]DIVE NLP Distribution Drift Audit[/bold cyan]")
    ref_ds = NLPDataset.from_file(ref_path, text_column=text_col)
    curr_ds = NLPDataset.from_file(curr_path, text_column=text_col)

    monitor = NLPDriftMonitor(reference_texts=ref_ds.texts, oov_threshold=oov_threshold)
    report = monitor.check_drift(current_texts=curr_ds.texts)
    console.print(report.render())
