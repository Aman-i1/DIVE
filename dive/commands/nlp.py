"""CLI Subcommand Group for DIVE NLP - `dive/commands/nlp.py`.

Provides user-facing terminal commands for Natural Language Processing:
- `dive nlp info <data_path>`: Quick inspection of dataset schema, detected text/target columns, and sample preview.
- `dive nlp profile <data_path>`: Deep profiling of document lengths, token distributions, vocabulary, and label contamination.
- `dive nlp train <data_path>`: Autonomous AutoNLP search across representations and models, selecting the champion.
- `dive nlp predict <model_path>`: Score new text datasets or run interactive terminal prediction.
- `dive nlp serve <model_path>`: Launch production REST API model server with Swagger UI.
- `dive nlp monitor <ref_path> <curr_path>`: Audit production distribution shift, length drift, and vocabulary OOV rate.
- `dive nlp benchmark <model_path>`: Benchmark latency percentiles (p50, p95, p99) and throughput.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd

from dive.nlp import (
    AutoNLP,
    NLPDataset,
    NLPDriftMonitor,
    NLPProfiler,
    load_nlp_predictor,
    save_nlp_predictor,
    serve_nlp_model,
)
from dive.nlp.optimization.onnx import export_nlp_to_onnx
from dive.utils.io import load_dataframe
from dive.utils.logging import get_console


@click.group("nlp")
def nlp_command() -> None:
    """Natural Language Processing (DIVE NLP) subcommands.

    \b
    Quickstart Examples:
      # 1. Quick dataset inspection (columns, text/target candidates, preview)
      dive nlp info spam.csv

      # 2. Deep text profiling & vocabulary diagnostics
      dive nlp profile spam.csv -x text -y label

      # 3. Autonomous AutoNLP training & model selection
      dive nlp train spam.csv --trials 5 --output spam_model.pkl

      # 4. Predict on a new CSV file or interactive terminal prompt
      dive nlp predict spam_model.pkl --data new_messages.csv --output preds.csv
      dive nlp predict spam_model.pkl --text "Congratulations! You won a $1,000 gift card!"
      dive nlp predict spam_model.pkl  # interactive prompt mode

      # 5. Serve model as high-performance REST API
      dive nlp serve spam_model.pkl --port 8000

      # 6. Monitor production distribution shift & OOV vocabulary drift
      dive nlp monitor baseline.csv production.csv -x text
    """
    pass


# ----------------------------------------------------------------------
# 1. dive nlp info
# ----------------------------------------------------------------------
@nlp_command.command("info")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
def info_cmd(data_path: str) -> None:
    """Inspect dataset schema, detected text/target columns, and sample preview.

    \b
    Examples:
      dive nlp info reviews.csv
      dive nlp info comments.jsonl
      dive nlp info data/dataset.parquet
    """
    console = get_console()
    console.rule("[bold cyan]DIVE NLP Dataset Inspector[/bold cyan]")

    ds = NLPDataset.from_file(data_path)
    df = ds.to_dataframe()

    console.print(f"[bold green]✓[/bold green] Source File: [cyan]{data_path}[/cyan]")
    console.print(f"[bold green]✓[/bold green] Total Rows: [cyan]{len(df):,}[/cyan]")
    console.print(f"[bold green]✓[/bold green] Columns: [cyan]{', '.join(df.columns)}[/cyan]")
    console.print(f"[bold green]✓[/bold green] Detected Text Column: [bold yellow]{'text' if 'text' in df.columns else df.columns[0]}[/bold yellow]")
    if ds.has_labels:
        unique_labels = sorted(list(set(ds.labels)))
        n_classes = len(unique_labels)
        console.print(f"[bold green]✓[/bold green] Detected Target Column: [bold yellow]{'label' if 'label' in df.columns else 'labels'}[/bold yellow] ([cyan]{n_classes} classes[/cyan]: {unique_labels[:8]}{'...' if n_classes > 8 else ''})")
    else:
        console.print("[bold yellow]![/bold yellow] Target Column: [italic dim]None (unsupervised / raw text collection)[/italic dim]")

    stats = ds.summary_stats()
    console.print(f"[bold green]✓[/bold green] Document Lengths: Avg [cyan]{stats['avg_word_count']} words[/cyan] (Median: [cyan]{stats['median_word_count']}[/cyan]), Avg [cyan]{stats['avg_char_length']} chars[/cyan]")

    console.print("\n[bold]Preview Sample Records:[/bold]")
    sample_preview = df.head(3)
    for idx, row in sample_preview.iterrows():
        txt_snippet = str(row.get("text", row.iloc[0]))[:100].replace("\n", " ")
        lbl_info = f" | [bold magenta]Target:[/bold magenta] {row.get('label', row.iloc[1])}" if ds.has_labels and len(row) > 1 else ""
        console.print(f"  [dim]#{idx + 1}[/dim] {txt_snippet}...{lbl_info}")

    console.print("\n[bold cyan]Suggested Next Commands:[/bold cyan]")
    console.print(f"  [dim]# 1. Run deep text profiling report:[/dim]")
    console.print(f"  dive nlp profile \"{data_path}\"")
    console.print(f"  [dim]# 2. Train champion AutoNLP model:[/dim]")
    console.print(f"  dive nlp train \"{data_path}\" --trials 5 --output champion.pkl\n")


# ----------------------------------------------------------------------
# 2. dive nlp profile
# ----------------------------------------------------------------------
@nlp_command.command("profile")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--text-col", "-x", default=None, help="Name of the text feature column.")
@click.option("--target-col", "-y", default=None, help="Name of the target label column.")
def profile_cmd(data_path: str, text_col: Optional[str], target_col: Optional[str]) -> None:
    """Profile NLP dataset, character/token distributions, and label audits.

    \b
    Examples:
      dive nlp profile dataset.csv
      dive nlp profile reviews.tsv -x review_text -y sentiment
      dive nlp profile tickets.jsonl --text-col description --target-col category
    """
    console = get_console()
    console.rule("[bold cyan]DIVE NLP Dataset Profiling[/bold cyan]")
    ds = NLPDataset.from_file(data_path, text_column=text_col, target_column=target_col)
    profiler = NLPProfiler()
    report = profiler.profile(ds)
    console.print(report.render())


# ----------------------------------------------------------------------
# 3. dive nlp train
# ----------------------------------------------------------------------
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
    help="Multi-objective optimization criterion (balanced, accuracy, latency).",
)
def train_cmd(
    data_path: str,
    target_col: Optional[str],
    text_col: Optional[str],
    output: str,
    trials: int,
    optimize_for: str,
) -> None:
    """Autonomously evaluate representations and models, select champion, and serialize predictor.

    \b
    Examples:
      dive nlp train dataset.csv
      dive nlp train dataset.csv --trials 10 --optimize-for accuracy --output ./best_model.pkl
      dive nlp train reviews.tsv -x text -y sentiment --trials 5 --output model.pkl
    """
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


# ----------------------------------------------------------------------
# 4. dive nlp predict
# ----------------------------------------------------------------------
@nlp_command.command("predict")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", "data_path", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to input dataset file (CSV, JSON, Parquet).")
@click.option("--text", "-t", "single_text", default=None, help="Single text string to score directly from terminal.")
@click.option("--text-col", "-x", default=None, help="Text column name in data file.")
@click.option("--output", "-o", "output_path", default=None, help="Optional output CSV path to write predictions.")
@click.option("--proba", is_flag=True, default=False, help="Include class probability distributions in output.")
def predict_cmd(
    model_path: str,
    data_path: Optional[str],
    single_text: Optional[str],
    text_col: Optional[str],
    output_path: Optional[str],
    proba: bool,
) -> None:
    """Score new text datasets or run interactive terminal prediction machine.

    \b
    Examples:
      # Predict on a batch file and save output CSV
      dive nlp predict model.pkl --data new_data.csv --output predictions.csv

      # Predict on a single string
      dive nlp predict model.pkl --text "Amazing fast shipping and high quality!"

      # Interactive terminal prediction machine
      dive nlp predict model.pkl
    """
    console = get_console()
    console.rule("[bold cyan]DIVE NLP Prediction Engine[/bold cyan]")

    predictor = load_nlp_predictor(model_path)
    model_name = getattr(predictor, "model_name", "NLPPredictor")
    console.print(f"[bold green]✓[/bold green] Loaded Predictor: [cyan]{model_path}[/cyan] ([dim]{model_name}[/dim])")

    # 1. Single text prediction
    if single_text is not None:
        pred = predictor.predict([single_text])[0]
        console.print(f"\n[bold]Input Text:[/bold] {single_text}")
        console.print(f"[bold green]Predicted Label:[/bold green] [bold yellow]{pred}[/bold yellow]")
        if proba and predictor.has_proba:
            probabilities = predictor.predict_proba([single_text])[0]
            classes = predictor.class_names or [f"class_{i}" for i in range(len(probabilities))]
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probabilities)}
            console.print(f"[bold cyan]Probabilities:[/bold cyan] {prob_dict}")
        return

    # 2. Batch dataset prediction
    if data_path is not None:
        ds = NLPDataset.from_file(data_path, text_column=text_col)
        console.print(f"[bold green]✓[/bold green] Scoring [cyan]{len(ds):,}[/cyan] documents...")
        start_t = time.perf_counter()
        preds = predictor.predict(ds.texts)
        dur_ms = (time.perf_counter() - start_t) * 1000.0
        throughput = len(ds.texts) / max(dur_ms / 1000.0, 1e-6)

        df_out = ds.to_dataframe()
        df_out["predicted_label"] = preds

        if proba and predictor.has_proba:
            probs = predictor.predict_proba(ds.texts)
            classes = predictor.class_names or [f"class_{i}" for i in range(probs.shape[1])]
            for idx, c in enumerate(classes):
                df_out[f"proba_{c}"] = probs[:, idx]

        console.success(f"Scored {len(ds):,} documents in {dur_ms:.1f} ms ({throughput:,.1f} docs/sec)")

        if output_path is not None:
            df_out.to_csv(output_path, index=False)
            console.success(f"Saved predictions to: {output_path}")
        else:
            console.print("\n[bold]Sample Predictions Preview:[/bold]")
            for i in range(min(5, len(df_out))):
                txt_preview = str(ds.texts[i])[:70].replace("\n", " ")
                console.print(f"  [dim]#{i + 1}[/dim] \"{txt_preview}...\" -> [bold yellow]{preds[i]}[/bold yellow]")
        return

    # 3. Interactive prediction prompt
    console.print("\n[bold cyan]Interactive NLP Prediction Machine[/bold cyan]")
    console.print("[dim]Type any text and press Enter to score. Type 'exit' or 'quit' to stop.[/dim]\n")

    while True:
        try:
            line = input("dive-nlp > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting interactive machine.")
            break

        if not line or line.lower() in ("exit", "quit", "q"):
            console.print("Exiting interactive machine.")
            break

        pred = predictor.predict([line])[0]
        prob_str = ""
        if predictor.has_proba:
            probabilities = predictor.predict_proba([line])[0]
            classes = predictor.classes_
            top_prob = max(probabilities)
            prob_str = f" [dim](confidence: {top_prob:.1%})[/dim]"
        console.print(f"  --> [bold yellow]{pred}[/bold yellow]{prob_str}\n")


# ----------------------------------------------------------------------
# 5. dive nlp serve
# ----------------------------------------------------------------------
@nlp_command.command("serve")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--host", default="127.0.0.1", help="Host interface to bind REST server.")
@click.option("--port", default=8000, type=int, help="Port to listen for requests.")
def serve_cmd(model_path: str, host: str, port: int) -> None:
    """Launch production REST API server for a saved NLP predictor.

    \b
    Endpoints provided:
      POST /nlp/predict         - Single/Batch document scoring
      POST /nlp/predict_proba   - Calibrated class probabilities
      POST /nlp/batch_predict   - High-throughput batch inference
      GET  /health              - Live health probe
      GET  /metrics             - Prometheus-style latency & request metrics
      GET  /docs                - Interactive Swagger UI

    \b
    Examples:
      dive nlp serve champion.pkl --port 8000
      dive nlp serve model.pkl --host 0.0.0.0 --port 8080
    """
    console = get_console()
    console.rule(f"[bold cyan]Launching DIVE NLP Model Server on {host}:{port}[/bold cyan]")
    predictor = load_nlp_predictor(model_path)
    serve_nlp_model(predictor, host=host, port=port)


# ----------------------------------------------------------------------
# 6. dive nlp monitor
# ----------------------------------------------------------------------
@nlp_command.command("monitor")
@click.argument("ref_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("curr_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--text-col", "-x", default=None, help="Text column name.")
@click.option("--oov-threshold", default=0.15, type=float, help="OOV rate alert threshold.")
def monitor_cmd(ref_path: str, curr_path: str, text_col: Optional[str], oov_threshold: float) -> None:
    """Audit production distribution shift, length drift, and vocabulary OOV rate.

    \b
    Examples:
      dive nlp monitor baseline.csv production.csv
      dive nlp monitor train_ref.tsv inference_stream.tsv -x text --oov-threshold 0.10
    """
    console = get_console()
    console.rule("[bold cyan]DIVE NLP Distribution Drift Audit[/bold cyan]")
    ref_ds = NLPDataset.from_file(ref_path, text_column=text_col)
    curr_ds = NLPDataset.from_file(curr_path, text_column=text_col)

    monitor = NLPDriftMonitor(reference_texts=ref_ds.texts, oov_threshold=oov_threshold)
    report = monitor.check_drift(current_texts=curr_ds.texts)
    console.print(report.render())


# ----------------------------------------------------------------------
# 7. dive nlp benchmark
# ----------------------------------------------------------------------
@nlp_command.command("benchmark")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", "data_path", default=None, type=click.Path(exists=True, dir_okay=False), help="Test dataset path for benchmarking.")
@click.option("--samples", "-n", default=100, type=int, help="Number of test iterations.")
def benchmark_cmd(model_path: str, data_path: Optional[str], samples: int) -> None:
    """Benchmark prediction latency percentiles (p50, p95, p99) and throughput.

    \b
    Examples:
      dive nlp benchmark champion.pkl
      dive nlp benchmark champion.pkl --data test.csv --samples 200
    """
    console = get_console()
    console.rule("[bold cyan]DIVE NLP Model Latency & Throughput Benchmark[/bold cyan]")

    predictor = load_nlp_predictor(model_path)
    console.print(f"[bold green]✓[/bold green] Loaded Predictor: [cyan]{model_path}[/cyan]")

    if data_path is not None:
        ds = NLPDataset.from_file(data_path)
        texts = ds.texts[:samples]
    else:
        texts = [
            "This is a standard test sentence designed to benchmark text scoring latency.",
            "Another short sentence for performance measurements across NLP pipeline components.",
            "Fast, reliable and accurate inference latency benchmarking across model representations.",
        ] * (samples // 3 + 1)
        texts = texts[:samples]

    # Warmup
    _ = predictor.predict(texts[:5])

    latencies: List[float] = []
    for text in texts:
        t0 = time.perf_counter()
        _ = predictor.predict([text])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    avg_lat = float(np.mean(latencies))
    throughput = 1000.0 / max(avg_lat, 1e-6)

    console.print("\n[bold]Latency & Throughput Profile:[/bold]")
    console.print(f"  • [bold cyan]p50 (Median Latency):[/bold cyan]  {p50:.2f} ms/doc")
    console.print(f"  • [bold cyan]p95 Latency:[/bold cyan]          {p95:.2f} ms/doc")
    console.print(f"  • [bold cyan]p99 Latency:[/bold cyan]          {p99:.2f} ms/doc")
    console.print(f"  • [bold cyan]Single-Thread Rate:[/bold cyan]   {throughput:,.1f} docs/sec\n")
