"""CLI Subcommand Group for DIVE NLP - `dive/commands/nlp.py`.

Provides user-facing terminal commands for Natural Language Processing:
- `dive nlp info <data_path>`: Quick inspection of dataset schema, detected text/target columns, and sample preview.
- `dive nlp profile <data_path>`: Deep profiling of document lengths, token distributions, vocabulary, and label contamination.
- `dive nlp train <data_path>`: Autonomous AutoNLP search across representations and models, selecting the champion.
- `dive nlp predict <model_path>`: Score new text datasets or run interactive terminal prediction.
- `dive nlp serve <model_path>`: Launch production REST API model server with Swagger UI.
- `dive nlp monitor <ref_path> <curr_path>`: Audit production distribution shift, length drift, and vocabulary OOV rate.
- `dive nlp benchmark <model_path>`: Benchmark latency percentiles (p50, p95, p99) and throughput.

Output goes through :class:`~dive.utils.logging.Console` and
:class:`~dive.utils.report.ReportBuilder` - the same primitives the tabular ML
domain uses - so both domains render one grammar. Every command resolves its
console from the click context so the root ``--quiet`` flag is honoured.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import click
import numpy as np

from dive.nlp import (
    AutoNLP,
    NLPDataset,
    NLPDriftMonitor,
    NLPProfiler,
    load_nlp_predictor,
    save_nlp_predictor,
    serve_nlp_model,
)
from dive.commands._context import console_from
from dive.utils.logging import Console
from dive.utils.report import ReportBuilder


def _console(ctx: click.Context) -> Console:
    """Resolve the console from the root context so ``--quiet`` is respected.

    Delegates to the shared helper in :mod:`dive.commands._context`. Calling bare
    ``get_console()`` here is what used to make ``dive --quiet nlp ...`` print
    everything anyway.
    """
    return console_from(ctx)


def _class_names(predictor: Any, n_classes: int) -> List[str]:
    """Resolve display names for probability columns.

    ``class_names`` and ``classes_`` are both optional on a predictor and may be
    ``None``, which previously raised in the interactive branch.
    """
    for attribute in ("class_names", "classes_"):
        candidate = getattr(predictor, attribute, None)
        if candidate is not None and len(candidate):
            return [str(name) for name in candidate]
    return [f"class_{index}" for index in range(n_classes)]


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
@click.argument("data_path", required=False, default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to input dataset file.")
@click.pass_context
def info_cmd(ctx: click.Context, data_path: Optional[str], data: Optional[str]) -> None:
    """Inspect dataset schema, detected text/target columns, and sample preview.

    \b
    Examples:
      dive nlp info reviews.csv
      dive nlp info --data comments.jsonl
      dive nlp info data/dataset.parquet
    """
    resolved_data = data_path or data
    if not resolved_data:
        raise click.UsageError("Missing dataset path. Provide DATA_PATH argument or --data option.")

    console = _console(ctx)
    ds = NLPDataset.from_file(resolved_data)
    frame = ds.to_dataframe()

    builder = ReportBuilder("DIVE NLP DATASET INSPECTOR", console=console)
    builder.kv("Source file", resolved_data)
    builder.kv("Rows x columns", f"{len(frame):,} x {len(frame.columns)}")
    builder.kv("Columns", ", ".join(map(str, frame.columns)))

    # Report what the loader actually resolved rather than guessing again. The
    # previous implementation printed `'text' if 'text' in columns else
    # columns[0]`, which could name a column the loader had not chosen - or, for
    # the target, one absent from the frame entirely.
    builder.kv("Text column", ds.text_column or "(resolved positionally)")
    if ds.has_labels:
        unique_labels = sorted({str(label) for label in ds.labels})
        preview = ", ".join(unique_labels[:8])
        if len(unique_labels) > 8:
            preview += ", ..."
        builder.kv(
            "Target column",
            f"{ds.target_column or '(auto-detected)'} "
            f"[{len(unique_labels)} classes: {preview}]",
        )
    else:
        builder.kv("Target column", "none (unsupervised / raw text collection)")

    stats = ds.summary_stats()
    builder.kv(
        "Document lengths",
        f"avg {stats['avg_word_count']} words "
        f"(median {stats['median_word_count']}), avg {stats['avg_char_length']} chars",
    )

    builder.section("PREVIEW SAMPLE RECORDS")
    for position in range(min(3, len(ds.texts))):
        snippet = str(ds.texts[position])[:100].replace("\n", " ")
        label = f"  ->  {ds.labels[position]}" if ds.has_labels else ""
        builder.bullet(f"#{position + 1} {snippet}...{label}")
    if ds.text_column is None:
        builder.note("Text column was resolved positionally; pass -x to name it explicitly.")

    builder.next_steps(
        [
            f'dive nlp profile "{resolved_data}"',
            f'dive nlp train "{resolved_data}" --trials 5 --output champion.pkl',
        ]
    )
    console.report(builder)


# ----------------------------------------------------------------------
# 2. dive nlp profile & dive nlp audit
# ----------------------------------------------------------------------
def _run_profile(
    ctx: click.Context,
    data_path: Optional[str],
    data: Optional[str],
    text_col: Optional[str],
    target_col: Optional[str],
    label_col: Optional[str],
) -> None:
    resolved_data = data_path or data
    if not resolved_data:
        raise click.UsageError("Missing dataset path. Provide DATA_PATH argument or --data option.")
    resolved_target = target_col or label_col
    console = _console(ctx)
    ds = NLPDataset.from_file(resolved_data, text_column=text_col, target_column=resolved_target)
    console.report(NLPProfiler().profile(ds))


@nlp_command.command("profile")
@click.argument("data_path", required=False, default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to input dataset file.")
@click.option("--text-col", "-x", default=None, help="Name of the text feature column.")
@click.option("--target-col", "-y", default=None, help="Name of the target label column.")
@click.option("--label-col", default=None, help="Alias for --target-col.")
@click.pass_context
def profile_cmd(
    ctx: click.Context,
    data_path: Optional[str],
    data: Optional[str],
    text_col: Optional[str],
    target_col: Optional[str],
    label_col: Optional[str],
) -> None:
    """Profile NLP dataset, character/token distributions, and label audits.

    \b
    Examples:
      dive nlp profile dataset.csv
      dive nlp profile reviews.tsv -x review_text -y sentiment
      dive nlp profile --data tickets.jsonl --text-col description --target-col category
    """
    _run_profile(ctx, data_path, data, text_col, target_col, label_col)


@nlp_command.command("audit")
@click.argument("data_path", required=False, default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to input dataset file.")
@click.option("--text-col", "-x", default=None, help="Name of the text feature column.")
@click.option("--target-col", "-y", default=None, help="Name of the target label column.")
@click.option("--label-col", default=None, help="Alias for --target-col.")
@click.pass_context
def audit_cmd(
    ctx: click.Context,
    data_path: Optional[str],
    data: Optional[str],
    text_col: Optional[str],
    target_col: Optional[str],
    label_col: Optional[str],
) -> None:
    """Run NLP health audit on dataset (alias for dive nlp profile).

    \b
    Examples:
      dive nlp audit dataset.csv
      dive nlp audit --data sentiment.csv --text-col text --label-col label
    """
    _run_profile(ctx, data_path, data, text_col, target_col, label_col)


# ----------------------------------------------------------------------
# 3. dive nlp train
# ----------------------------------------------------------------------
@nlp_command.command("train")
@click.argument("data_path", required=False, default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to input dataset file.")
@click.option("--target-col", "-y", default=None, help="Target label column.")
@click.option("--label-col", default=None, help="Alias for target label column.")
@click.option("--text-col", "-x", default=None, help="Text feature column.")
@click.option("--output", "-o", default="nlp_champion.pkl", help="Destination path for trained champion model.")
@click.option("--output-dir", default=None, help="Destination directory or path for trained model.")
@click.option("--trials", "-n", default=5, type=int, help="Maximum number of candidate trials to evaluate.")
@click.option(
    "--optimize-for",
    type=click.Choice(["balanced", "accuracy", "latency"]),
    default="balanced",
    help="Multi-objective optimization criterion (balanced, accuracy, latency).",
)
@click.pass_context
def train_cmd(
    ctx: click.Context,
    data_path: Optional[str],
    data: Optional[str],
    target_col: Optional[str],
    label_col: Optional[str],
    text_col: Optional[str],
    output: str,
    output_dir: Optional[str],
    trials: int,
    optimize_for: str,
) -> None:
    """Autonomously evaluate representations and models, select champion, and serialize predictor.

    \b
    Examples:
      dive nlp train dataset.csv
      dive nlp train --data dataset.csv --trials 10 --optimize-for accuracy --output ./best_model.pkl
      dive nlp train reviews.tsv -x text -y sentiment --trials 5 --output model.pkl
      dive nlp train --data sentiment.csv --text-col text --label-col label --output-dir ./nlp_out
    """
    resolved_data = data_path or data
    if not resolved_data:
        raise click.UsageError("Missing dataset path. Provide DATA_PATH argument or --data option.")

    resolved_target = target_col or label_col

    resolved_output = output
    if output_dir:
        if (
            os.path.isdir(output_dir)
            or output_dir.endswith("/")
            or output_dir.endswith("\\")
            or "." not in os.path.basename(output_dir)
        ):
            os.makedirs(output_dir, exist_ok=True)
            resolved_output = os.path.join(output_dir, "nlp_champion.pkl")
        else:
            parent_dir = os.path.dirname(output_dir)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            resolved_output = output_dir

    console = _console(ctx)
    console.rule("DIVE AutoNLP Autonomous Search")
    engine = AutoNLP(max_trials=trials, optimize_for=optimize_for)
    predictor, leaderboard = engine.fit(
        data=resolved_data,
        target_column=resolved_target,
        text_column=text_col,
    )
    console.report(leaderboard)
    save_nlp_predictor(predictor, resolved_output)
    console.success(f"Champion predictor saved to: {resolved_output}")
    console.print(f"  Next: dive nlp predict {resolved_output} --data <new_rows.csv>")


# ----------------------------------------------------------------------
# 4. dive nlp predict
# ----------------------------------------------------------------------
@nlp_command.command("predict")
@click.argument("model_path", required=False, default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--model", "-m", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to trained model file.")
@click.option("--data", "-d", "data_path", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to input dataset file (CSV, JSON, Parquet).")
@click.option("--text", "-t", "single_text", default=None, help="Single text string to score directly from terminal.")
@click.option("--text-col", "-x", default=None, help="Text column name in data file.")
@click.option("--output", "-o", "output_path", default=None, help="Optional output CSV path to write predictions.")
@click.option("--proba", is_flag=True, default=False, help="Include class probability distributions in output.")
@click.pass_context
def predict_cmd(
    ctx: click.Context,
    model_path: Optional[str],
    model: Optional[str],
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
    resolved_model = model_path or model
    if not resolved_model:
        raise click.UsageError("Missing model path. Provide MODEL_PATH argument or --model option.")

    console = _console(ctx)
    console.rule("DIVE NLP Prediction Engine")

    predictor = load_nlp_predictor(resolved_model)
    model_name = getattr(predictor, "model_name", "NLPPredictor")
    console.kv("Predictor", f"{resolved_model} ({model_name})")

    # 1. Single text prediction
    if single_text is not None:
        prediction = predictor.predict([single_text])[0]
        builder = ReportBuilder(console=console)
        builder.kv("Input text", single_text)
        builder.kv("Predicted label", prediction)
        if proba and predictor.has_proba:
            probabilities = predictor.predict_proba([single_text])[0]
            builder.section("CLASS PROBABILITIES")
            for class_name, probability in zip(
                _class_names(predictor, len(probabilities)), probabilities
            ):
                builder.bar(str(class_name), float(probability), 1.0)
        console.report(builder)
        return

    # 2. Batch dataset prediction
    if data_path is not None:
        ds = NLPDataset.from_file(data_path, text_column=text_col)
        started = time.perf_counter()
        predictions = predictor.predict(ds.texts)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        throughput = len(ds.texts) / max(elapsed_ms / 1000.0, 1e-6)

        frame = ds.to_dataframe()
        frame["predicted_label"] = predictions

        if proba and predictor.has_proba:
            probabilities = predictor.predict_proba(ds.texts)
            for index, class_name in enumerate(
                _class_names(predictor, probabilities.shape[1])
            ):
                frame[f"proba_{class_name}"] = probabilities[:, index]

        console.success(
            f"Scored {len(ds):,} documents in {elapsed_ms:.1f} ms "
            f"({throughput:,.1f} docs/sec)"
        )

        if output_path is not None:
            frame.to_csv(output_path, index=False)
            console.success(f"Saved predictions to: {output_path}")
            return

        builder = ReportBuilder("SAMPLE PREDICTIONS", console=console)
        builder.table(
            ["#", "Text", "Predicted"],
            [
                [
                    position + 1,
                    str(ds.texts[position])[:60].replace("\n", " "),
                    predictions[position],
                ]
                for position in range(min(5, len(frame)))
            ],
        )
        console.report(builder)
        return

    # 3. Interactive prediction prompt
    console.print("")
    console.print("  Interactive NLP prediction. Type text and press Enter to score.")
    console.print("  Type 'exit', 'quit', or 'q' to stop.")
    console.print("")

    while True:
        try:
            line = input("dive-nlp > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("")
            console.print("Exiting interactive mode.")
            break

        if not line or line.lower() in ("exit", "quit", "q"):
            console.print("Exiting interactive mode.")
            break

        prediction = predictor.predict([line])[0]
        confidence = ""
        if predictor.has_proba:
            probabilities = predictor.predict_proba([line])[0]
            if len(probabilities):
                confidence = f"  (confidence: {max(probabilities):.1%})"
        console.print(f"  {console.symbol('arrow')} {prediction}{confidence}")
        console.print("")


# ----------------------------------------------------------------------
# 5. dive nlp serve
# ----------------------------------------------------------------------
@nlp_command.command("serve")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--host", default="127.0.0.1", help="Host interface to bind REST server.")
@click.option("--port", default=8000, type=int, help="Port to listen for requests.")
@click.pass_context
def serve_cmd(ctx: click.Context, model_path: str, host: str, port: int) -> None:
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
    console = _console(ctx)
    console.rule(f"DIVE NLP Model Server - {host}:{port}")
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
@click.pass_context
def monitor_cmd(
    ctx: click.Context,
    ref_path: str,
    curr_path: str,
    text_col: Optional[str],
    oov_threshold: float,
) -> None:
    """Audit production distribution shift, length drift, and vocabulary OOV rate.

    \b
    Examples:
      dive nlp monitor baseline.csv production.csv
      dive nlp monitor train_ref.tsv inference_stream.tsv -x text --oov-threshold 0.10
    """
    console = _console(ctx)
    reference = NLPDataset.from_file(ref_path, text_column=text_col)
    current = NLPDataset.from_file(curr_path, text_column=text_col)

    monitor = NLPDriftMonitor(reference_texts=reference.texts, oov_threshold=oov_threshold)
    console.report(monitor.check_drift(current_texts=current.texts))


# ----------------------------------------------------------------------
# 7. dive nlp benchmark
# ----------------------------------------------------------------------
@nlp_command.command("benchmark")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", "data_path", default=None, type=click.Path(exists=True, dir_okay=False), help="Test dataset path for benchmarking.")
@click.option("--samples", "-n", default=100, type=int, help="Number of test iterations.")
@click.pass_context
def benchmark_cmd(
    ctx: click.Context,
    model_path: str,
    data_path: Optional[str],
    samples: int,
) -> None:
    """Benchmark prediction latency percentiles (p50, p95, p99) and throughput.

    \b
    Examples:
      dive nlp benchmark champion.pkl
      dive nlp benchmark champion.pkl --data test.csv --samples 200
    """
    console = _console(ctx)
    predictor = load_nlp_predictor(model_path)

    if data_path is not None:
        texts = NLPDataset.from_file(data_path).texts[:samples]
    else:
        texts = [
            "This is a standard test sentence designed to benchmark text scoring latency.",
            "Another short sentence for performance measurements across NLP pipeline components.",
            "Fast, reliable and accurate inference latency benchmarking across model representations.",
        ] * (samples // 3 + 1)
        texts = texts[:samples]

    if not texts:
        console.warn("No documents available to benchmark.")
        return

    # Warm up so first-call import/JIT cost does not land in the percentiles.
    predictor.predict(texts[:5])

    latencies: List[float] = []
    for text in texts:
        started = time.perf_counter()
        predictor.predict([text])
        latencies.append((time.perf_counter() - started) * 1000.0)

    average = float(np.mean(latencies))
    builder = ReportBuilder("DIVE NLP LATENCY & THROUGHPUT BENCHMARK", console=console)
    builder.kv("Predictor", model_path)
    builder.kv("Documents scored", f"{len(texts):,}")
    builder.section("LATENCY PROFILE")
    builder.table(
        ["Percentile", "Latency (ms/doc)"],
        [
            ["p50 (median)", f"{float(np.percentile(latencies, 50)):.2f}"],
            ["p95", f"{float(np.percentile(latencies, 95)):.2f}"],
            ["p99", f"{float(np.percentile(latencies, 99)):.2f}"],
            ["mean", f"{average:.2f}"],
        ],
    )
    builder.kv("Single-thread rate", f"{1000.0 / max(average, 1e-6):,.1f} docs/sec")
    console.report(builder)


# ----------------------------------------------------------------------
@nlp_command.command("zero-shot")
@click.argument("text", type=str)
@click.option("--labels", "-l", required=True, type=str, help="Comma-separated candidate labels.")
@click.option("--temperature", "-t", type=float, default=0.15, help="Softmax temperature calibration.")
@click.pass_context
def zero_shot_command(
    ctx: click.Context,
    text: str,
    labels: str,
    temperature: float,
) -> None:
    """Classify raw text into candidate categories without any training data.

    \b
    Examples:
      dive nlp zero-shot "Urgent! Claim your free gift card now" --labels "spam,news,finance"
      dive nlp zero-shot "Quarterly revenue increased by 14%" --labels "earnings,product,hr"
    """
    from dive.nlp.inference.zero_shot import ZeroShotClassifier

    console = _console(ctx)
    candidate_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]

    classifier = ZeroShotClassifier(temperature=temperature)
    results = classifier.predict(text, candidate_list)
    res = results[0]

    builder = ReportBuilder("DIVE NLP ZERO-SHOT CLASSIFIER", console=console)
    builder.kv("Input Text", text)
    builder.kv("Predicted Label", res["predicted_label"])
    builder.kv("Confidence", f"{res['confidence'] * 100:.1f}%")
    builder.section("CALIBRATED CLASS PROBABILITIES")
    rows = []
    for lbl, prob in res["ranking"]:
        bar_len = int(prob * 24)
        bar = "#" * bar_len + "." * (24 - bar_len)
        rows.append([lbl, f"[{bar}]", f"{prob * 100:.1f}%"])
    builder.table(["Candidate Class", "Distribution Bar", "Probability"], rows)
    console.report(builder)

