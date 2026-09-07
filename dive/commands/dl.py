"""CLI Subcommand Group for DIVE Deep Learning - ``dive/commands/dl.py``.

Provides user-facing terminal commands for Deep Learning across tabular, text,
image, audio, and video modalities:
- `dive dl info <source>`: Inspect dataset schema, modality layout, sample counts, and hardware sizing.
- `dive dl train <source>`: Train a deep learning model with torch or scikit-learn fallback.
- `dive dl predict <model_path>`: Score new samples or run interactive prediction.
- `dive dl benchmark <model_path>`: Measure latency percentiles (p50, p95, p99) and throughput.
- `dive dl doctor <source>`: Audit data files, hardware compatibility, and dependency status.
- `dive dl auto <source>`: Multi-trial architecture exploration and champion model selection.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd

from dive.commands._context import console_from
from dive.dl import (
    AutoDL,
    DLConfig,
    DLDataConfig,
    DLPredictor,
    DLTask,
    DLTrainer,
    DLTrainingConfig,
    Modality,
    assess_dl_workload,
    available_modalities,
    get_adapter,
    load_dl_predictor,
    save_dl_predictor,
)
from dive.dl.core.device import select_device
from dive.resources import ResourceManager
from dive.utils.logging import Console
from dive.utils.optional import is_available
from dive.utils.report import FAIL, INFO, PASS, TOP, WARN, ReportBuilder


def _console(ctx: click.Context) -> Console:
    return console_from(ctx)


def _infer_modality_from_path(path: Path) -> Modality:
    """Infer modality from directory contents or file extension."""
    if path.is_file():
        suffix = path.suffix.lower()
        from dive.dl.modalities.base import AUDIO_SUFFIXES, IMAGE_SUFFIXES, VIDEO_SUFFIXES

        if suffix in VIDEO_SUFFIXES:
            return Modality.VIDEO
        if suffix in IMAGE_SUFFIXES:
            return Modality.IMAGE
        if suffix in AUDIO_SUFFIXES:
            return Modality.AUDIO
        return Modality.TABULAR

    if path.is_dir():
        from dive.dl.modalities.base import AUDIO_SUFFIXES, IMAGE_SUFFIXES, VIDEO_SUFFIXES

        for item in path.rglob("*"):
            if item.is_file():
                s = item.suffix.lower()
                if s in VIDEO_SUFFIXES:
                    return Modality.VIDEO
                if s in IMAGE_SUFFIXES:
                    return Modality.IMAGE
                if s in AUDIO_SUFFIXES:
                    return Modality.AUDIO
    return Modality.TABULAR


@click.group("dl")
def dl_command() -> None:
    """Deep Learning capability domain across tables, text, audio, images, and video.

    \b
    Supported modalities:
      tabular  -- Mixed-type tabular DataFrames (CSV, Parquet, TSV)
      text     -- Raw text documents or text columns
      image    -- Folders of images or manifest tables
      audio    -- Sound recordings, speech, or audio manifests
      video    -- Video clips or video manifest tables

    \b
    Examples:
      dive dl info data.csv
      dive dl train dataset/ --modality image --epochs 10 --output model.pkl
      dive dl predict model.pkl --data test_clips/
      dive dl doctor dataset/ --modality audio
      dive dl benchmark model.pkl
    """
    pass


# ----------------------------------------------------------------------
# 1. dive dl info
# ----------------------------------------------------------------------
@dl_command.command("info")
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--modality",
    "-m",
    default=None,
    type=click.Choice(available_modalities(), case_sensitive=False),
    help="Explicitly declare data modality.",
)
@click.option("--input-col", "-x", default=None, help="Input feature or file path column name.")
@click.option("--target-col", "-y", default=None, help="Target label column name.")
@click.pass_context
def info_cmd(
    ctx: click.Context,
    source: str,
    modality: Optional[str],
    input_col: Optional[str],
    target_col: Optional[str],
) -> None:
    """Inspect dataset schema, modality layout, sample counts, and hardware requirements."""
    console = _console(ctx)
    path = Path(source)

    modality_enum = Modality.from_str(modality) if modality else _infer_modality_from_path(path)
    data_config = DLDataConfig(input_column=input_col, target_column=target_col)
    adapter = get_adapter(modality_enum)

    batch = adapter.load(source, data_config)

    builder = ReportBuilder("DIVE DL DATASET INSPECTOR", console=console)
    builder.kv("Source", source)
    builder.kv("Modality", modality_enum.value)
    builder.kv("Total samples", f"{batch.n_samples:,}")
    if batch.input_column:
        builder.kv("Input column", batch.input_column)
    if batch.target_column:
        builder.kv("Target column", batch.target_column)

    if batch.has_targets:
        unique_targets = sorted({str(t) for t in batch.targets})
        preview = ", ".join(unique_targets[:6])
        if len(unique_targets) > 6:
            preview += ", ..."
        builder.kv("Classes", f"{len(unique_targets)} ({preview})")
    else:
        builder.kv("Labels", "none (unlabelled collection)")

    # Host sizing
    verdict = assess_dl_workload(modality_enum, batch.n_samples)
    builder.section("HARDWARE SIZING")
    builder.status(verdict.status_token, f"Host capability: {verdict.decision}")
    builder.kv("Projected RAM", f"{verdict.required_mb:,.0f} MB")
    builder.kv("Available RAM", f"{verdict.available_mb:,.0f} MB")

    if batch.notes:
        builder.section("NOTES")
        for note in batch.notes:
            builder.note(note)

    builder.next_steps(
        [
            f'dive dl doctor "{source}" --modality {modality_enum.value}',
            f'dive dl train "{source}" --modality {modality_enum.value} --output model.pkl',
        ]
    )
    console.report(builder)


# ----------------------------------------------------------------------
# 2. dive dl train
# ----------------------------------------------------------------------
@dl_command.command("train")
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--modality",
    "-m",
    default=None,
    type=click.Choice(available_modalities(), case_sensitive=False),
    help="Data modality.",
)
@click.option(
    "--task",
    "-t",
    default="classification",
    type=click.Choice(["classification", "regression"], case_sensitive=False),
    help="Prediction task type.",
)
@click.option("--input-col", "-x", default=None, help="Input column name.")
@click.option("--target-col", "-y", default=None, help="Target column name.")
@click.option(
    "--mode",
    default="balanced",
    type=click.Choice(["fast", "balanced", "quality"]),
    help="Training preset budget.",
)
@click.option("--epochs", "-e", default=None, type=int, help="Number of training epochs.")
@click.option("--lr", default=1e-3, type=float, help="Learning rate.")
@click.option("--batch-size", "-b", default=32, type=int, help="Batch size.")
@click.option("--device", default="auto", type=click.Choice(["auto", "cpu", "cuda", "mps"]), help="Execution device.")
@click.option("--no-torch", is_flag=True, default=False, help="Force Scikit-Learn MLP fallback.")
@click.option("--output", "-o", default="dl_model.pkl", help="Output path for saved predictor.")
@click.pass_context
def train_cmd(
    ctx: click.Context,
    source: str,
    modality: Optional[str],
    task: str,
    input_col: Optional[str],
    target_col: Optional[str],
    mode: str,
    epochs: Optional[int],
    lr: float,
    batch_size: int,
    device: str,
    no_torch: bool,
    output: str,
) -> None:
    """Train a deep learning model for the given modality and task."""
    console = _console(ctx)
    path = Path(source)
    modality_enum = Modality.from_str(modality) if modality else _infer_modality_from_path(path)
    task_enum = DLTask.from_str(task)

    config = DLConfig(
        modality=modality_enum.value,
        task=task_enum.value,
        mode=mode,
    )
    config.apply_mode()
    if epochs is not None:
        config.training.epochs = epochs
    config.training.learning_rate = lr
    config.resources.batch_size = batch_size
    config.resources.device = device

    console.rule(f"DIVE Deep Learning Training [{modality_enum.value.upper()}]")
    console.kv("Modality", modality_enum.value)
    console.kv("Task", task_enum.value)
    console.kv("Mode", mode)

    # 1. Load Data
    adapter = get_adapter(modality_enum, options=config.data.options)
    config.data.input_column = input_col
    config.data.target_column = target_col
    batch = adapter.load(source, config.data)

    if not batch.has_targets:
        console.error("No targets found in dataset. Supervised training requires labels.")
        ctx.exit(1)

    console.kv("Samples", f"{batch.n_samples:,}")

    # 2. Featurize
    console.info(f"Extracting features via {adapter.__class__.__name__}...")
    t0 = time.time()
    X = adapter.fit_transform(batch.inputs)
    y = np.asarray(batch.targets)

    if adapter.failed_indices:
        keep = [i for i in range(len(X)) if i not in adapter.failed_indices]
        if keep:
            X = X[keep]
            y = y[keep]

    console.info(f"Feature extraction finished in {time.time() - t0:.2f}s (features: {X.shape[1]}).")

    # 3. Train Model
    device_info = select_device(device)
    trainer = DLTrainer(
        task=task_enum,
        config=config.training,
        device=device_info,
        batch_size=batch_size,
        force_fallback=no_torch,
    )
    console.info(f"Fitting model using backend: {trainer.backend}...")
    trainer.fit(X, y)

    # 4. Save Predictor
    predictor = DLPredictor(
        adapter=adapter,
        trainer=trainer,
        config=config,
        modality=modality_enum,
        task=task_enum,
        input_column=batch.input_column,
        target_column=batch.target_column,
        metrics={"n_samples": len(X), "n_features": X.shape[1]},
    )
    save_dl_predictor(predictor, output)
    console.success(f"DLPredictor saved successfully to: {output}")
    console.print(f"  Next: dive dl predict {output} --data <new_data>")


# ----------------------------------------------------------------------
# 3. dive dl predict
# ----------------------------------------------------------------------
@dl_command.command("predict")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", "data_path", default=None, help="Path to input data file or directory.")
@click.option("--input", "-i", "single_input", default=None, help="Single raw input string or file path.")
@click.option("--output", "-o", "output_path", default=None, help="Optional output CSV path for predictions.")
@click.option("--proba", is_flag=True, default=False, help="Include class probability distribution.")
@click.pass_context
def predict_cmd(
    ctx: click.Context,
    model_path: str,
    data_path: Optional[str],
    single_input: Optional[str],
    output_path: Optional[str],
    proba: bool,
) -> None:
    """Score inputs or batch datasets using a trained DLPredictor."""
    console = _console(ctx)
    console.rule("DIVE DL Prediction Engine")

    predictor = load_dl_predictor(model_path)
    console.kv("Model artifact", model_path)
    console.kv("Modality", predictor.modality)
    console.kv("Backend", predictor.backend)

    classes_list = list(predictor.classes_) if predictor.classes_ is not None else []
    if single_input is not None:
        prediction = predictor.predict([single_input])[0]
        builder = ReportBuilder(console=console)
        builder.kv("Input", single_input)
        builder.kv("Predicted", str(prediction))
        if proba and predictor.has_proba:
            probabilities = predictor.predict_proba([single_input])[0]
            builder.section("CLASS PROBABILITIES")
            for cls_name, p in zip(classes_list, probabilities):
                builder.bar(str(cls_name), float(p), suffix=f"{p:.1%}")
        console.report(builder)
        return

    if data_path is not None:
        batch = predictor.adapter.load(data_path, predictor.config.data)
        preds = predictor.predict(batch.inputs)
        df_out = pd.DataFrame({"id": batch.ids, "prediction": preds})

        if proba and predictor.has_proba:
            probs = predictor.predict_proba(batch.inputs)
            for i, cls in enumerate(classes_list):
                df_out[f"prob_{cls}"] = probs[:, i]

        if output_path:
            df_out.to_csv(output_path, index=False)
            console.success(f"Predictions written to: {output_path}")
        else:
            builder = ReportBuilder("BATCH PREDICTION SUMMARY", console=console)
            builder.kv("Total scored", f"{len(df_out):,}")
            builder.dataframe(df_out.head(10))
            console.report(builder)
        return

    console.warn("Specify --input <value> for a single item or --data <path> for batch scoring.")


# ----------------------------------------------------------------------
# 4. dive dl benchmark
# ----------------------------------------------------------------------
@dl_command.command("benchmark")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--samples", "-n", default=20, type=int, help="Number of benchmark iterations.")
@click.pass_context
def benchmark_cmd(
    ctx: click.Context,
    model_path: str,
    samples: int,
) -> None:
    """Benchmark prediction latency percentiles and throughput."""
    console = _console(ctx)
    predictor = load_dl_predictor(model_path)

    # Generate synthetic input matching adapter requirements
    n_feat = predictor.trainer.n_features_ or 16
    mock_matrix = np.random.randn(samples, n_feat).astype(np.float32)

    # Warmup
    _ = predictor.trainer.predict(mock_matrix[:2])

    latencies = []
    for i in range(samples):
        row = mock_matrix[i : i + 1]
        t0 = time.perf_counter()
        _ = predictor.trainer.predict(row)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    builder = ReportBuilder("DIVE DL LATENCY & THROUGHPUT BENCHMARK", console=console)
    builder.kv("Predictor", model_path)
    builder.kv("Modality", predictor.modality)
    builder.kv("Backend", predictor.backend)
    builder.kv("Benchmark samples", f"{samples:,}")
    builder.section("LATENCY PROFILE")
    builder.table(
        ["Percentile", "Latency (ms/sample)"],
        [
            ["p50 (median)", f"{float(np.percentile(latencies, 50)):.2f}"],
            ["p95", f"{float(np.percentile(latencies, 95)):.2f}"],
            ["p99", f"{float(np.percentile(latencies, 99)):.2f}"],
            ["mean", f"{avg_ms:.2f}"],
        ],
    )
    builder.kv("Throughput", f"{1000.0 / max(avg_ms, 1e-6):,.1f} samples/sec")
    console.report(builder)


# ----------------------------------------------------------------------
# 5. dive dl doctor
# ----------------------------------------------------------------------
@dl_command.command("doctor")
@click.argument("source", type=click.Path(exists=True), required=False)
@click.option(
    "--modality",
    "-m",
    default=None,
    type=click.Choice(available_modalities(), case_sensitive=False),
    help="Modality to audit.",
)
@click.pass_context
def doctor_cmd(
    ctx: click.Context,
    source: Optional[str],
    modality: Optional[str],
) -> None:
    """Pre-flight audit of hardware limits, dependencies, and modality data readiness."""
    console = _console(ctx)
    builder = ReportBuilder("DIVE DL PRE-FLIGHT DOCTOR AUDIT", console=console)

    # 1. Environment & Hardware
    builder.section("HARDWARE & BACKENDS")
    torch_ok = is_available("torch")
    builder.status(PASS if torch_ok else WARN, f"PyTorch Backend: {'AVAILABLE' if torch_ok else 'MISSING (using sklearn MLP)'}")

    gpu_info = select_device("auto")
    builder.kv("Target Device", f"{gpu_info.name.upper()} ({gpu_info.gpu_name or 'CPU'})")
    builder.note(f"Device resolution: {gpu_info.reason}")

    res_mgr = ResourceManager()
    sys_res = res_mgr.get_system_resources()
    builder.kv("RAM total / available", f"{sys_res.ram_total_mb:,.0f} MB / {sys_res.ram_available_mb:,.0f} MB")

    # 2. Modality packages
    builder.section("MODALITY DEPENDENCIES")
    deps = [
        ("torchvision", "Vision backbones"),
        ("pillow", "Image loading"),
        ("librosa", "Audio log-mel spectrograms"),
        ("soundfile", "Audio WAV/FLAC decoding"),
        ("av", "Video PyAV container decoding"),
        ("opencv-python", "Video / Vision decoding"),
    ]
    for pkg, purpose in deps:
        ok = is_available(pkg)
        builder.status(PASS if ok else INFO, f"{pkg:<16} : {'AVAILABLE' if ok else 'OPTIONAL'} ({purpose})")

    # 3. If source passed, inspect source
    if source:
        path = Path(source)
        mod_enum = Modality.from_str(modality) if modality else _infer_modality_from_path(path)
        builder.section(f"DATA READINESS: {mod_enum.value.upper()}")
        try:
            adapter = get_adapter(mod_enum)
            batch = adapter.load(source, DLDataConfig())
            builder.status(PASS, f"Data loaded successfully: {batch.n_samples:,} samples")
            verdict = assess_dl_workload(mod_enum, batch.n_samples)
            builder.status(verdict.status_token, f"Host capability verdict: {verdict.decision}")
        except Exception as exc:
            builder.status(FAIL, f"Data validation error: {exc}")

    console.report(builder)


# ----------------------------------------------------------------------
# 6. dive dl auto
# ----------------------------------------------------------------------
@dl_command.command("auto")
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--modality",
    "-m",
    default=None,
    type=click.Choice(available_modalities(), case_sensitive=False),
    help="Data modality.",
)
@click.option(
    "--task",
    "-t",
    default="classification",
    type=click.Choice(["classification", "regression"], case_sensitive=False),
    help="Prediction task type.",
)
@click.option("--target-col", "-y", default=None, help="Target column name.")
@click.option("--input-col", "-x", default=None, help="Input column name.")
@click.option("--trials", "-n", default=3, type=int, help="Number of exploration trials.")
@click.option("--mode", default="balanced", type=click.Choice(["fast", "balanced", "quality"]), help="Search preset.")
@click.option("--output", "-o", default="dl_champion.pkl", help="Champion predictor save path.")
@click.pass_context
def auto_cmd(
    ctx: click.Context,
    source: str,
    modality: Optional[str],
    task: str,
    target_col: Optional[str],
    input_col: Optional[str],
    trials: int,
    mode: str,
    output: str,
) -> None:
    """Run autonomous AutoDL multi-trial exploration and save champion predictor."""
    console = _console(ctx)
    path = Path(source)
    mod_enum = Modality.from_str(modality) if modality else _infer_modality_from_path(path)

    console.rule(f"AutoDL Architecture Search - {mod_enum.value.upper()}")
    autodl = AutoDL(max_trials=trials, mode=mode)
    champion, leaderboard = autodl.fit(
        data=source,
        modality=mod_enum,
        task=task,
        target_column=target_col,
        input_column=input_col,
    )

    console.report(leaderboard)
    save_dl_predictor(champion, output)
    console.success(f"Champion DL predictor saved to: {output}")
    console.print(f"  Next: dive dl predict {output} --data <new_samples>")
