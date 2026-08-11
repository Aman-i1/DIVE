"""CLI Command logic for `dive export`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dive.commands.predict import _load_artifact
from dive.onnx_export import ONNXExporter
from dive.predictor import DivePredictor
from dive.utils.logging import Console


def run_export(
    console: Console,
    model_path: str,
    output_path: Optional[str] = None,
    format_type: str = "onnx",
) -> None:
    """Export trained model artifact into ONNX or polyglot format."""
    console.banner("DIVE POLYGLOT MODEL EXPORTER", f"Exporting model artifact: {model_path}")

    artifact = _load_artifact(model_path)
    if isinstance(artifact, DivePredictor):
        predictor = artifact
    else:
        best_est = artifact.best_estimator_
        feat_eng = artifact.feature_engineer_
        feat_cols = artifact.feature_columns_
        target = artifact.target
        prob_type = artifact.problem_type
        label_enc = getattr(artifact, "label_encoder_", None)
        schema = artifact._metadata.get("schema", {})
        predictor = DivePredictor(
            model_name=artifact.best_model_name_,
            estimator=best_est,
            feature_engineer=feat_eng,
            feature_columns=feat_cols,
            label_encoder=label_enc,
            target=target,
            problem_type=prob_type,
            input_schema=schema,
        )

    out_file = Path(output_path or "model.onnx")
    exporter = ONNXExporter(predictor)

    with console.spinner(f"Converting pipeline to ONNX format ({out_file})..."):
        exporter.export(out_file, console=console)

    console.print("")
    console.rule("Export Complete")
    console.success(f"Polyglot deployment artifact ready at {out_file}")
