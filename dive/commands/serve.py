"""CLI Command logic for `dive serve`."""

from __future__ import annotations

from dive.predictor import load_predictor
from dive.serving import serve_model
from dive.utils.logging import Console


def run_serve(console: Console, model_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    console.rule("DIVE Production Model Server")
    console.info(f"Loading predictor artifact from {model_path}...")
    predictor = load_predictor(model_path)

    console.success(f"Loaded predictor model: {predictor.model_name} (Version: {predictor.dive_version})")
    console.info(f"Starting REST server on http://{host}:{port}...")
    serve_model(predictor, host=host, port=port)
