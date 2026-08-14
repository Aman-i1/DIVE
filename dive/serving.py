"""Production Model Server - REST API Endpoints.

Provides FastAPI-based HTTP server endpoints:
- GET /health
- GET /metadata
- GET /schema
- POST /predict
- POST /predict_proba
- GET /metrics
Reuses the exact fitted DivePredictor preprocessing pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from dive.predictor import DivePredictor
from dive.utils.optional import is_available, load_optional


@dataclass
class ServerMetricsTracker:
    """Tracks REST server request volume, latencies, and status metrics."""

    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)

    def log_request(self, latency_ms: float, is_error: bool = False) -> None:
        self.request_count += 1
        if is_error:
            self.error_count += 1
        self.total_latency_ms += latency_ms
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 1000:
            self.latencies_ms = self.latencies_ms[-1000:]

    def get_summary(self) -> Dict[str, Any]:
        l_sorted = sorted(self.latencies_ms) if self.latencies_ms else [0.0]
        n = len(l_sorted)
        p50 = l_sorted[int(n * 0.50)]
        p95 = l_sorted[int(n * 0.95)] if n > 20 else l_sorted[-1]
        p99 = l_sorted[int(n * 0.99)] if n > 100 else l_sorted[-1]

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.total_latency_ms / max(1, self.request_count), 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
        }


def create_serving_app(predictor: DivePredictor) -> Any:
    """Build a FastAPI application exposing predictor endpoints."""
    if not is_available("fastapi"):
        raise ImportError(
            "fastapi is required for model serving. Install with `pip install fastapi uvicorn`."
        )

    fastapi = load_optional("fastapi")
    FastAPI = fastapi.FastAPI

    app = FastAPI(
        title=f"DIVE Production Server - {predictor.model_name}",
        description="Production REST API for tabular predictions powered by DIVE.",
        version="1.0.0",
    )

    metrics_tracker = ServerMetricsTracker()

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model_name": predictor.model_name,
            "problem_type": predictor.problem_type,
            "target": predictor.target,
            "dive_version": predictor.dive_version,
        }

    @app.get("/metadata")
    def get_metadata() -> Dict[str, Any]:
        return {
            "model_name": predictor.model_name,
            "problem_type": predictor.problem_type,
            "target": predictor.target,
            "metrics": predictor.metrics,
            "dataset_name": predictor.dataset_name,
            "trained_at": predictor.trained_at,
            "required_columns": predictor.required_columns,
        }

    @app.get("/schema")
    def get_schema() -> Dict[str, Any]:
        return predictor.input_schema

    @app.post("/predict")
    def predict(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            predictions = predictor.predict(data)
            latency = (time.perf_counter() - start) * 1000.0
            metrics_tracker.log_request(latency, is_error=False)
            
            # Coerce numpy arrays to Python lists
            pred_list = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
            return {
                "predictions": pred_list,
                "model_name": predictor.model_name,
                "latency_ms": round(latency, 2),
            }
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000.0
            metrics_tracker.log_request(latency, is_error=True)
            raise fastapi.HTTPException(status_code=400, detail=str(exc))

    @app.post("/predict_proba")
    def predict_proba(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        if not predictor.has_proba:
            raise fastapi.HTTPException(
                status_code=400, detail="Predictor does not support probability predictions."
            )
        start = time.perf_counter()
        try:
            proba = predictor.predict_proba(data)
            latency = (time.perf_counter() - start) * 1000.0
            metrics_tracker.log_request(latency, is_error=False)

            proba_list = proba.tolist() if hasattr(proba, "tolist") else list(proba)
            return {
                "probabilities": proba_list,
                "class_names": predictor.class_names,
                "model_name": predictor.model_name,
                "latency_ms": round(latency, 2),
            }
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000.0
            metrics_tracker.log_request(latency, is_error=True)
            raise fastapi.HTTPException(status_code=400, detail=str(exc))

    @app.get("/metrics")
    def get_server_metrics() -> Dict[str, Any]:
        return metrics_tracker.get_summary()

    return app


def _serve_builtin_http_server(
    predictor: DivePredictor, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """Zero-dependency HTTP server fallback for REST model serving."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json

    metrics_tracker = ServerMetricsTracker()

    class ModelHTTPHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {
                    "status": "healthy",
                    "model_name": predictor.model_name,
                    "problem_type": predictor.problem_type,
                    "target": predictor.target,
                    "dive_version": predictor.dive_version,
                })
            elif self.path == "/metadata":
                self._send_json(200, {
                    "model_name": predictor.model_name,
                    "problem_type": predictor.problem_type,
                    "target": predictor.target,
                    "metrics": predictor.metrics,
                    "dataset_name": predictor.dataset_name,
                    "trained_at": predictor.trained_at,
                    "required_columns": predictor.required_columns,
                })
            elif self.path == "/schema":
                self._send_json(200, predictor.input_schema)
            elif self.path == "/metrics":
                self._send_json(200, metrics_tracker.get_summary())
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            start = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body) if body else {}
                data = payload.get("data", payload)
            except Exception as exc:
                self._send_json(400, {"error": f"Invalid JSON body: {exc}"})
                return

            if self.path == "/predict":
                try:
                    preds = predictor.predict(data)
                    latency = (time.perf_counter() - start) * 1000.0
                    metrics_tracker.log_request(latency, is_error=False)
                    pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
                    self._send_json(200, {
                        "predictions": pred_list,
                        "model_name": predictor.model_name,
                        "latency_ms": round(latency, 2),
                    })
                except Exception as exc:
                    latency = (time.perf_counter() - start) * 1000.0
                    metrics_tracker.log_request(latency, is_error=True)
                    self._send_json(400, {"error": str(exc)})

            elif self.path == "/predict_proba":
                if not predictor.has_proba:
                    self._send_json(400, {"error": "Predictor does not support probability predictions."})
                    return
                try:
                    proba = predictor.predict_proba(data)
                    latency = (time.perf_counter() - start) * 1000.0
                    metrics_tracker.log_request(latency, is_error=False)
                    proba_list = proba.values.tolist() if hasattr(proba, "values") else (
                        proba.tolist() if hasattr(proba, "tolist") else list(proba)
                    )
                    self._send_json(200, {
                        "probabilities": proba_list,
                        "class_names": predictor.class_names,
                        "model_name": predictor.model_name,
                        "latency_ms": round(latency, 2),
                    })
                except Exception as exc:
                    latency = (time.perf_counter() - start) * 1000.0
                    metrics_tracker.log_request(latency, is_error=True)
                    self._send_json(400, {"error": str(exc)})
            else:
                self._send_json(404, {"error": "Not found"})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer((host, port), ModelHTTPHandler)
    server.serve_forever()


def serve_model(
    predictor: DivePredictor, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """Launch HTTP server serving the predictor (FastAPI+Uvicorn if available, else built-in HTTP server)."""
    if is_available("uvicorn") and is_available("fastapi"):
        uvicorn = load_optional("uvicorn")
        app = create_serving_app(predictor)
        uvicorn.run(app, host=host, port=port)
    else:
        _serve_builtin_http_server(predictor, host=host, port=port)
