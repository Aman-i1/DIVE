"""Production NLP Model Server - FastAPI & HTTP Endpoints - `dive/nlp/serving/app.py`.

Provides production-ready REST API endpoints for NLP model serving:
- GET  /health
- GET  /nlp/info
- POST /nlp/predict (and /predict)
- POST /nlp/predict_proba (and /predict_proba)
- POST /nlp/batch_predict (and /batch_predict)
- GET  /metrics

Supports single raw text, JSON text collections, and dictionary record formats.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from dive.nlp.inference.predictor import NLPPredictor
from dive.nlp.interfaces import NLPPredictorProtocol
from dive.nlp.optimization.predictor import OptimizedNLPPredictor
from dive.serving import ServerMetricsTracker
from dive.utils.optional import is_available, load_optional


def _parse_nlp_request_payload(
    payload: Any, text_column: str = "text"
) -> List[str]:
    """Parse flexible incoming HTTP payloads into a clean List[str]."""
    if isinstance(payload, str):
        return [payload]

    if isinstance(payload, list):
        if not payload:
            return []
        if isinstance(payload[0], str):
            return [str(s) for s in payload]
        if isinstance(payload[0], dict):
            res = []
            for row in payload:
                if text_column in row:
                    res.append(str(row[text_column]))
                elif len(row) == 1:
                    res.append(str(next(iter(row.values()))))
                else:
                    raise ValueError(f"Record missing expected text column '{text_column}'.")
            return res

    if isinstance(payload, dict):
        if "text" in payload and isinstance(payload["text"], str):
            return [payload["text"]]
        if "texts" in payload and isinstance(payload["texts"], list):
            return [str(s) for s in payload["texts"]]
        if "data" in payload:
            return _parse_nlp_request_payload(payload["data"], text_column=text_column)
        if text_column in payload:
            return [str(payload[text_column])]
        if len(payload) == 1:
            return [str(next(iter(payload.values())))]

    raise ValueError(
        "Invalid NLP payload format. Expected raw string, list of strings, "
        "or JSON object with 'text', 'texts', or 'data' keys."
    )


def create_nlp_serving_app(
    predictor: Union[NLPPredictor, OptimizedNLPPredictor, Any],
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Any:
    """Build a FastAPI application exposing REST endpoints for NLP predictions."""
    if not is_available("fastapi"):
        raise ImportError(
            "fastapi is required for NLP model serving. Install with `pip install fastapi uvicorn`."
        )

    fastapi = load_optional("fastapi")
    FastAPI = fastapi.FastAPI

    app = FastAPI(
        title=f"DIVE NLP Server - {predictor.model_name}",
        description="Production REST API for natural language predictions powered by DIVE NLP.",
        version="1.0.0",
    )

    metrics_tracker = ServerMetricsTracker()
    text_col = getattr(predictor, "text_column", "text")

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "domain": "nlp",
            "model_name": predictor.model_name,
            "has_probabilities": getattr(predictor, "has_proba", False),
        }

    @app.get("/nlp/info")
    @app.get("/info")
    def get_info() -> Dict[str, Any]:
        if hasattr(predictor, "describe_input"):
            return predictor.describe_input()
        return {"model_name": predictor.model_name}

    @app.post("/nlp/predict")
    @app.post("/predict")
    def predict_endpoint(payload: Any = fastapi.Body(...)) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            texts = _parse_nlp_request_payload(payload, text_column=text_col)
            if not texts:
                raise ValueError("Payload contains no text documents to score.")

            preds = predictor.predict(texts)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics_tracker.log_request(latency_ms, is_error=False)

            pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
            return {
                "predictions": pred_list,
                "n_samples": len(pred_list),
                "model_name": predictor.model_name,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics_tracker.log_request(latency_ms, is_error=True)
            raise fastapi.HTTPException(status_code=400, detail=str(exc))

    @app.post("/nlp/predict_proba")
    @app.post("/predict_proba")
    def predict_proba_endpoint(payload: Any = fastapi.Body(...)) -> Dict[str, Any]:
        if not getattr(predictor, "has_proba", False):
            raise fastapi.HTTPException(
                status_code=400,
                detail="Underlying NLP model does not support probability distributions.",
            )

        t0 = time.perf_counter()
        try:
            texts = _parse_nlp_request_payload(payload, text_column=text_col)
            if not texts:
                raise ValueError("Payload contains no text documents to score.")

            probas = predictor.predict_proba(texts)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics_tracker.log_request(latency_ms, is_error=False)

            proba_list = probas.tolist() if hasattr(probas, "tolist") else list(probas)
            return {
                "probabilities": proba_list,
                "class_names": getattr(predictor, "class_names", None),
                "n_samples": len(proba_list),
                "model_name": predictor.model_name,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics_tracker.log_request(latency_ms, is_error=True)
            raise fastapi.HTTPException(status_code=400, detail=str(exc))

    @app.post("/nlp/batch_predict")
    @app.post("/batch_predict")
    def batch_predict_endpoint(payload: Any = fastapi.Body(...)) -> Dict[str, Any]:
        return predict_endpoint(payload=payload)

    @app.get("/metrics")
    def get_metrics() -> Dict[str, Any]:
        summary = metrics_tracker.get_summary()
        if hasattr(predictor, "stats"):
            summary["predictor_stats"] = predictor.stats()
        return summary

    return app


def serve_nlp_model(
    predictor: Union[NLPPredictor, OptimizedNLPPredictor, Any],
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Launch production HTTP server for the NLP predictor."""
    if is_available("uvicorn") and is_available("fastapi"):
        uvicorn = load_optional("uvicorn")
        app = create_nlp_serving_app(predictor=predictor, host=host, port=port)
        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        # Zero-dependency HTTP server fallback
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class NLPHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/health", "/health/"):
                    self._send_json(200, {"status": "healthy", "domain": "nlp", "model": predictor.model_name})
                elif self.path in ("/info", "/nlp/info"):
                    info = predictor.describe_input() if hasattr(predictor, "describe_input") else {}
                    self._send_json(200, info)
                else:
                    self._send_json(404, {"error": "Not Found"})

            def do_POST(self) -> None:
                content_len = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_len)
                try:
                    payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                    texts = _parse_nlp_request_payload(payload)
                    preds = predictor.predict(texts)
                    self._send_json(200, {"predictions": list(preds), "model_name": predictor.model_name})
                except Exception as e:
                    self._send_json(400, {"error": str(e)})

            def _send_json(self, status: int, data: Any) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

        server = HTTPServer((host, port), NLPHandler)
        server.serve_forever()
