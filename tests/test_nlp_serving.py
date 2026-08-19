"""Phase 11 Tests - DIVE NLP Production REST Model Serving.

Verifies:
1. Payload parser (_parse_nlp_request_payload) on raw strings, string arrays, dict records, and nested formats.
2. Error validation on malformed, missing, or empty input bodies.
3. ServerMetricsTracker latency and percentiles calculation.
4. FastAPI endpoint routing via create_nlp_serving_app() when available.
5. Root dive.serving.create_serving_app() delegation to NLP serving application.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from dive.nlp import (
    create_nlp_serving_app,
    train_baseline,
)
from dive.nlp.serving.app import _parse_nlp_request_payload
from dive.serving import ServerMetricsTracker, create_serving_app
from dive.utils.optional import is_available


@pytest.fixture
def trained_nlp_predictor() -> Any:
    """Fixture providing a fitted NLP baseline predictor."""
    df = pd.DataFrame(
        {
            "text": [
                "Exceptional build quality and super fast shipping, loved it!",
                "Amazing performance, totally exceeded my expectations, great product!",
                "Delightful customer support, great onboarding and seamless setup.",
                "Awesome durability, loved everything about this device, top quality.",
                "Worst purchase ever made, broken on arrival, terrible item.",
                "Terrible customer service, completely unhelpful and rude, awful.",
                "Defective unit, useless customer support, refund was refused, terrible.",
                "Awful product, broke within one hour, completely useless and worst.",
            ],
            "label": [
                "positive",
                "positive",
                "positive",
                "positive",
                "negative",
                "negative",
                "negative",
                "negative",
            ],
        }
    )
    predictor, _ = train_baseline(
        data=df,
        text_column="text",
        target_column="label",
        model_name="LogisticRegression",
    )
    return predictor


def test_parse_nlp_request_payload_variations() -> None:
    """Verify parser handles all standard JSON payload structures."""
    # 1. Single string
    assert _parse_nlp_request_payload("Simple test") == ["Simple test"]

    # 2. List of strings
    assert _parse_nlp_request_payload(["Item 1", "Item 2"]) == ["Item 1", "Item 2"]

    # 3. Dict with 'text'
    assert _parse_nlp_request_payload({"text": "Hello world"}) == ["Hello world"]

    # 4. Dict with 'texts'
    assert _parse_nlp_request_payload({"texts": ["A", "B"]}) == ["A", "B"]

    # 5. Dict with 'data' array
    assert _parse_nlp_request_payload({"data": ["Doc 1", "Doc 2"]}) == ["Doc 1", "Doc 2"]

    # 6. List of dict records
    assert _parse_nlp_request_payload([{"text": "Rec 1"}, {"text": "Rec 2"}]) == ["Rec 1", "Rec 2"]

    # 7. Invalid format raises ValueError
    with pytest.raises(ValueError, match="Invalid NLP payload"):
        _parse_nlp_request_payload(12345)


def test_server_metrics_tracker_computations() -> None:
    """Verify ServerMetricsTracker logging and latency percentiles."""
    tracker = ServerMetricsTracker()

    for lat in [10.0, 20.0, 30.0, 40.0, 50.0]:
        tracker.log_request(lat, is_error=False)
    tracker.log_request(100.0, is_error=True)

    summary = tracker.get_summary()
    assert summary["request_count"] == 6
    assert summary["error_count"] == 1
    assert summary["avg_latency_ms"] > 0.0
    assert summary["p50_latency_ms"] > 0.0


def test_nlp_serving_endpoints(trained_nlp_predictor: Any) -> None:
    """Verify all REST API endpoints using FastAPI TestClient when available."""
    if not is_available("fastapi"):
        pytest.skip("FastAPI not installed in current environment.")

    from fastapi.testclient import TestClient

    app = create_nlp_serving_app(predictor=trained_nlp_predictor)
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["domain"] == "nlp"

    # 2. Info / Schema
    res_info = client.get("/nlp/info")
    assert res_info.status_code == 200
    data_info = res_info.json()
    assert data_info["text_column"] == "text"
    assert data_info["has_probabilities"] is True

    # 3. Predict with single string dict
    res_pred1 = client.post("/nlp/predict", json={"text": "Exceptional build quality!"})
    assert res_pred1.status_code == 200
    data_pred1 = res_pred1.json()
    assert len(data_pred1["predictions"]) == 1
    assert data_pred1["predictions"][0] == "positive"

    # 4. Predict with list of strings
    res_pred2 = client.post(
        "/nlp/predict",
        json={"texts": ["Exceptional quality!", "Broken and terrible."]},
    )
    assert res_pred2.status_code == 200
    data_pred2 = res_pred2.json()
    assert len(data_pred2["predictions"]) == 2
    assert data_pred2["predictions"][0] == "positive"
    assert data_pred2["predictions"][1] == "negative"

    # 5. Predict proba
    res_proba = client.post(
        "/nlp/predict_proba",
        json={"texts": ["Exceptional quality!"]},
    )
    assert res_proba.status_code == 200
    data_proba = res_proba.json()
    assert len(data_proba["probabilities"]) == 1
    assert len(data_proba["probabilities"][0]) == 2
    assert set(data_proba["class_names"]) == {"positive", "negative"}

    # 6. Batch predict alias
    res_batch = client.post(
        "/batch_predict",
        json={"texts": ["Great", "Awful"]},
    )
    assert res_batch.status_code == 200
    assert len(res_batch.json()["predictions"]) == 2

    # 7. Metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    data_metrics = res_metrics.json()
    assert data_metrics["request_count"] >= 4

    # 8. Error handling
    res_err = client.post("/nlp/predict", json={})
    assert res_err.status_code == 400


def test_unified_dive_serving_delegation(trained_nlp_predictor: Any) -> None:
    """Verify root dive.serving.create_serving_app() automatically routes NLP predictors."""
    if not is_available("fastapi"):
        pytest.skip("FastAPI not installed in current environment.")

    from fastapi.testclient import TestClient

    # Pass NLP predictor to standard dive.serving.create_serving_app
    app = create_serving_app(predictor=trained_nlp_predictor)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["domain"] == "nlp"
