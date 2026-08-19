"""DIVE NLP Serving Layer - `dive/nlp/serving`.

Provides REST API applications, endpoint routers, and HTTP server launchers for NLP predictors.
"""

from __future__ import annotations

from dive.nlp.serving.app import (
    create_nlp_serving_app,
    serve_nlp_model,
)

__all__ = [
    "create_nlp_serving_app",
    "serve_nlp_model",
]
