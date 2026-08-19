"""CLI Subcommand Group for DIVE ML - `dive/commands/ml.py`.

Provides domain-specific terminal commands for tabular & structured machine learning:
- `dive ml train`: Train a model zoo on tabular data.
- `dive ml auto`: Fully automated tabular AutoML search.
- `dive ml autopilot`: Autonomous multi-phase data auditing, modeling, and guardrails.
- `dive ml review`: Senior ML reliability and risk review.
- `dive ml contract`: Validate dataset contracts and schemas.
- `dive ml doctor`: Comprehensive tabular data readiness audit.
- `dive ml predict`: Score new rows with a saved tabular model.
- `dive ml validate`: Pre-flight dataset crosscheck suite.
- `dive ml explain`: Compute SHAP & feature importances.
- `dive ml report`: Generate interactive HTML audit reports.
- `dive ml serve`: Launch production REST model server.
- `dive ml drift`: Audit tabular feature and prediction drift.
- `dive ml gate`: Production deployment gatekeeper.
- `dive ml audit`: Full forensic audit of tabular pipeline.
- `dive ml export`: Export pipeline to standalone python code or ONNX.
- `dive ml benchmark`: Benchmark model latency and throughput.
- `dive ml reproduce`: Reproduce run from training manifest.
- `dive ml info`: Display model metadata and input schema.
"""

from __future__ import annotations

import click


@click.group("ml")
def ml_command() -> None:
    """Tabular & Structured Data Machine Learning (DIVE ML) subcommands.

    \b
    Examples:
      dive ml doctor data.csv --target churn
      dive ml auto data.csv --target churn
      dive ml train data.csv --target revenue --mode fast
      dive ml predict model.pkl --data new.csv --output preds.csv
      dive ml serve model.pkl --port 8000
    """
    pass
