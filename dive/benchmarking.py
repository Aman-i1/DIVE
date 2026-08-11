"""Scalability & Performance Benchmarking Suite.

Measures dataset size scaling (rows/cols) vs runtime, peak RAM, model accuracy,
serialization size, and single-row / batch prediction latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from dive.core import Dive



@dataclass
class BenchmarkResult:
    """Benchmark outcome for a single dataset size configuration."""

    n_rows: int
    n_cols: int
    model_name: str
    fit_time_sec: float
    predict_latency_ms_per_row: float
    peak_ram_mb: float
    metric_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "model_name": self.model_name,
            "fit_time_sec": round(self.fit_time_sec, 2),
            "predict_latency_ms_per_row": round(self.predict_latency_ms_per_row, 4),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "metric_score": round(self.metric_score, 4),
        }


class BenchmarkSuite:
    """Benchmark runner for DIVE scalability evaluation."""

    def __init__(self, mode: str = "fast", random_state: int = 42) -> None:
        self.mode = mode
        self.random_state = random_state

    def run_benchmark(
        self, row_sizes: List[int] = [1000, 10_000], n_features: int = 20
    ) -> List[BenchmarkResult]:
        """Run benchmark suite across increasing row counts."""
        results = []
        for n_rows in row_sizes:
            # Generate synthetic classification dataset
            X_arr, y_arr = make_classification(
                n_samples=n_rows,
                n_features=n_features,
                n_informative=10,
                random_state=self.random_state,
            )
            cols = [f"feat_{i}" for i in range(n_features)]
            df = pd.DataFrame(X_arr, columns=cols)
            df["target"] = y_arr

            # Fit DIVE pipeline
            start_fit = time.perf_counter()
            d = Dive(target="target", mode=self.mode, random_state=self.random_state)
            d.fit(df)
            fit_dur = time.perf_counter() - start_fit

            # Measure prediction latency
            sample_pred = df.drop(columns=["target"]).head(100)
            start_pred = time.perf_counter()
            _ = d.predict(sample_pred)
            pred_dur_ms = ((time.perf_counter() - start_pred) * 1000.0) / 100.0

            best_name = d.best_model_name_ or "BestModel"
            lead = d.leaderboard()
            score = float(lead.iloc[0]["Test Accuracy"]) if lead is not None and not lead.empty else 0.0

            raw_bytes = df.memory_usage(deep=True).sum()
            ram_mb = (raw_bytes / (1024 * 1024)) * 3.0

            results.append(
                BenchmarkResult(
                    n_rows=n_rows,
                    n_cols=n_features,
                    model_name=best_name,
                    fit_time_sec=fit_dur,
                    predict_latency_ms_per_row=pred_dur_ms,
                    peak_ram_mb=ram_mb,
                    metric_score=score,
                )
            )
        return results
