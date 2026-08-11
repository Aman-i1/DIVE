"""CLI Command logic for `dive benchmark`."""

from __future__ import annotations

from typing import List

from dive.benchmarking import BenchmarkSuite
from dive.utils.logging import Console


def run_benchmark(console: Console, mode: str = "fast") -> None:
    console.rule("DIVE Scalability & Performance Benchmarking")
    console.info("Running benchmarks across synthetic dataset scaling levels...")
    suite = BenchmarkSuite(mode=mode)
    results = suite.run_benchmark([1000, 10_000])

    lines = [f"{'Rows':<10} {'Cols':<6} {'Winner Model':<16} {'Fit (s)':<10} {'Latency (ms)':<14} {'Metric':<10}"]
    lines.append("-" * 72)
    for r in results:
        lines.append(
            f"{r.n_rows:<10} "
            f"{r.n_cols:<6} "
            f"{r.model_name:<16} "
            f"{r.fit_time_sec:<10.2f} "
            f"{r.predict_latency_ms_per_row:<14.4f} "
            f"{r.metric_score:<10.4f}"
        )
    console.print("\n".join(lines))
    console.success("Benchmark completed successfully.")
