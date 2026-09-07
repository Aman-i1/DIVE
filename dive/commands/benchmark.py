"""CLI Command logic for `dive benchmark`."""

from __future__ import annotations

from dive.benchmarking import BenchmarkSuite
from dive.utils.logging import Console
from dive.utils.report import ReportBuilder


def run_benchmark(console: Console, mode: str = "fast") -> None:
    console.rule("DIVE Scalability & Performance Benchmarking")
    console.info("Running benchmarks across synthetic dataset scaling levels...")
    suite = BenchmarkSuite(mode=mode)
    results = suite.run_benchmark([1000, 10_000])

    builder = ReportBuilder(console=console)
    builder.table(
        ["Rows", "Cols", "Winner Model", "Fit (s)", "Latency (ms)", "Metric"],
        [
            [
                f"{result.n_rows:,}",
                result.n_cols,
                result.model_name,
                f"{result.fit_time_sec:.2f}",
                f"{result.predict_latency_ms_per_row:.4f}",
                f"{result.metric_score:.4f}",
            ]
            for result in results
        ],
    )
    console.report(builder)
    console.success("Benchmark completed successfully.")
