"""High-Performance Chunked Batch Inference Engine - `dive/batch_inference.py`.

Processes large datasets (CSV, Parquet) in memory-bounded streaming chunks,
generating predictions, probabilities, and prediction intervals efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class BatchInferenceStats:
    """Statistics summarizing batch inference execution."""

    total_rows: int
    chunk_count: int
    elapsed_seconds: float
    throughput_rows_per_sec: float
    output_file: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "chunk_count": self.chunk_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "throughput_rows_per_sec": round(self.throughput_rows_per_sec, 1),
            "output_file": str(self.output_file) if self.output_file else None,
        }


class BatchInferenceEngine:
    """Streaming chunked batch inference processor for large production datasets."""

    def __init__(self, predictor: Any, chunk_size: int = 10_000) -> None:
        self.predictor = predictor
        self.chunk_size = chunk_size

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        include_probabilities: bool = False,
    ) -> pd.DataFrame:
        """Execute chunked inference in-memory with bounded peak allocation."""
        results = []
        n_rows = len(df)

        for start in range(0, n_rows, self.chunk_size):
            chunk = df.iloc[start : start + self.chunk_size]
            preds = self.predictor.predict(chunk)
            chunk_res = pd.DataFrame({"prediction": preds}, index=chunk.index)

            if include_probabilities and hasattr(self.predictor, "predict_proba"):
                probs = self.predictor.predict_proba(chunk)
                if probs.ndim > 1:
                    for c_idx in range(probs.shape[1]):
                        chunk_res[f"prob_class_{c_idx}"] = probs[:, c_idx]
                else:
                    chunk_res["prob_class_1"] = probs

            results.append(chunk_res)

        return pd.concat(results, axis=0) if results else pd.DataFrame()

    def predict_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_column: Optional[str] = None,
        include_probabilities: bool = False,
    ) -> BatchInferenceStats:
        """Stream input CSV file, generate predictions, and write to output CSV."""
        import time
        start_time = time.time()
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        chunk_count = 0
        first_chunk = True

        for chunk in pd.read_csv(in_path, chunksize=self.chunk_size):
            chunk_count += 1
            total_rows += len(chunk)

            X_chunk = chunk.drop(columns=[target_column]) if target_column and target_column in chunk.columns else chunk
            pred_df = self.predict_dataframe(X_chunk, include_probabilities=include_probabilities)

            combined_chunk = pd.concat([chunk.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

            # Write streaming chunk to output CSV
            if first_chunk:
                combined_chunk.to_csv(out_path, index=False, mode="w")
                first_chunk = False
            else:
                combined_chunk.to_csv(out_path, index=False, mode="a", header=False)

        elapsed = time.time() - start_time
        throughput = total_rows / max(elapsed, 1e-6)

        return BatchInferenceStats(
            total_rows=total_rows,
            chunk_count=chunk_count,
            elapsed_seconds=elapsed,
            throughput_rows_per_sec=throughput,
            output_file=out_path,
        )
