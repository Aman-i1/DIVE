"""Experiment Tracking, Dataset Versioning, Comparison & Reproducibility Engine.

Manages immutable experiment records stored under `.dive/experiments/`, computes deterministic
dataset fingerprints via streaming hashing, compares experiment runs side-by-side, and generates
reproducibility manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dive import __version__
from dive.utils.io import ensure_dir, write_text, load_json, save_json


def get_git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        import subprocess

        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


class DatasetHasher:
    """Computes deterministic dataset fingerprints without loading entire file into memory."""

    @staticmethod
    def hash_dataframe(df: pd.DataFrame, sample_size: int = 10_000) -> Dict[str, Any]:
        hasher = hashlib.sha256()

        # Schema & column names
        col_str = "|".join(map(str, df.columns))
        dtype_str = "|".join(map(str, df.dtypes))
        hasher.update(col_str.encode("utf-8"))
        hasher.update(dtype_str.encode("utf-8"))

        # Row count & shape
        n_rows, n_cols = df.shape
        hasher.update(f"{n_rows}x{n_cols}".encode("utf-8"))

        # Sample fingerprint
        sample = df.head(min(sample_size, n_rows))
        try:
            sample_bytes = pd.util.hash_pandas_object(sample.astype(str), index=False).to_numpy().tobytes()
            hasher.update(sample_bytes)
        except Exception:
            hasher.update(str(sample.values).encode("utf-8"))

        dataset_hash = hasher.hexdigest()[:16]

        return {
            "dataset_hash": f"DS-{dataset_hash}",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": list(df.columns),
            "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
        }

    @staticmethod
    def hash_file(file_path: Union[str, Path]) -> str:
        """Streaming SHA-256 hash of a file on disk."""
        path = Path(file_path)
        if not path.is_file():
            return "unknown"
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return f"FILE-{hasher.hexdigest()[:16]}"


@dataclass
class ExperimentRecord:
    """Immutable record of a single training run."""

    experiment_id: str
    timestamp: str
    dataset_name: str
    dataset_hash: str
    dataset_shape: Tuple[int, int]
    target: str
    problem_type: str
    validation_strategy: str
    model_name: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    training_time_seconds: float
    peak_memory_mb: float
    random_seed: int
    python_version: str
    dive_version: str
    git_commit: str
    readiness_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "dataset_shape": list(self.dataset_shape),
            "target": self.target,
            "problem_type": self.problem_type,
            "validation_strategy": self.validation_strategy,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "training_time_seconds": round(self.training_time_seconds, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "random_seed": self.random_seed,
            "python_version": self.python_version,
            "dive_version": self.dive_version,
            "git_commit": self.git_commit,
            "readiness_score": self.readiness_score,
        }


class ExperimentTracker:
    """Records and queries experiment artifacts on disk under `.dive/experiments/`."""

    def __init__(self, storage_dir: Union[str, Path] = ".dive/experiments") -> None:
        self.storage_dir = ensure_dir(storage_dir)

    def record_run(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        target: str,
        problem_type: str,
        validation_strategy: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        training_time_sec: float,
        peak_memory_mb: float = 0.0,
        random_seed: int = 42,
        readiness_score: Optional[float] = None,
    ) -> ExperimentRecord:
        """Create and persist an immutable experiment log."""
        fingerprint = DatasetHasher.hash_dataframe(df)

        # Generate sequential / timestamped experiment ID
        count = len(list(self.storage_dir.glob("EXP-*.json")))
        exp_id = f"EXP-{count + 1:06d}"
        timestamp = datetime.now().isoformat()

        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=timestamp,
            dataset_name=dataset_name,
            dataset_hash=fingerprint["dataset_hash"],
            dataset_shape=df.shape,
            target=target,
            problem_type=problem_type,
            validation_strategy=validation_strategy,
            model_name=model_name,
            hyperparameters=hyperparameters,
            metrics=metrics,
            training_time_seconds=training_time_sec,
            peak_memory_mb=peak_memory_mb,
            random_seed=random_seed,
            python_version=sys.version.split()[0],
            dive_version=__version__,
            git_commit=get_git_commit(),
            readiness_score=readiness_score,
        )

        target_file = self.storage_dir / f"{exp_id}.json"
        save_json(target_file, record.to_dict())
        return record

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Return list of all tracked experiments."""
        files = sorted(self.storage_dir.glob("EXP-*.json"))
        experiments = []
        for f in files:
            try:
                data = load_json(f)
                experiments.append(data)
            except Exception:
                pass
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a specific experiment ID."""
        target = self.storage_dir / f"{experiment_id}.json"
        if not target.exists():
            return None
        return load_json(target)

    def compare_experiments(
        self, experiment_ids: List[str]
    ) -> pd.DataFrame:
        """Compare multiple experiments side-by-side."""
        rows = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                row = {
                    "Experiment": exp["experiment_id"],
                    "Model": exp["model_name"],
                    "Dataset Hash": exp["dataset_hash"],
                    "Time (s)": exp["training_time_seconds"],
                    "RAM (MB)": exp["peak_memory_mb"],
                }
                row.update(exp.get("metrics", {}))
                rows.append(row)
        return pd.DataFrame(rows)


class ReproducibilityEngine:
    """Generates reproducibility manifests and configuration bundles."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        self.tracker = tracker

    def export_reproduce_manifest(
        self, experiment_id: str, output_dir: Union[str, Path] = "reproduce_bundle"
    ) -> Path:
        """Export experiment.json, environment.json, dataset_fingerprint.json."""
        exp = self.tracker.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment '{experiment_id}' not found.")

        out_path = ensure_dir(output_dir)

        # 1. experiment.json
        save_json(out_path / "experiment.json", exp)

        # 2. environment.json
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "dive_version": __version__,
            "git_commit": get_git_commit(),
            "cpu_count": os.cpu_count(),
        }
        save_json(out_path / "environment.json", env_info)

        # 3. dataset_fingerprint.json
        fp_info = {
            "dataset_hash": exp.get("dataset_hash"),
            "dataset_shape": exp.get("dataset_shape"),
            "target": exp.get("target"),
        }
        save_json(out_path / "dataset_fingerprint.json", fp_info)

        return out_path
