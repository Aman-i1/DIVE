"""Resource-Aware AutoML Planning & Memory Safety Management.

Estimates dataset memory footprints, peak training memory overhead, wall-clock runtimes,
and constructs safe execution plans to prevent Out-Of-Memory (OOM) crashes and CPU starvation.
"""

from __future__ import annotations

import os
import psutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class AutoMLResourcePlan:
    """Resource-aware execution plan for model zoo training."""

    n_samples: int
    n_features: int
    estimated_dataset_mb: float
    estimated_peak_ram_mb: float
    memory_limit_mb: float
    time_budget_sec: float
    recommended_workers: int
    included_models: List[str]
    excluded_models: List[str]
    estimated_runtime_sec: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "estimated_dataset_mb": round(self.estimated_dataset_mb, 2),
            "estimated_peak_ram_mb": round(self.estimated_peak_ram_mb, 2),
            "memory_limit_mb": round(self.memory_limit_mb, 2),
            "time_budget_sec": self.time_budget_sec,
            "recommended_workers": self.recommended_workers,
            "included_models": self.included_models,
            "excluded_models": self.excluded_models,
            "estimated_runtime_sec": round(self.estimated_runtime_sec, 2),
            "warnings": self.warnings,
        }

    def render(self) -> str:
        lines = [
            "DIVE RESOURCE-AWARE AUTOML PLAN",
            "===============================",
            f"Dataset Dimensions  : {self.n_samples:,} rows x {self.n_features} columns",
            f"Memory Estimate     : {self.estimated_dataset_mb:.1f} MB (Peak Training: {self.estimated_peak_ram_mb:.1f} MB)",
            f"Memory Limit        : {self.memory_limit_mb:.1f} MB",
            f"Allocated Workers   : {self.recommended_workers} CPU cores",
            f"Time Budget         : {self.time_budget_sec:.0f} seconds",
            f"Included Models     : {', '.join(self.included_models)}",
        ]
        if self.excluded_models:
            lines.append(f"Excluded Models     : {', '.join(self.excluded_models)}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


@dataclass
class SystemResources:
    """Hardware resource snapshot."""

    cpu_count: int
    ram_total_mb: float
    ram_available_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "ram_total_mb": round(self.ram_total_mb, 1),
            "ram_available_mb": round(self.ram_available_mb, 1),
        }


class ResourceManager:
    """Estimates resource consumption and configures safe parallel workers."""

    def __init__(
        self,
        time_budget_sec: float = 1800.0,
        memory_limit_gb: Optional[float] = None,
        n_jobs: Optional[int] = None,
        use_gpu: bool = False,
    ) -> None:
        self.time_budget_sec = float(time_budget_sec)
        
        # Determine available system RAM
        sys_ram_gb = 8.0
        try:
            sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass

        self.memory_limit_mb = (memory_limit_gb * 1024.0) if memory_limit_gb else (sys_ram_gb * 0.8 * 1024.0)

        # Determine CPU count
        cpu_count = os.cpu_count() or 4
        if n_jobs is None or n_jobs < 1:
            self.n_jobs = max(1, cpu_count - 1)
        else:
            self.n_jobs = min(n_jobs, cpu_count)

        self.use_gpu = use_gpu

    def get_system_resources(self) -> SystemResources:
        """Return system hardware resources snapshot."""
        sys_ram_mb = 8192.0
        sys_avail_mb = 4096.0
        try:
            vm = psutil.virtual_memory()
            sys_ram_mb = vm.total / (1024 * 1024)
            sys_avail_mb = vm.available / (1024 * 1024)
        except Exception:
            pass
        return SystemResources(
            cpu_count=self.n_jobs,
            ram_total_mb=sys_ram_mb,
            ram_available_mb=sys_avail_mb,
        )

    def create_plan(
        self,
        df: pd.DataFrame,
        base_model_zoo: List[str],
        mode: str = "balanced",
    ) -> AutoMLResourcePlan:
        """Analyze dataframe and build resource-aware execution plan."""
        n_samples, n_features = df.shape
        raw_bytes = df.memory_usage(deep=True).sum()
        est_ds_mb = raw_bytes / (1024 * 1024)

        # Peak RAM overhead multiplier depends on feature expansion and ensembling
        peak_multiplier = 4.5 if mode == "competition" else (3.5 if mode == "balanced" else 2.5)
        est_peak_mb = est_ds_mb * peak_multiplier

        warnings: List[str] = []
        included: List[str] = []
        excluded: List[str] = []

        # Check if dataset exceeds RAM limit
        if est_peak_mb > self.memory_limit_mb:
            warnings.append(
                f"Estimated peak memory ({est_peak_mb:.1f} MB) exceeds configured limit ({self.memory_limit_mb:.1f} MB)."
            )

        # Model filtering based on size and memory budget
        for model in base_model_zoo:
            if model == "KNN" and n_samples > 50_000:
                excluded.append("KNN (Row count > 50K)")
            elif model in ("MLP", "ExtraTrees") and est_peak_mb > (self.memory_limit_mb * 0.7):
                excluded.append(f"{model} (Memory constraint)")
            else:
                included.append(model)

        # Adjust worker count for large datasets to avoid multiprocessing memory multiplication
        safe_workers = self.n_jobs
        if est_ds_mb > 500.0:
            safe_workers = min(safe_workers, 2)
            warnings.append("Parallel worker count reduced to 2 to prevent RAM exhaustion on large dataset.")

        # Estimate runtime
        base_per_model_sec = (n_samples / 10_000.0) * (n_features / 20.0) * 1.5
        est_runtime_sec = min(self.time_budget_sec, base_per_model_sec * len(included))

        return AutoMLResourcePlan(
            n_samples=n_samples,
            n_features=n_features,
            estimated_dataset_mb=est_ds_mb,
            estimated_peak_ram_mb=est_peak_mb,
            memory_limit_mb=self.memory_limit_mb,
            time_budget_sec=self.time_budget_sec,
            recommended_workers=safe_workers,
            included_models=included,
            excluded_models=excluded,
            estimated_runtime_sec=est_runtime_sec,
            warnings=warnings,
        )
