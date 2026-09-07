"""Resource-Aware AutoML Planning & Memory Safety Management.

Estimates dataset memory footprints, peak training memory overhead, wall-clock runtimes,
and constructs safe execution plans to prevent Out-Of-Memory (OOM) crashes and CPU starvation.

Two properties matter here beyond estimation:

* ``psutil`` is an *optional* dependency (the ``ops`` extra), so it is loaded
  through :mod:`dive.utils.optional`. Importing it at module scope made
  ``import dive`` fail outright on a core-only install, because
  ``dive/__init__.py`` imports :class:`ResourceManager` eagerly.
* A projection that exceeds the host's real headroom produces a
  :class:`CapabilityVerdict` that can **refuse** the run. Warning and proceeding
  anyway is what lets a laptop thrash or die mid-training.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from dive.exceptions import ResourceError
from dive.utils.optional import detect_gpu, load_optional
from dive.utils.report import FAIL, PASS, WARN, ReportBuilder

# Conservative assumptions used when psutil is unavailable. Deliberately modest:
# under-promising headroom degrades the plan, over-promising it crashes the host.
_FALLBACK_RAM_MB = 8192.0
_FALLBACK_AVAILABLE_MB = 4096.0

# Verdict outcomes.
ALLOW = "ALLOW"
DEGRADE = "DEGRADE"
REFUSE = "REFUSE"


def _virtual_memory() -> Optional[Any]:
    """Return psutil's memory snapshot, or ``None`` when it cannot be read."""
    psutil = load_optional("psutil")
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory()
    except Exception:
        return None


def _free_disk_mb(path: str = ".") -> float:
    """Return free disk space in MB, or ``0.0`` when it cannot be determined."""
    try:
        return shutil.disk_usage(path).free / (1024 * 1024)
    except Exception:
        return 0.0


@dataclass
class CapabilityVerdict:
    """Go / degrade / no-go decision for a projected workload.

    ``REFUSE`` is a real outcome, not a warning: :meth:`raise_if_refused` turns it
    into a :class:`~dive.exceptions.ResourceError` before any memory is committed.
    """

    decision: str
    required_mb: float
    available_mb: float
    reasons: List[str] = field(default_factory=list)
    remedies: List[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.decision != REFUSE

    @property
    def headroom_mb(self) -> float:
        return self.available_mb - self.required_mb

    @property
    def status_token(self) -> str:
        return {ALLOW: PASS, DEGRADE: WARN, REFUSE: FAIL}.get(self.decision, WARN)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "required_mb": round(self.required_mb, 2),
            "available_mb": round(self.available_mb, 2),
            "headroom_mb": round(self.headroom_mb, 2),
            "reasons": self.reasons,
            "remedies": self.remedies,
        }

    def raise_if_refused(self) -> None:
        """Abort the run when the host cannot take it."""
        if self.allowed:
            return
        reason = self.reasons[0] if self.reasons else "insufficient memory headroom"
        hint = "\n".join(f"  - {remedy}" for remedy in self.remedies)
        raise ResourceError(
            f"Refusing to run: {reason}",
            ("Try one of:\n" + hint) if hint else "",
        )

    def render(self) -> str:
        builder = ReportBuilder("HOST CAPABILITY VERDICT")
        builder.kv("Decision", self.decision)
        builder.kv("Projected peak", f"{self.required_mb:,.0f} MB")
        builder.kv("Usable headroom", f"{self.available_mb:,.0f} MB")
        builder.status(self.status_token, f"Headroom after run: {self.headroom_mb:,.0f} MB")
        if self.reasons:
            builder.section("REASONS")
            builder.bullets(self.reasons)
        if self.remedies:
            builder.next_steps(self.remedies)
        return builder.build()


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
    verdict: Optional[CapabilityVerdict] = None

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
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }

    def render(self) -> str:
        builder = ReportBuilder("DIVE RESOURCE-AWARE AUTOML PLAN")
        builder.kv("Dataset Dimensions", f"{self.n_samples:,} rows x {self.n_features} columns")
        builder.kv(
            "Memory Estimate",
            f"{self.estimated_dataset_mb:.1f} MB "
            f"(Peak Training: {self.estimated_peak_ram_mb:.1f} MB)",
        )
        builder.kv("Memory Limit", f"{self.memory_limit_mb:.1f} MB")
        builder.kv("Allocated Workers", f"{self.recommended_workers} CPU cores")
        builder.kv("Time Budget", f"{self.time_budget_sec:.0f} seconds")
        builder.kv("Included Models", ", ".join(self.included_models) or "none")
        if self.excluded_models:
            builder.kv("Excluded Models", ", ".join(self.excluded_models))
        if self.verdict is not None:
            builder.status(self.verdict.status_token, f"Host capability: {self.verdict.decision}")
        if self.warnings:
            builder.section("WARNINGS")
            for warning in self.warnings:
                builder.status(WARN, warning)
        return builder.build()


@dataclass
class SystemResources:
    """Hardware resource snapshot."""

    cpu_count: int
    ram_total_mb: float
    ram_available_mb: float
    n_jobs: int = 1
    has_gpu: bool = False
    disk_free_mb: float = 0.0
    detection_source: str = "psutil"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "ram_total_mb": round(self.ram_total_mb, 1),
            "ram_available_mb": round(self.ram_available_mb, 1),
            "n_jobs": self.n_jobs,
            "has_gpu": self.has_gpu,
            "disk_free_mb": round(self.disk_free_mb, 1),
            "detection_source": self.detection_source,
        }

    def render(self) -> str:
        builder = ReportBuilder("HOST HARDWARE")
        builder.kv("CPU cores", self.cpu_count)
        builder.kv("Worker processes", self.n_jobs)
        builder.kv("RAM total", f"{self.ram_total_mb:,.0f} MB")
        builder.kv("RAM available", f"{self.ram_available_mb:,.0f} MB")
        builder.kv("Disk free", f"{self.disk_free_mb:,.0f} MB")
        builder.kv("GPU", "detected" if self.has_gpu else "not detected")
        if self.detection_source != "psutil":
            builder.status(
                WARN,
                "psutil is not installed - RAM figures are conservative defaults, "
                "not measurements (pip install psutil).",
            )
        return builder.build()


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

        # Determine available system RAM.
        memory = _virtual_memory()
        self._detection_source = "psutil" if memory is not None else "fallback-defaults"
        sys_ram_gb = (memory.total / (1024 ** 3)) if memory is not None else (
            _FALLBACK_RAM_MB / 1024.0
        )

        self.memory_limit_mb = (
            (memory_limit_gb * 1024.0) if memory_limit_gb else (sys_ram_gb * 0.8 * 1024.0)
        )

        # True core count, kept distinct from the worker count derived from it.
        self.cpu_count = os.cpu_count() or 4
        if n_jobs is None or n_jobs < 1:
            self.n_jobs = max(1, self.cpu_count - 1)
        else:
            self.n_jobs = min(n_jobs, self.cpu_count)

        self.use_gpu = use_gpu

    def get_system_resources(self) -> SystemResources:
        """Return system hardware resources snapshot."""
        memory = _virtual_memory()
        if memory is not None:
            ram_total_mb = memory.total / (1024 * 1024)
            ram_available_mb = memory.available / (1024 * 1024)
        else:
            ram_total_mb = _FALLBACK_RAM_MB
            ram_available_mb = _FALLBACK_AVAILABLE_MB
        return SystemResources(
            cpu_count=self.cpu_count,
            ram_total_mb=ram_total_mb,
            ram_available_mb=ram_available_mb,
            n_jobs=self.n_jobs,
            has_gpu=self.use_gpu or detect_gpu(),
            disk_free_mb=_free_disk_mb(),
            detection_source=self._detection_source,
        )

    def assess(
        self,
        required_mb: float,
        label: str = "workload",
        remedies: Optional[List[str]] = None,
        safety_fraction: float = 0.85,
    ) -> CapabilityVerdict:
        """Decide whether the host can take a workload of ``required_mb``.

        ``ALLOW`` below half of usable headroom, ``DEGRADE`` up to the safety
        fraction, ``REFUSE`` beyond it. Measured against *available* RAM rather
        than total, since the rest of the OS does not vanish during training.
        """
        resources = self.get_system_resources()
        usable_mb = min(resources.ram_available_mb, self.memory_limit_mb) * safety_fraction
        reasons: List[str] = []
        suggestions = list(remedies or [])

        if required_mb <= usable_mb * 0.5:
            decision = ALLOW
        elif required_mb <= usable_mb:
            decision = DEGRADE
            reasons.append(
                f"projected peak for {label} ({required_mb:,.0f} MB) uses most of the "
                f"{usable_mb:,.0f} MB safely available"
            )
        else:
            decision = REFUSE
            reasons.append(
                f"projected peak for {label} ({required_mb:,.0f} MB) exceeds the "
                f"{usable_mb:,.0f} MB safely available on this machine"
            )
            if not suggestions:
                suggestions = [
                    "reduce the dataset size, batch size, or model size",
                    "raise the ceiling explicitly with --memory-limit-gb if you accept the risk",
                ]

        if resources.detection_source != "psutil":
            reasons.append(
                "RAM figures are conservative defaults because psutil is not installed"
            )

        return CapabilityVerdict(
            decision=decision,
            required_mb=float(required_mb),
            available_mb=float(usable_mb),
            reasons=reasons,
            remedies=suggestions,
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

        verdict = self.assess(
            est_peak_mb,
            label=f"{mode} training on {n_samples:,} rows",
            remedies=[
                f"sample the dataset, e.g. the first {max(1000, n_samples // 4):,} rows",
                "use --mode fast, which lowers the peak-memory multiplier",
                "raise the ceiling explicitly with --memory-limit-gb if you accept the risk",
            ],
        )
        warnings.extend(verdict.reasons)

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
            verdict=verdict,
        )
