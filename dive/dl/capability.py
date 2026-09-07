"""Pre-flight host hardware capability assessment for DIVE Deep Learning.

Projects RAM, GPU headroom, and execution time before starting a heavy deep learning
training run across tabular, text, image, audio, or video modalities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dive.dl.config import DLConfig, Modality
from dive.dl.modalities.base import get_adapter
from dive.resources import (
    ALLOW,
    DEGRADE,
    REFUSE,
    CapabilityVerdict,
    ResourceManager,
    SystemResources,
)
from dive.utils.report import ReportBuilder, WARN


def project_dl_memory_mb(
    modality: Union[str, Modality],
    n_samples: int,
    config: Optional[DLConfig] = None,
) -> float:
    """Project the peak memory consumption (in MB) for a DL training workload."""
    cfg = config or DLConfig(modality=str(modality))
    resolved_modality = Modality.from_str(modality) if not isinstance(modality, Modality) else modality

    try:
        adapter = get_adapter(resolved_modality, options=cfg.data.options)
        bytes_per_sample = getattr(adapter, "bytes_per_sample", 4096.0)
    except Exception:
        bytes_per_sample = 4096.0

    # Raw features in memory: n_samples * bytes_per_sample
    raw_data_mb = (n_samples * bytes_per_sample) / (1024.0 * 1024.0)

    # Optimization overhead multiplier: gradients, activations, optimizer states, batches
    # fast mode uses smaller batch overhead; quality mode buffers more states
    multiplier = 1.8 if cfg.mode == "fast" else (2.5 if cfg.mode == "balanced" else 3.5)
    model_overhead_mb = 120.0  # Base framework footprint (torch/scikit-learn runtime)

    projected_peak_mb = (raw_data_mb * multiplier) + model_overhead_mb
    return max(150.0, projected_peak_mb)


def assess_dl_workload(
    modality: Union[str, Modality],
    n_samples: int,
    config: Optional[DLConfig] = None,
    manager: Optional[ResourceManager] = None,
) -> CapabilityVerdict:
    """Evaluate whether the current host can safely execute this DL workload."""
    cfg = config or DLConfig(modality=str(modality))
    res_mgr = manager or ResourceManager(
        time_budget_sec=cfg.resources.time_budget_secs,
        memory_limit_gb=cfg.resources.memory_budget_mb / 1024.0 if cfg.resources.memory_budget_mb else None,
    )

    projected_mb = project_dl_memory_mb(modality, n_samples, cfg)
    modality_name = getattr(modality, "value", str(modality))
    label = f"{modality_name} DL training on {n_samples:,} samples ({cfg.mode} mode)"

    remedies = [
        f"reduce dataset size or batch size (currently {cfg.resources.batch_size})",
        "use --mode fast to reduce peak activation caching and epoch budgets",
        "switch device using --device cpu or expand hardware limits via --memory-limit-gb",
    ]

    return res_mgr.assess(
        required_mb=projected_mb,
        label=label,
        remedies=remedies,
        safety_fraction=cfg.resources.safety_fraction,
    )
