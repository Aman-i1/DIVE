"""Compute-device selection for DIVE Deep Learning - ``dive/dl/core/device.py``.

Answers one question - *what will this actually train on?* - without importing
torch unless it is already installed. ``nvidia-smi`` detection is reused from
:func:`dive.utils.optional.detect_gpu` so a host with a GPU but no torch is still
reported honestly: the card exists, the runtime to use it does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from dive.utils.optional import detect_gpu, load_optional
from dive.utils.report import INFO, PASS, WARN, ReportBuilder

#: Device preferences accepted from the CLI and config.
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


@dataclass
class DeviceInfo:
    """Resolved compute device plus how the decision was reached."""

    name: str = "cpu"
    torch_available: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    mps_available: bool = False
    #: A GPU visible to the OS. True with ``cuda_available`` False means the card
    #: is there but torch cannot use it - a CPU-only wheel, or no torch at all.
    gpu_detected: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[float] = None
    reason: str = "torch is not installed; using the scikit-learn fallback on CPU"

    @property
    def is_accelerated(self) -> bool:
        return self.name in ("cuda", "mps")

    @property
    def status_token(self) -> str:
        if not self.torch_available:
            return INFO
        return PASS if self.is_accelerated else WARN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.name,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
            "gpu_detected": self.gpu_detected,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": None if self.gpu_memory_mb is None else round(self.gpu_memory_mb, 1),
            "reason": self.reason,
        }

    def render(self) -> str:
        builder = ReportBuilder("DIVE DL COMPUTE DEVICE")
        builder.kv("Selected device", self.name)
        builder.kv("PyTorch", self.torch_version or "not installed")
        builder.kv("CUDA usable", "yes" if self.cuda_available else "no")
        builder.kv("Apple MPS usable", "yes" if self.mps_available else "no")
        builder.kv("GPU visible to OS", "yes" if self.gpu_detected else "no")
        if self.gpu_name:
            builder.kv("GPU", self.gpu_name)
        if self.gpu_memory_mb:
            builder.kv("GPU memory", f"{self.gpu_memory_mb:,.0f} MB")
        builder.status(self.status_token, self.reason)
        return builder.build()


def _cuda_details(torch: Any) -> DeviceInfo:
    """Read CUDA/MPS capability off an imported torch, defensively.

    Every probe is individually guarded: a broken CUDA install raises from
    ``torch.cuda.is_available()`` on some driver versions, and that must degrade
    to CPU rather than abort the command.
    """
    info = DeviceInfo(torch_available=True, torch_version=getattr(torch, "__version__", None))
    try:
        info.cuda_available = bool(torch.cuda.is_available())
    except Exception:
        info.cuda_available = False
    if info.cuda_available:
        try:
            info.gpu_name = str(torch.cuda.get_device_name(0))
            properties = torch.cuda.get_device_properties(0)
            info.gpu_memory_mb = float(properties.total_memory) / (1024 * 1024)
        except Exception:
            pass
    try:
        backends = getattr(torch.backends, "mps", None)
        info.mps_available = bool(backends is not None and backends.is_available())
    except Exception:
        info.mps_available = False
    return info


def select_device(preference: str = "auto") -> DeviceInfo:
    """Resolve ``preference`` against what this host can actually do.

    An explicit ``cuda``/``mps`` that is unavailable falls back to CPU with the
    reason recorded, rather than raising. Refusing a run is reserved for cases
    where continuing would harm the host (see :mod:`dive.dl.capability`); a
    missing accelerator only makes training slower.
    """
    choice = str(preference or "auto").strip().lower()
    if choice not in DEVICE_CHOICES:
        choice = "auto"

    gpu_detected = detect_gpu()
    torch = load_optional("torch")

    if torch is None:
        return DeviceInfo(
            name="cpu",
            torch_available=False,
            gpu_detected=gpu_detected,
            reason=(
                "torch is not installed - training will use the scikit-learn "
                "fallback on CPU. Run 'dive dl doctor' to install it."
            ),
        )

    info = _cuda_details(torch)
    info.gpu_detected = gpu_detected or info.cuda_available

    if choice == "cpu":
        info.name = "cpu"
        info.reason = "CPU requested explicitly"
        return info

    if choice == "cuda":
        if info.cuda_available:
            info.name = "cuda"
            info.reason = f"CUDA requested and available ({info.gpu_name or 'GPU 0'})"
        else:
            info.name = "cpu"
            info.reason = (
                "CUDA was requested but torch cannot see a CUDA device; falling back to CPU"
                + (" (a GPU is visible to the OS, so this torch build is likely CPU-only)"
                   if gpu_detected else "")
            )
        return info

    if choice == "mps":
        if info.mps_available:
            info.name = "mps"
            info.reason = "Apple MPS requested and available"
        else:
            info.name = "cpu"
            info.reason = "Apple MPS was requested but is unavailable; falling back to CPU"
        return info

    # auto
    if info.cuda_available:
        info.name = "cuda"
        info.reason = f"CUDA selected automatically ({info.gpu_name or 'GPU 0'})"
    elif info.mps_available:
        info.name = "mps"
        info.reason = "Apple MPS selected automatically"
    else:
        info.name = "cpu"
        info.reason = (
            "no usable accelerator found; training on CPU"
            + (" although a GPU is visible to the OS - check the torch build"
               if gpu_detected else "")
        )
    return info
