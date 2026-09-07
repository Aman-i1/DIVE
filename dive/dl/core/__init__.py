"""Training core for DIVE Deep Learning.

``device`` resolves *where* training runs, ``trainer`` runs it, ``callbacks``
reports it. Nothing here knows what a modality is - see
:mod:`dive.dl.modalities`.
"""

from __future__ import annotations

from dive.dl.core.callbacks import (
    CallbackList,
    ConsoleCallback,
    EpochRecord,
    HistoryCallback,
    TrainingCallback,
)
from dive.dl.core.device import DEVICE_CHOICES, DeviceInfo, select_device
from dive.dl.core.trainer import (
    BACKEND_SKLEARN,
    BACKEND_TORCH,
    DLTrainer,
)

__all__ = [
    "BACKEND_SKLEARN",
    "BACKEND_TORCH",
    "CallbackList",
    "ConsoleCallback",
    "DEVICE_CHOICES",
    "DLTrainer",
    "DeviceInfo",
    "EpochRecord",
    "HistoryCallback",
    "TrainingCallback",
    "select_device",
]
