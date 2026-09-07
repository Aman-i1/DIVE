"""DIVE Deep Learning Capability Domain - ``dive.dl``.

Modality-agnostic deep learning across tabular data, text documents, image files,
audio recordings, and video streams with dual backends (PyTorch MLP and Scikit-Learn fallback).
"""

from __future__ import annotations

from dive.dl.automl import AutoDL, DLLeaderboard, DLTrial
from dive.dl.capability import assess_dl_workload, project_dl_memory_mb
from dive.dl.config import (
    DLConfig,
    DLDataConfig,
    DLResourceConfig,
    DLTask,
    DLTrainingConfig,
    Modality,
)
from dive.dl.core import (
    BACKEND_SKLEARN,
    BACKEND_TORCH,
    CallbackList,
    ConsoleCallback,
    DLTrainer,
    DeviceInfo,
    EpochRecord,
    HistoryCallback,
    TrainingCallback,
    select_device,
)
from dive.dl.exceptions import (
    DLBackendError,
    DLConfigError,
    DLDataError,
    DLError,
    DLInferenceError,
    DLModalityError,
    DLTrainingError,
)
from dive.dl.inference import (
    DLPredictor,
    load_dl_predictor,
    save_dl_predictor,
)
from dive.dl.modalities import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    AudioAdapter,
    ImageAdapter,
    ModalityAdapter,
    RawBatch,
    TabularAdapter,
    TextAdapter,
    VideoAdapter,
    available_modalities,
    get_adapter,
)

__all__ = [
    "AUDIO_SUFFIXES",
    "AutoDL",
    "AudioAdapter",
    "BACKEND_SKLEARN",
    "BACKEND_TORCH",
    "CallbackList",
    "ConsoleCallback",
    "DLBackendError",
    "DLConfig",
    "DLConfigError",
    "DLDataConfig",
    "DLDataError",
    "DLError",
    "DLInferenceError",
    "DLLeaderboard",
    "DLModalityError",
    "DLPredictor",
    "DLResourceConfig",
    "DLTask",
    "DLTrainer",
    "DLTrainingConfig",
    "DLTrainingError",
    "DLTrial",
    "DeviceInfo",
    "EpochRecord",
    "HistoryCallback",
    "IMAGE_SUFFIXES",
    "ImageAdapter",
    "Modality",
    "ModalityAdapter",
    "RawBatch",
    "TabularAdapter",
    "TextAdapter",
    "TrainingCallback",
    "VIDEO_SUFFIXES",
    "VideoAdapter",
    "assess_dl_workload",
    "available_modalities",
    "get_adapter",
    "load_dl_predictor",
    "project_dl_memory_mb",
    "save_dl_predictor",
    "select_device",
]
