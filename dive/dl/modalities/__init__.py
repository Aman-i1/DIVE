"""Modalities package for DIVE Deep Learning - ``dive.dl.modalities``.

Provides adapters that translate modality-specific raw inputs (tabular frames,
text documents, audio clips, image files, and video streams) into dense numeric
feature matrices for the shared DL training core.
"""

from __future__ import annotations

from dive.dl.modalities.audio import AudioAdapter
from dive.dl.modalities.base import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    ModalityAdapter,
    RawBatch,
    available_modalities,
    get_adapter,
    load_media_manifest,
    scan_directory,
)
from dive.dl.modalities.image import ImageAdapter
from dive.dl.modalities.tabular import TabularAdapter
from dive.dl.modalities.text import TextAdapter
from dive.dl.modalities.video import VideoAdapter

__all__ = [
    "AUDIO_SUFFIXES",
    "AudioAdapter",
    "IMAGE_SUFFIXES",
    "ImageAdapter",
    "ModalityAdapter",
    "RawBatch",
    "TabularAdapter",
    "TextAdapter",
    "VIDEO_SUFFIXES",
    "VideoAdapter",
    "available_modalities",
    "get_adapter",
    "load_media_manifest",
    "scan_directory",
]
