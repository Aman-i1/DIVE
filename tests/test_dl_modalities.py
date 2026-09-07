"""Unit tests for all 5 DIVE Deep Learning modality adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dive.dl.config import DLDataConfig, Modality
from dive.dl.modalities import (
    AudioAdapter,
    ImageAdapter,
    TabularAdapter,
    TextAdapter,
    VideoAdapter,
    available_modalities,
    get_adapter,
)


def test_modality_registry():
    modalities = available_modalities()
    assert "tabular" in modalities
    assert "text" in modalities
    assert "image" in modalities
    assert "audio" in modalities
    assert "video" in modalities

    for m in modalities:
        adapter = get_adapter(m)
        assert adapter is not None
        assert adapter.modality.value == m


def test_tabular_adapter(tmp_path: Path):
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat1": ["a", "b", "a", "b", "c", "c"],
            "target": [0, 1, 0, 1, 1, 0],
        }
    )
    csv_path = tmp_path / "tab.csv"
    df.to_csv(csv_path, index=False)

    adapter = TabularAdapter()
    batch = adapter.load(csv_path, DLDataConfig(target_column="target"))
    assert batch.n_samples == 6
    assert batch.has_targets

    X = adapter.fit_transform(batch.inputs)
    assert X.shape[0] == 6
    assert X.ndim == 2
    assert adapter.fitted

    X2 = adapter.transform(batch.inputs)
    assert X2.shape == X.shape


def test_text_adapter(tmp_path: Path):
    df = pd.DataFrame(
        {
            "text": [
                "Fast customer support and great product quality",
                "Terrible service, slow delivery and damaged box",
                "Super happy with the purchase, five stars",
                "Not working as described, asking for refund",
            ],
            "label": ["positive", "negative", "positive", "negative"],
        }
    )
    csv_path = tmp_path / "corpus.csv"
    df.to_csv(csv_path, index=False)

    adapter = TextAdapter()
    batch = adapter.load(csv_path, DLDataConfig(input_column="text", target_column="label"))
    assert batch.n_samples == 4
    assert batch.has_targets

    X = adapter.fit_transform(batch.inputs)
    assert X.shape[0] == 4
    assert X.ndim == 2
    assert adapter.fitted

    X2 = adapter.transform(["A brand new review to score"])
    assert X2.shape == (1, X.shape[1])


def test_image_adapter_synthetic_frames():
    adapter = ImageAdapter()
    # 4 synthetic RGB image frames of shape (32, 32, 3)
    frames = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(4)]

    X = adapter.fit_transform(frames)
    assert X.shape[0] == 4
    assert X.ndim == 2
    assert adapter.fitted

    X2 = adapter.transform([frames[0]])
    assert X2.shape == (1, X.shape[1])


def test_audio_adapter_synthetic_signals():
    adapter = AudioAdapter()
    # 3 synthetic audio waveforms (1 second of noise at 22050 Hz)
    signals = [np.random.randn(22050).astype(np.float32) for _ in range(3)]

    X = adapter.fit_transform(signals)
    assert X.shape[0] == 3
    assert X.ndim == 2
    assert adapter.fitted

    X2 = adapter.transform([signals[0]])
    assert X2.shape == (1, X.shape[1])


def test_video_adapter_synthetic_clips():
    adapter = VideoAdapter()
    # 2 synthetic clips, each with 4 RGB frames
    clip1 = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(4)]
    clip2 = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(4)]

    X = adapter.fit_transform([clip1, clip2])
    assert X.shape[0] == 2
    assert X.ndim == 2
    assert adapter.fitted

    X2 = adapter.transform([clip1])
    assert X2.shape == (1, X.shape[1])
