"""Video modality for DIVE Deep Learning - ``dive/dl/modalities/video.py``.

Videos are sampled across uniform time intervals into a sequence of frames,
featurised through :class:`~dive.dl.modalities.image.ImageAdapter`, and
temporally pooled (mean and standard deviation across frames) into a fixed-length
numeric representation. That architecture keeps video classification compatible
with the shared modality-blind MLP trainer without exploding memory consumption.

Decoders tried, in order:
1. ``av`` (PyAV) - robust cross-platform decoding with uniform packet seeking.
2. ``cv2`` (OpenCV) - widely available fallback decoding via ``cv2.VideoCapture``.
3. Direct frame sequence / array fallback - already decoded frames or synthetic clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from dive.dl.config import DLDataConfig, Modality
from dive.dl.exceptions import DLBackendError, DLDataError
from dive.dl.modalities.base import (
    VIDEO_SUFFIXES,
    ModalityAdapter,
    RawBatch,
    load_media_manifest,
)
from dive.dl.modalities.image import ImageAdapter
from dive.utils.optional import is_available, load_optional

#: Default frames sampled per video clip.
DEFAULT_FRAMES_PER_CLIP = 8

#: Approximate memory requirement per video sample (8 frames * 224x224x3 float).
DEFAULT_BYTES_PER_SAMPLE = 2_400_000.0


class VideoAdapter(ModalityAdapter):
    """Turns video clips - paths, containers, or frame sequences - into dense vectors."""

    modality = Modality.VIDEO
    accelerated_packages = ("av", "torchvision", "opencv-python")
    required_packages = ("pillow",)
    fallback_caveat = (
        "Without PyAV or OpenCV, video decoding requires already-extracted frame sequences "
        "or supported containers. Videos are sampled across uniform temporal intervals "
        "and aggregated through temporal pooling."
    )
    bytes_per_sample = DEFAULT_BYTES_PER_SAMPLE

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(options)
        self.frames_per_clip = max(1, int(self.option("frames_per_clip", DEFAULT_FRAMES_PER_CLIP)))
        self.image_adapter = ImageAdapter(options=options)
        self._fitted = False

    # ------------------------------------------------------------------
    def load(self, source: Any, data_config: DLDataConfig) -> RawBatch:
        return load_media_manifest(source, data_config, VIDEO_SUFFIXES, "video")

    def _normalize_items(self, inputs: Any) -> List[Any]:
        """Convert arbitrary input batches into a list of clip references or frame sequences."""
        if isinstance(inputs, (str, Path)):
            return [str(inputs)]
        if isinstance(inputs, np.ndarray):
            # A 5-D array is (n_clips, n_frames, H, W, C)
            if inputs.ndim == 5:
                return [clip for clip in inputs]
            # A 4-D array is a single clip (n_frames, H, W, C)
            if inputs.ndim == 4:
                return [inputs]
            if inputs.dtype == object:
                return list(inputs.ravel().tolist())
            return list(inputs.tolist())
        if isinstance(inputs, (list, tuple)):
            # If it is a list of frame images/arrays, treat as 1 clip
            if len(inputs) > 0 and hasattr(inputs[0], "shape") and getattr(inputs[0], "ndim", 0) in (2, 3):
                return [inputs]
            return list(inputs)
        return [inputs]

    def _extract_frames_pyav(self, path: Path) -> List[Any]:
        """Decode video frames uniformly using PyAV."""
        av = load_optional("av", purpose="video decoding")
        if av is None:
            return []
        try:
            container = av.open(str(path))
            video_stream = next(s for s in container.streams if s.type == "video")
            total_frames = video_stream.frames
            if total_frames <= 0:
                # Seek duration approximation
                total_frames = int(video_stream.duration * video_stream.time_base * 25) if video_stream.duration else 100

            step = max(1, total_frames // self.frames_per_clip)
            target_indices = set(range(0, total_frames, step)[:self.frames_per_clip])
            extracted = []

            for idx, frame in enumerate(container.decode(video=0)):
                if idx in target_indices or not target_indices:
                    extracted.append(frame.to_ndarray(format="rgb24"))
                    if len(extracted) >= self.frames_per_clip:
                        break
            container.close()
            return extracted
        except Exception:
            return []

    def _extract_frames_cv2(self, path: Path) -> List[Any]:
        """Decode video frames uniformly using OpenCV."""
        cv2 = load_optional("opencv-python", purpose="video decoding")
        if cv2 is None:
            return []
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 100

            step = max(1, total_frames // self.frames_per_clip)
            indices = [i * step for i in range(self.frames_per_clip)]
            frames = []

            for target_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                else:
                    break
            cap.release()
            return frames
        except Exception:
            return []

    def _decode_clip(self, clip: Any) -> List[Any]:
        """Extract or parse frames for a single clip."""
        # Case 1: Already a sequence of frames / ndarray
        if isinstance(clip, np.ndarray):
            if clip.ndim == 4:  # (n_frames, H, W, C)
                return [clip[i] for i in range(min(len(clip), self.frames_per_clip))]
            if clip.ndim in (2, 3):  # Single image/frame
                return [clip] * self.frames_per_clip
        if isinstance(clip, (list, tuple)) and len(clip) > 0 and not isinstance(clip[0], (str, Path)):
            return list(clip)[:self.frames_per_clip]

        # Case 2: File path
        path = Path(str(clip))
        if not path.is_file():
            return []

        # Try decoders
        frames = self._extract_frames_pyav(path)
        if not frames:
            frames = self._extract_frames_cv2(path)

        if not frames:
            # Fallback for mock/plain files or when decoders are unavailable:
            # Generate deterministic surrogate frames so training pipeline remains executable
            try:
                from PIL import Image

                with Image.open(path) as img:
                    img_rgb = img.convert("RGB")
                    return [np.array(img_rgb)] * self.frames_per_clip
            except Exception:
                pass

        return frames

    # ------------------------------------------------------------------
    def fit_transform(self, inputs: Any) -> np.ndarray:
        items = self._normalize_items(inputs)
        all_clips_frames: List[List[Any]] = []
        flat_frames_for_fit: List[Any] = []
        self.failed_indices = []

        for idx, item in enumerate(items):
            frames = self._decode_clip(item)
            if not frames:
                self.failed_indices.append(idx)
                zeros_frame = np.zeros((32, 32, 3), dtype=np.uint8)
                frames = [zeros_frame] * self.frames_per_clip
            all_clips_frames.append(frames)
            flat_frames_for_fit.extend(frames[:2])

        # Fit underlying image adapter
        if not flat_frames_for_fit:
            flat_frames_for_fit = [np.zeros((32, 32, 3), dtype=np.uint8)]
        self.image_adapter.fit_transform(flat_frames_for_fit)

        # Transform and pool each clip
        clip_features = []
        for frames in all_clips_frames:
            # shape: (n_frames, n_image_features)
            frame_feats = self.image_adapter.transform(frames)
            if len(frame_feats) == 0:
                frame_feats = np.zeros((self.frames_per_clip, len(self.image_adapter.feature_names) or 1))

            mean_pool = np.mean(frame_feats, axis=0)
            std_pool = np.std(frame_feats, axis=0)
            clip_vector = np.concatenate([mean_pool, std_pool], axis=0)
            clip_features.append(clip_vector)

        self.fitted = True
        self.extractor = f"video-sampled-frames({self.frames_per_clip}) + {self.image_adapter.extractor}"
        base_names = self.image_adapter.feature_names or [f"f{i}" for i in range(clip_features[0].shape[0] // 2)]
        self.feature_names = [f"mean_{n}" for n in base_names] + [f"std_{n}" for n in base_names]

        return np.asarray(clip_features, dtype=np.float32)

    def transform(self, inputs: Any) -> np.ndarray:
        if not self.fitted:
            raise DLBackendError("VideoAdapter must be fit before transform() is called.")
        items = self._normalize_items(inputs)
        self.failed_indices = []
        clip_features = []

        for idx, item in enumerate(items):
            frames = self._decode_clip(item)
            if not frames:
                self.failed_indices.append(idx)
                zeros_frame = np.zeros((32, 32, 3), dtype=np.uint8)
                frames = [zeros_frame] * self.frames_per_clip

            frame_feats = self.image_adapter.transform(frames)
            if len(frame_feats) == 0:
                frame_feats = np.zeros((self.frames_per_clip, len(self.feature_names) // 2))

            mean_pool = np.mean(frame_feats, axis=0)
            std_pool = np.std(frame_feats, axis=0)
            clip_vector = np.concatenate([mean_pool, std_pool], axis=0)
            clip_features.append(clip_vector)

        return np.asarray(clip_features, dtype=np.float32)
