"""Image modality for DIVE Deep Learning - ``dive/dl/modalities/image.py``.

Two tiers:

1. **Frozen pretrained backbone** (``torch`` + ``torchvision``) - a ResNet-18 with
   its classifier head removed, giving 512 semantic features per image. This is
   transfer learning: the backbone stays frozen and the shared MLP is the trained
   head, which is what makes it viable on a CPU-only laptop.
2. **Pixel + colour statistics** (Pillow only) - a downsampled grayscale grid plus
   per-channel histograms. Pillow ships with matplotlib, a hard dependency, so
   this tier is effectively always available.

The fallback is a genuine baseline for coarse tasks (is this a document scan or a
photograph?) and genuinely weak for fine-grained ones. :attr:`fallback_caveat`
says so, and ``dive dl doctor`` prints it.

The featurising path deliberately accepts *already-decoded frames* as well as file
paths, because :mod:`dive.dl.modalities.video` samples frames from a clip and
feeds them through exactly this code rather than duplicating it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dive.dl.config import DLDataConfig, Modality
from dive.dl.exceptions import DLBackendError, DLDataError
from dive.dl.modalities.base import (
    IMAGE_SUFFIXES,
    ModalityAdapter,
    RawBatch,
    load_media_manifest,
)
from dive.utils.optional import is_available, load_optional

#: Edge length the fallback downsamples to. 32x32 grayscale is 1024 features -
#: large enough to carry layout, small enough that an MLP trains in seconds.
DEFAULT_FALLBACK_SIZE = 32

#: Edge length the torch backbone expects. ResNet was trained at 224.
BACKBONE_SIZE = 224

#: Feature width of a ResNet-18 with its classifier head removed.
BACKBONE_WIDTH = 512

#: Histogram bins per colour channel in the fallback.
_HIST_BINS = 16

#: Edge gradient, contrast, and color balance statistics.
_EDGE_STAT_COUNT = 8


class ImageAdapter(ModalityAdapter):

    """Turns images - file paths, arrays or PIL objects - into dense vectors."""

    modality = Modality.IMAGE
    accelerated_packages = ("torch", "torchvision")
    required_packages = ("pillow",)
    fallback_caveat = (
        "Without torch + torchvision, images are represented by a downsampled "
        "grayscale grid and colour histograms. That separates visually distinct "
        "classes (documents vs photos, day vs night) but will not do fine-grained "
        "recognition - it has no learned notion of shape or object."
    )
    bytes_per_sample = 600_000.0  # a decoded 224x224x3 float tensor plus overhead

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(options)
        self._backbone: Any = None
        self._preprocess: Any = None
        self._torch: Any = None
        self._size = int(self.option("image_size", DEFAULT_FALLBACK_SIZE))
        self._use_backbone = False
        self._prepared = False

    # ------------------------------------------------------------------
    def load(self, source: Any, data_config: DLDataConfig) -> RawBatch:
        return load_media_manifest(source, data_config, IMAGE_SUFFIXES, "image")

    def _items(self, inputs: Any) -> List[Any]:
        """Normalise input into a flat list of one-image-per-element.

        A single path, a list of paths, a numpy array of paths, a list of PIL
        images and a list of HxWxC frame arrays all arrive here.
        """
        if isinstance(inputs, (str, Path)):
            return [inputs]
        if isinstance(inputs, np.ndarray):
            if inputs.dtype == object:
                return list(inputs.ravel().tolist())
            # A lone HxW or HxWxC array is one image; a 4-D stack is a batch.
            if inputs.ndim == 4:
                return [frame for frame in inputs]
            if inputs.ndim in (2, 3):
                return [inputs]
            return list(inputs.tolist())
        if isinstance(inputs, (list, tuple)):
            return list(inputs)
        if hasattr(inputs, "convert"):  # a PIL image
            return [inputs]
        raise DLDataError(
            f"Could not interpret image input of type {type(inputs).__name__}.",
            "Pass a file path, a list of file paths, or decoded frames.",
        )

    def _open(self, item: Any) -> Any:
        """Return an RGB PIL image for ``item``, whatever form it arrived in."""
        from PIL import Image

        if isinstance(item, (str, Path)):
            with Image.open(item) as handle:
                # ``.convert`` copies, so the returned image outlives the context.
                return handle.convert("RGB")
        if isinstance(item, np.ndarray):
            array = item
            if array.dtype != np.uint8:
                # Float frames may be 0-1 or 0-255; scale only when they are 0-1.
                finite = np.nan_to_num(array.astype(np.float32), nan=0.0)
                scale = 255.0 if float(np.nanmax(np.abs(finite)) or 0.0) <= 1.0 else 1.0
                array = np.clip(finite * scale, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                return Image.fromarray(array, mode="L").convert("RGB")
            return Image.fromarray(array[:, :, :3]).convert("RGB")
        if hasattr(item, "convert"):
            return item.convert("RGB")
        raise DLDataError(f"Cannot decode an image from {type(item).__name__}.")

    # ------------------------------------------------------------------
    # tier selection
    # ------------------------------------------------------------------
    def prepare(self) -> bool:
        """Choose the featuriser tier once, and report whether torch won.

        Split out of :meth:`fit_transform` so the video adapter can select the tier
        before it starts decoding frames - the choice determines how much memory a
        batch of frames will occupy.
        """
        if self._prepared:
            return self._use_backbone
        self._use_backbone = bool(self.option("use_backbone", True)) and self._load_backbone()
        self._prepared = True
        return self._use_backbone

    def _load_backbone(self) -> bool:
        """Build a frozen ResNet-18 feature extractor, or report failure.

        Weight download is the common failure here (offline host, proxy). It is
        treated as a fallback trigger with the reason recorded, not an error.
        """
        if not (is_available("torch") and is_available("torchvision")):
            return False
        torch = load_optional("torch", purpose="image feature extraction")
        torchvision = load_optional("torchvision", purpose="pretrained image backbones")
        if torch is None or torchvision is None:
            return False
        try:
            # ``weights=`` replaced ``pretrained=`` in torchvision 0.13; both are
            # probed so the adapter works on either. The last candidate builds an
            # untrained backbone, used only when the weight download fails - and it
            # records a note, because random features are worth knowing about.
            model = None
            for index, build in enumerate(
                (
                    lambda: torchvision.models.resnet18(
                        weights=torchvision.models.ResNet18_Weights.DEFAULT
                    ),
                    lambda: torchvision.models.resnet18(pretrained=True),
                    lambda: torchvision.models.resnet18(weights=None),
                )
            ):
                try:
                    model = build()
                    if index == 2:
                        self.notes.append(
                            "Pretrained ResNet-18 weights could not be downloaded, so "
                            "the backbone is randomly initialised - it is a random "
                            "projection, not transfer learning. Expect the "
                            "pixel-statistics fallback to do at least as well."
                        )
                    break
                except Exception:
                    continue
            if model is None:
                return False
            model.fc = torch.nn.Identity()
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            self._backbone = model
            self._torch = torch
            self._preprocess = self._build_preprocess(torchvision)
            return True
        except Exception as exc:
            self.notes.append(
                f"Could not initialise the torchvision backbone ({type(exc).__name__}: {exc}); "
                "using the pixel-statistics fallback."
            )
            return False

    def _build_preprocess(self, torchvision: Any) -> Any:
        transforms = torchvision.transforms
        return transforms.Compose(
            [
                transforms.Resize((BACKBONE_SIZE, BACKBONE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # ------------------------------------------------------------------
    # featurising
    # ------------------------------------------------------------------
    def width(self) -> int:
        """Feature count for the currently selected tier."""
        if self._use_backbone:
            return BACKBONE_WIDTH
        return self._size * self._size + 3 * _HIST_BINS + _EDGE_STAT_COUNT


    def featurise(self, items: Sequence[Any]) -> np.ndarray:
        """Featurise decoded-or-on-disk images with the selected tier.

        Public because the video adapter calls it per clip. Failed items are
        recorded in :attr:`failed_indices` and emitted as zero rows so the caller
        can drop them from ``X`` *and* ``y`` together.
        """
        self.prepare()
        listed = list(items)
        return (
            self._backbone_features(listed)
            if self._use_backbone
            else self._fallback_features(listed)
        )

    def _backbone_features(self, items: List[Any]) -> np.ndarray:
        torch = self._torch
        batch_size = int(self.option("batch_size", 16))
        vectors: List[np.ndarray] = []
        pending: List[Any] = []

        def flush() -> None:
            """Run the batch built so far, appending its rows in order.

            Called on a decode failure as well as when the batch fills, so that a
            zero row for a corrupt file lands at the right position rather than
            after the images already queued behind it.
            """
            if not pending:
                return
            with torch.no_grad():
                output = self._backbone(torch.stack(pending)).cpu().numpy()
            for row in output:
                vectors.append(np.asarray(row, dtype=np.float32))
            pending.clear()

        for position, item in enumerate(items):
            try:
                pending.append(self._preprocess(self._open(item)))
            except Exception:
                flush()
                self.failed_indices.append(position)
                vectors.append(np.zeros(BACKBONE_WIDTH, dtype=np.float32))
            if len(pending) >= batch_size:
                flush()
        flush()
        return (
            np.vstack(vectors)
            if vectors
            else np.zeros((0, BACKBONE_WIDTH), dtype=np.float32)
        )

    def _fallback_features(self, items: List[Any]) -> np.ndarray:
        if load_optional("pillow", purpose="image decoding") is None:
            raise DLBackendError(
                "No image decoder is available: Pillow is not installed.",
                "Install it with 'pip install pillow' (a few MB), "
                "or run 'dive dl doctor' for guided setup.",
            )
        width = self.width()
        rows: List[np.ndarray] = []
        for position, item in enumerate(items):
            try:
                rows.append(self._pixel_statistics(self._open(item)))
            except Exception:
                self.failed_indices.append(position)
                rows.append(np.zeros(width, dtype=np.float32))
        return np.vstack(rows) if rows else np.zeros((0, width), dtype=np.float32)

    def _pixel_statistics(self, image: Any) -> np.ndarray:
        small = image.resize((self._size, self._size))
        gray_2d = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
        gray = gray_2d.ravel()
        channels = np.asarray(small, dtype=np.uint8)
        blocks: List[np.ndarray] = [gray]
        for channel in range(3):
            counts, _ = np.histogram(
                channels[:, :, channel], bins=_HIST_BINS, range=(0, 256)
            )
            total = counts.sum() or 1
            blocks.append(counts.astype(np.float32) / float(total))

        # Edge gradients and texture statistics
        grad_x = np.diff(gray_2d, axis=1)
        grad_y = np.diff(gray_2d, axis=0)
        if grad_x.shape[0] > 1 and grad_y.shape[1] > 1:
            grad_mag = np.sqrt(grad_x[:-1, :] ** 2 + grad_y[:, :-1] ** 2)
        else:
            grad_mag = np.zeros(1, dtype=np.float32)

        grad_mean = float(np.mean(grad_mag))
        grad_std = float(np.std(grad_mag))
        grad_max = float(np.max(grad_mag)) if grad_mag.size > 0 else 0.0
        contrast = float(np.std(gray))
        dyn_range = float(np.max(gray) - np.min(gray))
        r_mean = float(channels[:, :, 0].mean() / 255.0)
        g_mean = float(channels[:, :, 1].mean() / 255.0)
        b_mean = float(channels[:, :, 2].mean() / 255.0)
        edge_stats = np.asarray(
            [grad_mean, grad_std, grad_max, contrast, dyn_range, r_mean, g_mean, b_mean],
            dtype=np.float32,
        )
        blocks.append(edge_stats)
        return np.concatenate(blocks).astype(np.float32)


    # ------------------------------------------------------------------
    def fit_transform(self, inputs: Any) -> np.ndarray:
        items = self._items(inputs)
        if not items:
            raise DLDataError("No images to featurise.")
        self.notes = []
        self.failed_indices = []

        matrix = self.featurise(items)
        if self._use_backbone:
            self.extractor = (
                f"frozen ResNet-18 features ({matrix.shape[1]} dims, {BACKBONE_SIZE}px)"
            )
            prefix = "resnet"
        else:
            self.extractor = (
                f"pixel + colour statistics ({self._size}x{self._size} grayscale "
                f"+ {3 * _HIST_BINS}-bin histograms)"
            )
            prefix = "px"
            self.notes.append(self.fallback_caveat)
        self.feature_names = [f"{prefix}{index}" for index in range(matrix.shape[1])]
        self._report_failures(len(items))
        self.fitted = True
        return matrix

    def transform(self, inputs: Any) -> np.ndarray:
        if not self.fitted:
            raise DLDataError("The image adapter must be fitted before transform().")
        items = self._items(inputs)
        self.failed_indices = []
        matrix = self.featurise(items)
        self._report_failures(len(items))
        return matrix

    def _report_failures(self, total: int) -> None:
        if not self.failed_indices:
            return
        self.notes.append(
            f"{len(self.failed_indices)} of {total} image(s) could not be decoded and "
            "were excluded (corrupt file, unsupported format, or truncated download)."
        )

    def coerce(self, data: Any) -> Any:
        return self._items(data)

    # A torch module and its transform pipeline are rebuilt on load rather than
    # pickled: the weights are a cached download, not user data, and pickling them
    # would add ~45 MB to every saved predictor.
    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_backbone"] = None
        state["_preprocess"] = None
        state["_torch"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._backbone = None
        self._preprocess = None
        self._torch = None
        if self._use_backbone and not self._load_backbone():
            raise DLBackendError(
                "This model was trained on torchvision ResNet-18 features, which "
                "are unavailable here.",
                "Install them with 'pip install torch torchvision', or retrain "
                "with --no-backbone to use the pixel-statistics representation.",
            )
        # Pin the tier that was used for training. Re-running the selection here
        # would silently switch feature width if torch appeared on the host after
        # the model was saved - and the trainer would then reject the input.
        self._prepared = True
