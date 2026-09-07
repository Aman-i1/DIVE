"""Modality adapter contract and shared loaders - ``dive/dl/modalities/base.py``.

Five modalities, one trainer. The seam that makes that possible is narrow and
stated here: **an adapter loads raw input and turns it into a dense numeric
feature matrix.** Everything downstream (:mod:`dive.dl.core.trainer`,
:mod:`dive.dl.automl`, :mod:`dive.dl.inference`) is modality-blind.

Adapters are deliberately honest about fidelity. Each declares:

* ``accelerated_packages`` - what unlocks the good featuriser (torch, torchvision,
  librosa, ...). Absent, the adapter still runs.
* ``required_packages`` - what even the fallback cannot do without. Absent, the
  command refuses with an install hint rather than producing nonsense.
* ``fallback_caveat`` - one sentence stating how much worse the fallback is. This
  is printed, not buried; a weak baseline presented as a model is the worse bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dive.dl.config import DLDataConfig, Modality
from dive.dl.exceptions import DLDataError, DLModalityError
from dive.utils.io import load_dataframe, resolve_path

#: File suffixes recognised when scanning a directory or validating a manifest.
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".ppm"}
)
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus"})
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"})

#: Column names tried, in order, when the user does not name the input column.
_PATH_COLUMN_CANDIDATES = ("path", "filepath", "file_path", "file", "filename", "image", "audio", "video", "clip")
_TEXT_COLUMN_CANDIDATES = ("text", "sentence", "content", "body", "message", "review", "document")
_TARGET_COLUMN_CANDIDATES = ("label", "labels", "target", "class", "category", "y", "sentiment")


@dataclass
class RawBatch:
    """Loaded-but-not-yet-featurised inputs, plus where they came from."""

    inputs: Any
    targets: Optional[np.ndarray] = None
    ids: List[str] = field(default_factory=list)
    source: str = ""
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.ids)

    @property
    def has_targets(self) -> bool:
        return self.targets is not None and len(self.targets) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "n_samples": self.n_samples,
            "input_column": self.input_column,
            "target_column": self.target_column,
            "has_targets": self.has_targets,
            "notes": list(self.notes),
        }


def pick_column(
    frame: pd.DataFrame,
    explicit: Optional[str],
    candidates: Sequence[str],
    role: str,
    required: bool = True,
) -> Optional[str]:
    """Resolve a column by explicit name, then by convention, then positionally.

    Reports what it chose to the caller instead of guessing again downstream -
    the bug that made ``dive nlp info`` print a column that did not exist.
    """
    if explicit:
        if explicit in frame.columns:
            return explicit
        raise DLDataError(
            f"Column '{explicit}' is not in the data.",
            f"Available columns: {', '.join(map(str, frame.columns[:15]))}",
        )
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    if not required:
        return None
    if len(frame.columns) == 0:
        raise DLDataError(f"The data has no columns, so no {role} column can be resolved.")
    return frame.columns[0]


def scan_directory(root: Path, suffixes: frozenset) -> RawBatch:
    """Load a folder of media files, using subdirectory names as labels.

    Two supported layouts, matching what people actually have on disk::

        root/cat/001.png   root/dog/002.png     -> labelled, class = folder name
        root/001.png       root/002.png         -> unlabelled collection

    A mixed layout (files *and* class folders at the top level) is treated as
    folder-per-class, with the loose files reported as skipped rather than
    silently assigned to a class.
    """
    subdirectories = sorted(child for child in root.iterdir() if child.is_dir())
    notes: List[str] = []

    if subdirectories:
        paths: List[str] = []
        labels: List[str] = []
        for directory in subdirectories:
            found = sorted(
                str(item)
                for item in directory.rglob("*")
                if item.is_file() and item.suffix.lower() in suffixes
            )
            paths.extend(found)
            labels.extend([directory.name] * len(found))
        loose = [
            item
            for item in root.iterdir()
            if item.is_file() and item.suffix.lower() in suffixes
        ]
        if loose:
            notes.append(
                f"{len(loose)} file(s) directly under '{root.name}' were skipped: "
                "in a folder-per-class layout every file must live inside a class folder."
            )
        if not paths:
            raise DLDataError(
                f"No files with a recognised extension were found under '{root}'.",
                f"Expected one of: {', '.join(sorted(suffixes))}",
            )
        return RawBatch(
            inputs=paths,
            targets=np.asarray(labels),
            ids=paths,
            source=str(root),
            target_column="(folder name)",
            notes=notes,
        )

    paths = sorted(
        str(item)
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in suffixes
    )
    if not paths:
        raise DLDataError(
            f"No files with a recognised extension were found under '{root}'.",
            f"Expected one of: {', '.join(sorted(suffixes))}",
        )
    notes.append(
        "No class subdirectories found, so this collection is unlabelled. "
        "Use a folder-per-class layout or a CSV manifest to train a classifier."
    )
    return RawBatch(inputs=paths, ids=paths, source=str(root), notes=notes)


def load_media_manifest(
    source: Any,
    data_config: DLDataConfig,
    suffixes: frozenset,
    modality: str,
) -> RawBatch:
    """Load path-based media from a directory tree or a table of file paths."""
    path = resolve_path(source, must_exist=True, kind="path")
    if path.is_dir():
        return scan_directory(path, suffixes)

    if path.is_file() and path.suffix.lower() in suffixes:
        return RawBatch(
            inputs=[str(path)],
            targets=None,
            ids=[str(path)],
            source=str(path),
        )

    frame = load_dataframe(path)

    if frame.empty:
        raise DLDataError(f"Manifest '{path}' contains no rows.")

    input_column = pick_column(
        frame, data_config.input_column, _PATH_COLUMN_CANDIDATES, f"{modality} path"
    )
    target_column = pick_column(
        frame, data_config.target_column, _TARGET_COLUMN_CANDIDATES, "target", required=False
    )
    if target_column == input_column:
        target_column = None

    # Manifest paths are conventionally relative to the manifest itself, which is
    # what makes a dataset directory portable.
    base = path.parent
    resolved: List[str] = []
    missing: List[str] = []
    kept_rows: List[int] = []
    for position, value in enumerate(frame[input_column].astype(str).tolist()):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_file():
            resolved.append(str(candidate))
            kept_rows.append(position)
        else:
            missing.append(value)

    if not resolved:
        raise DLDataError(
            f"None of the {len(frame)} path(s) in column '{input_column}' could be found.",
            f"Paths are resolved relative to '{base}'. First unresolved entry: "
            f"{missing[0] if missing else 'n/a'}",
        )

    notes: List[str] = []
    if missing:
        notes.append(
            f"{len(missing)} of {len(frame)} manifest path(s) do not exist and were skipped "
            f"(first: {missing[0]})."
        )

    targets = None
    if target_column is not None:
        targets = np.asarray(frame[target_column].iloc[kept_rows].tolist())

    return RawBatch(
        inputs=resolved,
        targets=targets,
        ids=resolved,
        source=str(path),
        input_column=input_column,
        target_column=target_column,
        notes=notes,
    )


class ModalityAdapter:
    """Base class: load raw input, produce a dense feature matrix.

    Subclasses implement :meth:`load`, :meth:`fit_transform` and
    :meth:`transform`. ``fit_transform`` may fit state (a vectoriser, an image
    size, a mel filterbank); ``transform`` must reuse exactly that state, because
    the trainer rejects a feature-count mismatch at inference time.
    """

    modality: Modality = Modality.TABULAR
    #: Packages that upgrade this adapter's featuriser when present.
    accelerated_packages: Tuple[str, ...] = ()
    #: Packages without which not even the fallback can run.
    required_packages: Tuple[str, ...] = ()
    #: One sentence on how much worse the no-torch path is. Printed, not hidden.
    fallback_caveat: str = ""
    #: Rough bytes held in memory per sample while featurising, for the
    #: host-capability projection in :mod:`dive.dl.capability`.
    bytes_per_sample: float = 4096.0

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options: Dict[str, Any] = dict(options or {})
        self.fitted = False
        #: Human-readable name of the featuriser actually used, decided at
        #: ``fit_transform`` time because it depends on what is installed.
        self.extractor = "unset"
        self.notes: List[str] = []
        self.feature_names: List[str] = []
        #: Positions in the most recent input that could not be read (a corrupt
        #: image, an undecodable clip). Rows stay aligned - a zero vector is
        #: emitted - and the caller drops these indices from X *and* y, because a
        #: zero row paired with a real label is training on noise.
        self.failed_indices: List[int] = []

    # -- to implement ---------------------------------------------------

    def load(self, source: Any, data_config: DLDataConfig) -> RawBatch:
        raise NotImplementedError

    def fit_transform(self, inputs: Any) -> np.ndarray:
        raise NotImplementedError

    def transform(self, inputs: Any) -> np.ndarray:
        raise NotImplementedError

    # -- shared ---------------------------------------------------------

    def coerce(self, data: Any) -> Any:
        """Normalise ad-hoc inference input into the shape :meth:`transform` wants.

        Overridden by adapters whose ``transform`` input is not a plain list.
        """
        if isinstance(data, (str, Path)):
            return [str(data)]
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, (list, tuple, np.ndarray)):
            return list(data)
        return [data]

    def option(self, key: str, default: Any) -> Any:
        value = self.options.get(key, default)
        return default if value is None else value

    def describe(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "extractor": self.extractor,
            "n_features": len(self.feature_names) or None,
            "accelerated_packages": list(self.accelerated_packages),
            "required_packages": list(self.required_packages),
            "notes": list(self.notes),
        }


_ADAPTERS: Dict[Modality, str] = {
    Modality.TABULAR: "dive.dl.modalities.tabular:TabularAdapter",
    Modality.TEXT: "dive.dl.modalities.text:TextAdapter",
    Modality.IMAGE: "dive.dl.modalities.image:ImageAdapter",
    Modality.AUDIO: "dive.dl.modalities.audio:AudioAdapter",
    Modality.VIDEO: "dive.dl.modalities.video:VideoAdapter",
}


def get_adapter(modality: Any, options: Optional[Dict[str, Any]] = None) -> ModalityAdapter:
    """Instantiate the adapter for ``modality``.

    Imported lazily and by name so that ``import dive.dl`` does not pull in every
    modality's dependencies - the image adapter should not cost anything to a
    user training on tabular data.
    """
    resolved = Modality.from_str(modality) if not isinstance(modality, Modality) else modality
    target = _ADAPTERS.get(resolved)
    if target is None:
        raise DLModalityError(f"No adapter is registered for modality '{resolved.value}'.")
    module_name, class_name = target.split(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)(options=options)


def available_modalities() -> List[str]:
    """Every modality ``dive dl`` can address."""
    return [modality.value for modality in _ADAPTERS]
