"""Tabular modality for DIVE Deep Learning - ``dive/dl/modalities/tabular.py``.

Deep learning on tables is where the fallback is closest to parity: a
``MLPClassifier`` and a torch MLP over the same encoded matrix differ in speed and
regularisation, not in kind. The interesting work is therefore the encoding -
median-imputed numerics, most-frequent-imputed categoricals, one-hot with a
cardinality cap so a free-text ID column cannot detonate the feature space.

``dive ml train`` remains the better tool for most tables; this exists so the DL
domain covers every modality the user asked for, and so a table can be compared
against the same trainer the other modalities use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dive.dl.config import DLDataConfig, Modality
from dive.dl.exceptions import DLDataError
from dive.dl.modalities.base import (
    ModalityAdapter,
    RawBatch,
    _TARGET_COLUMN_CANDIDATES,
    pick_column,
)
from dive.utils.io import load_dataframe, resolve_path

#: A categorical column with more distinct values than this is dropped rather
#: than one-hot encoded: it is almost always an identifier, and encoding it both
#: explodes the matrix and leaks the row index into the model.
_MAX_CATEGORIES = 32


class TabularAdapter(ModalityAdapter):
    """Encodes a DataFrame of mixed dtypes into a dense numeric matrix."""

    modality = Modality.TABULAR
    fallback_caveat = (
        "The scikit-learn MLP fallback is a fair substitute on tabular data; "
        "gradient boosting via 'dive ml train' is usually still stronger."
    )
    bytes_per_sample = 2048.0

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(options)
        self._numeric: List[str] = []
        self._categorical: List[str] = []
        self._transformer: Any = None

    # ------------------------------------------------------------------
    def load(self, source: Any, data_config: DLDataConfig) -> RawBatch:
        path = resolve_path(source, must_exist=True, kind="dataset")
        if path.is_dir():
            raise DLDataError(
                f"'{path}' is a directory; the tabular modality expects a data file.",
                "Pass a CSV/TSV/JSON/Parquet file, or use --modality image/audio/video.",
            )
        frame = load_dataframe(path)
        if frame.empty:
            raise DLDataError(f"'{path}' contains no rows.")

        target_column = pick_column(
            frame,
            data_config.target_column,
            _TARGET_COLUMN_CANDIDATES,
            "target",
            required=False,
        )
        notes: List[str] = []
        targets = None
        features = frame
        if target_column is not None:
            targets = np.asarray(frame[target_column].tolist())
            features = frame.drop(columns=[target_column])
        else:
            notes.append(
                "No target column was found, so this dataset is unlabelled. "
                "Name one with --target-col to train a model."
            )
        if features.shape[1] == 0:
            raise DLDataError(
                "The dataset has no feature columns once the target is removed.",
                "A single-column file cannot be used for tabular learning.",
            )

        return RawBatch(
            inputs=features,
            targets=targets,
            ids=[str(index) for index in frame.index],
            source=str(path),
            target_column=target_column,
            notes=notes,
        )

    # ------------------------------------------------------------------
    def _as_frame(self, inputs: Any) -> pd.DataFrame:
        if isinstance(inputs, pd.DataFrame):
            return inputs
        if isinstance(inputs, pd.Series):
            return inputs.to_frame()
        if isinstance(inputs, dict):
            return pd.DataFrame([inputs])
        if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], dict):
            return pd.DataFrame(list(inputs))
        try:
            return pd.DataFrame(inputs)
        except Exception as exc:
            raise DLDataError(
                f"Could not interpret tabular input of type {type(inputs).__name__}.",
                "Pass a DataFrame, a dict record, or a list of dict records.",
            ) from exc

    def _one_hot_encoder(self) -> Any:
        """Build a OneHotEncoder across the sklearn versions in the wild.

        ``sparse_output`` replaced ``sparse`` in scikit-learn 1.2, so both spellings
        are probed rather than assumed. ``max_categories`` is not needed: columns
        above :data:`_MAX_CATEGORIES` distinct values are dropped before they reach
        the encoder.
        """
        from sklearn.preprocessing import OneHotEncoder

        for kwargs in (
            {"handle_unknown": "ignore", "sparse_output": False},
            {"handle_unknown": "ignore", "sparse": False},
        ):
            try:
                return OneHotEncoder(**kwargs)
            except TypeError:
                continue
        return OneHotEncoder(handle_unknown="ignore")

    def fit_transform(self, inputs: Any) -> np.ndarray:
        frame = self._as_frame(inputs)
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        numeric = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])
        ]
        remaining = [column for column in frame.columns if column not in numeric]

        categorical: List[str] = []
        dropped: List[str] = []
        for column in remaining:
            distinct = int(frame[column].astype(str).nunique(dropna=True))
            if distinct <= _MAX_CATEGORIES:
                categorical.append(column)
            else:
                dropped.append(f"{column} ({distinct} distinct)")

        if not numeric and not categorical:
            raise DLDataError(
                "No usable feature columns: every column is high-cardinality text.",
                "Use --modality text if these columns are documents to learn from.",
            )

        self._numeric, self._categorical = numeric, categorical
        self.notes = []
        if dropped:
            self.notes.append(
                f"Dropped {len(dropped)} high-cardinality column(s) as identifiers: "
                + ", ".join(dropped[:5])
                + ("..." if len(dropped) > 5 else "")
            )

        blocks = []
        if numeric:
            blocks.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            blocks.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", self._one_hot_encoder()),
                        ]
                    ),
                    categorical,
                )
            )

        self._transformer = ColumnTransformer(blocks, remainder="drop")
        # Categoricals are cast to str so a column mixing 1 and "1" encodes as one
        # category, and so SimpleImputer's most_frequent strategy does not choke
        # on mixed types.
        matrix = self._transformer.fit_transform(self._prepare_frame(frame))
        matrix = np.asarray(matrix, dtype=np.float32)
        self.extractor = (
            f"column-encoder ({len(numeric)} numeric + {len(categorical)} categorical "
            f"-> {matrix.shape[1]} features)"
        )
        self.feature_names = [f"f{index}" for index in range(matrix.shape[1])]
        self.fitted = True
        return matrix

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for column in self._categorical:
            if column not in prepared.columns:
                prepared[column] = "missing"
            prepared[column] = prepared[column].astype(str)
        for column in self._numeric:
            if column not in prepared.columns:
                prepared[column] = np.nan
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        return prepared[self._numeric + self._categorical]

    def transform(self, inputs: Any) -> np.ndarray:
        if not self.fitted or self._transformer is None:
            raise DLDataError("The tabular adapter must be fitted before transform().")
        frame = self._as_frame(inputs)
        matrix = self._transformer.transform(self._prepare_frame(frame))
        return np.asarray(matrix, dtype=np.float32)

    def coerce(self, data: Any) -> Any:
        return self._as_frame(data)
