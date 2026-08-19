"""NLP Dataset Abstraction & Data Contract - `dive/nlp/data/dataset.py`.

Provides structured, flexible data representation for NLP workloads including
text sequences, optional labels, metadata, sample IDs, language tags, and splits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from dive.nlp.exceptions import TextDataError, NLPConfigError
from dive.nlp.interfaces import NLPDatasetProtocol
from dive.utils.io import load_dataframe, resolve_path


@dataclass
class NLPSample:
    """Represents a single document record in an NLP dataset."""

    text: str
    label: Optional[Any] = None
    sample_id: Optional[str] = None
    language: Optional[str] = None
    split: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"text": self.text}
        if self.label is not None:
            d["label"] = self.label
        if self.sample_id is not None:
            d["sample_id"] = self.sample_id
        if self.language is not None:
            d["language"] = self.language
        if self.split is not None:
            d["split"] = self.split
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class NLPDataset:
    """Core NLP dataset abstraction adhering to NLPDatasetProtocol.

    Encapsulates text documents with optional target labels, sample IDs,
    metadata DataFrame, language attributes, and split assignments.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[Any]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        metadata: Optional[pd.DataFrame] = None,
        languages: Optional[Sequence[str]] = None,
        splits: Optional[Sequence[str]] = None,
        name: str = "nlp_dataset",
    ) -> None:
        self._texts: List[str] = [str(t) if t is not None else "" for t in texts]
        n_samples = len(self._texts)

        if n_samples == 0:
            raise TextDataError("NLPDataset cannot be empty. Provided texts sequence has length 0.")

        # Optional labels
        if labels is not None:
            if len(labels) != n_samples:
                raise TextDataError(
                    f"Length mismatch: {n_samples} texts but {len(labels)} labels.",
                    "Ensure labels array has exactly one element per text document.",
                )
            self._labels: Optional[List[Any]] = list(labels)
        else:
            self._labels = None

        # Optional sample IDs
        if sample_ids is not None:
            if len(sample_ids) != n_samples:
                raise TextDataError(
                    f"Length mismatch: {n_samples} texts but {len(sample_ids)} sample_ids."
                )
            self._sample_ids: Optional[List[str]] = [str(sid) for sid in sample_ids]
        else:
            self._sample_ids = None

        # Optional metadata
        if metadata is not None:
            if len(metadata) != n_samples:
                raise TextDataError(
                    f"Length mismatch: {n_samples} texts but {len(metadata)} metadata rows."
                )
            self._metadata: Optional[pd.DataFrame] = metadata.reset_index(drop=True).copy()
        else:
            self._metadata = None

        # Optional languages
        if languages is not None:
            if len(languages) != n_samples:
                raise TextDataError(
                    f"Length mismatch: {n_samples} texts but {len(languages)} language entries."
                )
            self._languages: Optional[List[str]] = [str(l) for l in languages]
        else:
            self._languages = None

        # Optional split assignments
        if splits is not None:
            if len(splits) != n_samples:
                raise TextDataError(
                    f"Length mismatch: {n_samples} texts but {len(splits)} split tags."
                )
            self._splits: Optional[List[str]] = [str(s) for s in splits]
        else:
            self._splits = None

        self.name = name

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------
    @property
    def texts(self) -> List[str]:
        return self._texts

    @property
    def labels(self) -> Optional[List[Any]]:
        return self._labels

    @property
    def sample_ids(self) -> Optional[List[str]]:
        return self._sample_ids

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        return self._metadata

    @property
    def languages(self) -> Optional[List[str]]:
        return self._languages

    @property
    def splits(self) -> Optional[List[str]]:
        return self._splits

    @property
    def has_labels(self) -> bool:
        return self._labels is not None

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record: Dict[str, Any] = {"text": self._texts[index]}
        if self._labels is not None:
            record["label"] = self._labels[index]
        if self._sample_ids is not None:
            record["sample_id"] = self._sample_ids[index]
        if self._languages is not None:
            record["language"] = self._languages[index]
        if self._splits is not None:
            record["split"] = self._splits[index]
        if self._metadata is not None:
            record["metadata"] = self._metadata.iloc[index].to_dict()
        return record

    def __iter__(self) -> Iterator[NLPSample]:
        for i in range(len(self)):
            yield NLPSample(
                text=self._texts[i],
                label=self._labels[i] if self._labels is not None else None,
                sample_id=self._sample_ids[i] if self._sample_ids is not None else None,
                language=self._languages[i] if self._languages is not None else None,
                split=self._splits[i] if self._splits is not None else None,
                metadata=self._metadata.iloc[i].to_dict() if self._metadata is not None else None,
            )

    # ------------------------------------------------------------------
    # Factory Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_column: Optional[str] = None,
        target_column: Optional[str] = None,
        sample_id_column: Optional[str] = None,
        language_column: Optional[str] = None,
        split_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        drop_na_text: bool = True,
        name: str = "nlp_dataset",
    ) -> "NLPDataset":
        """Construct an NLPDataset from a pandas DataFrame with automatic column resolution."""
        if df.empty:
            raise TextDataError("Supplied DataFrame is empty.")

        frame = df.copy()

        # 1. Resolve text column
        if text_column is None:
            text_column = cls.detect_text_column(frame, exclude=[target_column, sample_id_column, language_column, split_column])
        elif text_column not in frame.columns:
            raise TextDataError(
                f"Specified text column '{text_column}' not found in DataFrame.",
                f"Available columns: {', '.join(map(str, frame.columns[:20]))}",
            )

        # 2. Handle missing texts
        if drop_na_text:
            valid_mask = frame[text_column].notna() & (frame[text_column].astype(str).str.strip() != "")
            if not valid_mask.any():
                raise TextDataError(
                    f"All rows in text column '{text_column}' are empty or null.",
                    "Provide a text column with non-empty string content.",
                )
            frame = frame[valid_mask].reset_index(drop=True)

        texts = frame[text_column].astype(str).tolist()

        # 3. Resolve target column
        labels = None
        if target_column is not None:
            if target_column not in frame.columns:
                raise TextDataError(
                    f"Specified target column '{target_column}' not found in DataFrame.",
                    f"Available columns: {', '.join(map(str, frame.columns[:20]))}",
                )
            labels = frame[target_column].tolist()

        # 4. Resolve sample IDs
        sample_ids = None
        if sample_id_column is not None and sample_id_column in frame.columns:
            sample_ids = frame[sample_id_column].astype(str).tolist()

        # 5. Resolve language column
        languages = None
        if language_column is not None and language_column in frame.columns:
            languages = frame[language_column].astype(str).tolist()

        # 6. Resolve split column
        splits = None
        if split_column is not None and split_column in frame.columns:
            splits = frame[split_column].astype(str).tolist()

        # 7. Resolve metadata columns
        used_cols = {text_column}
        for col in (target_column, sample_id_column, language_column, split_column):
            if col:
                used_cols.add(col)

        if metadata_columns is not None:
            meta_cols = [c for c in metadata_columns if c in frame.columns]
            metadata = frame[meta_cols] if meta_cols else None
        else:
            rem_cols = [c for c in frame.columns if c not in used_cols]
            metadata = frame[rem_cols] if rem_cols else None

        return cls(
            texts=texts,
            labels=labels,
            sample_ids=sample_ids,
            metadata=metadata,
            languages=languages,
            splits=splits,
            name=name,
        )

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        text_column: Optional[str] = None,
        target_column: Optional[str] = None,
        sample_id_column: Optional[str] = None,
        language_column: Optional[str] = None,
        split_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        drop_na_text: bool = True,
    ) -> "NLPDataset":
        """Load and construct an NLPDataset from any supported file format (CSV, JSON, JSONL, Parquet)."""
        path = resolve_path(file_path, must_exist=True)
        df = load_dataframe(str(path))
        return cls.from_dataframe(
            df=df,
            text_column=text_column,
            target_column=target_column,
            sample_id_column=sample_id_column,
            language_column=language_column,
            split_column=split_column,
            metadata_columns=metadata_columns,
            drop_na_text=drop_na_text,
            name=path.stem,
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[Dict[str, Any]],
        text_key: str = "text",
        target_key: Optional[str] = "label",
        sample_id_key: Optional[str] = "sample_id",
        name: str = "nlp_dataset",
    ) -> "NLPDataset":
        """Construct an NLPDataset from a sequence of dictionary records."""
        if not records:
            raise TextDataError("Cannot create NLPDataset from an empty records list.")
        df = pd.DataFrame(records)
        return cls.from_dataframe(
            df=df,
            text_column=text_key,
            target_column=target_key if target_key and target_key in df.columns else None,
            sample_id_column=sample_id_key if sample_id_key and sample_id_key in df.columns else None,
            name=name,
        )

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        labels: Optional[Sequence[Any]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        name: str = "nlp_dataset",
    ) -> "NLPDataset":
        """Construct an NLPDataset directly from a sequence of text strings and optional labels."""
        return cls(texts=texts, labels=labels, sample_ids=sample_ids, name=name)

    # ------------------------------------------------------------------
    # Heuristics & Analysis
    # ------------------------------------------------------------------
    @staticmethod
    def detect_text_column(
        df: pd.DataFrame, exclude: Optional[Sequence[Optional[str]]] = None
    ) -> str:
        """Autonomously detect the most likely text feature column in a DataFrame."""
        excluded = {str(col) for col in (exclude or []) if col is not None}
        candidates: List[Tuple[str, float]] = []

        for col in df.columns:
            if str(col) in excluded:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            # Check if column contains object/string data
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                # Exclude if all values parse as numeric or datetime
                num_check = pd.to_numeric(series.head(100), errors="coerce")
                if num_check.notna().mean() > 0.8:
                    continue

                sample_strs = series.head(200).astype(str)
                avg_char_length = float(sample_strs.str.len().mean())
                avg_word_count = float(sample_strs.str.split().str.len().mean())
                unique_ratio = float(series.nunique() / len(series))

                # Text columns typically have multi-word strings and high uniqueness
                score = (avg_char_length * 0.5) + (avg_word_count * 5.0) + (unique_ratio * 10.0)
                candidates.append((str(col), score))

        if not candidates:
            # Fallback to the first string/object column
            obj_cols = [str(c) for c in df.columns if str(c) not in excluded and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
            if obj_cols:
                return obj_cols[0]
            raise TextDataError(
                "Could not automatically detect a valid text column in the dataset.",
                "Explicitly specify `text_column='your_column_name'`.",
            )

        # Sort candidates by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # ------------------------------------------------------------------
    # Splitting & Serialization
    # ------------------------------------------------------------------
    def split(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Union[Tuple["NLPDataset", "NLPDataset"], Tuple["NLPDataset", "NLPDataset", "NLPDataset"]]:
        """Deterministically split the dataset into train/test or train/val/test subsets.

        Parameters
        ----------
        test_size : float
            Proportion of the dataset to allocate to the test split (default: 0.2).
        val_size : Optional[float]
            Optional proportion to allocate to a validation split.
        stratify : bool
            Whether to preserve target label distribution across splits (if labels exist).
        random_state : int
            Random seed for reproducibility.
        """
        n_samples = len(self)
        if n_samples < 2:
            raise TextDataError(f"Cannot split a dataset with only {n_samples} sample(s).")

        use_stratify = stratify and self._labels is not None and len(set(self._labels)) > 1

        if val_size is not None:
            # Three-way split: Train, Validation, Test
            total_holdout = test_size + val_size
            if total_holdout >= 1.0 or total_holdout <= 0.0:
                raise NLPConfigError("test_size + val_size must be between 0.0 and 1.0")

            if use_stratify:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=total_holdout, random_state=random_state)
                train_idx, holdout_idx = next(splitter.split(self._texts, self._labels))
            else:
                splitter = ShuffleSplit(n_splits=1, test_size=total_holdout, random_state=random_state)
                train_idx, holdout_idx = next(splitter.split(self._texts))

            # Split holdout into val and test
            rel_val_size = val_size / total_holdout
            holdout_texts = [self._texts[i] for i in holdout_idx]
            holdout_labels = [self._labels[i] for i in holdout_idx] if self._labels else None

            if use_stratify and holdout_labels and len(set(holdout_labels)) > 1:
                sub_splitter = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - rel_val_size), random_state=random_state)
                val_sub_idx, test_sub_idx = next(sub_splitter.split(holdout_texts, holdout_labels))
            else:
                sub_splitter = ShuffleSplit(n_splits=1, test_size=(1.0 - rel_val_size), random_state=random_state)
                val_sub_idx, test_sub_idx = next(sub_splitter.split(holdout_texts))

            val_idx = [holdout_idx[i] for i in val_sub_idx]
            test_idx = [holdout_idx[i] for i in test_sub_idx]

            train_ds = self.subset(train_idx, name=f"{self.name}_train")
            val_ds = self.subset(val_idx, name=f"{self.name}_val")
            test_ds = self.subset(test_idx, name=f"{self.name}_test")
            return train_ds, val_ds, test_ds

        # Two-way split: Train, Test
        if use_stratify:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(self._texts, self._labels))
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(self._texts))

        train_ds = self.subset(train_idx, name=f"{self.name}_train")
        test_ds = self.subset(test_idx, name=f"{self.name}_test")
        return train_ds, test_ds

    def subset(self, indices: Sequence[int], name: Optional[str] = None) -> "NLPDataset":
        """Return a new NLPDataset containing only the specified index subset."""
        idx = list(indices)
        sub_texts = [self._texts[i] for i in idx]
        sub_labels = [self._labels[i] for i in idx] if self._labels is not None else None
        sub_ids = [self._sample_ids[i] for i in idx] if self._sample_ids is not None else None
        sub_meta = self._metadata.iloc[idx].reset_index(drop=True) if self._metadata is not None else None
        sub_langs = [self._languages[i] for i in idx] if self._languages is not None else None
        sub_splits = [self._splits[i] for i in idx] if self._splits is not None else None

        return NLPDataset(
            texts=sub_texts,
            labels=sub_labels,
            sample_ids=sub_ids,
            metadata=sub_meta,
            languages=sub_langs,
            splits=sub_splits,
            name=name or self.name,
        )

    def summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the dataset."""
        char_lens = [len(t) for t in self._texts]
        word_counts = [len(t.split()) for t in self._texts]

        stats: Dict[str, Any] = {
            "name": self.name,
            "n_samples": len(self._texts),
            "has_labels": self.has_labels,
            "avg_char_length": round(float(np.mean(char_lens)), 2) if char_lens else 0.0,
            "median_char_length": int(np.median(char_lens)) if char_lens else 0,
            "avg_word_count": round(float(np.mean(word_counts)), 2) if word_counts else 0.0,
            "median_word_count": int(np.median(word_counts)) if word_counts else 0,
            "has_metadata": self._metadata is not None,
            "metadata_columns": list(self._metadata.columns) if self._metadata is not None else [],
        }

        if self._labels is not None:
            unique_labels, counts = np.unique(self._labels, return_counts=True)
            stats["n_classes"] = len(unique_labels)
            stats["label_distribution"] = {str(k): int(v) for k, v in zip(unique_labels, counts)}

        return stats

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the NLPDataset to a pandas DataFrame."""
        data: Dict[str, Any] = {"text": self._texts}
        if self._labels is not None:
            data["label"] = self._labels
        if self._sample_ids is not None:
            data["sample_id"] = self._sample_ids
        if self._languages is not None:
            data["language"] = self._languages
        if self._splits is not None:
            data["split"] = self._splits

        df = pd.DataFrame(data)
        if self._metadata is not None:
            for col in self._metadata.columns:
                if col not in df.columns:
                    df[col] = self._metadata[col].values
        return df
