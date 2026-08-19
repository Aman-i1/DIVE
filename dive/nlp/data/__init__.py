"""DIVE NLP Data Contract and Ingestion Layer - `dive/nlp/data`.

Provides universal text dataset loading, schema resolution, and deterministic splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from dive.nlp.data.dataset import NLPDataset, NLPSample
from dive.nlp.data.splitter import DatasetSplitter


def load_nlp_dataset(
    source: Union[str, Path, Sequence[Dict[str, Any]], Sequence[str]],
    text_column: Optional[str] = None,
    target_column: Optional[str] = None,
    sample_id_column: Optional[str] = None,
    language_column: Optional[str] = None,
    split_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None,
    labels: Optional[Sequence[Any]] = None,
    drop_na_text: bool = True,
) -> NLPDataset:
    """Universal loader constructing an NLPDataset from files, records, or text sequences."""
    if isinstance(source, (str, Path)):
        return NLPDataset.from_file(
            file_path=source,
            text_column=text_column,
            target_column=target_column,
            sample_id_column=sample_id_column,
            language_column=language_column,
            split_column=split_column,
            metadata_columns=metadata_columns,
            drop_na_text=drop_na_text,
        )
    elif isinstance(source, Sequence) and len(source) > 0 and isinstance(source[0], dict):
        return NLPDataset.from_records(
            records=source,  # type: ignore
            text_key=text_column or "text",
            target_key=target_column,
            sample_id_key=sample_id_column,
        )
    elif isinstance(source, Sequence):
        return NLPDataset.from_texts(
            texts=source,  # type: ignore
            labels=labels,
        )
    raise ValueError(f"Unsupported data source type: {type(source).__name__}")


__all__ = [
    "NLPDataset",
    "NLPSample",
    "DatasetSplitter",
    "load_nlp_dataset",
]
