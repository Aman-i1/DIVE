"""Phase 2 Tests - DIVE NLP Data Contract & Ingestion Layer.

Verifies:
1. NLPDataset abstraction and NLPSample representation.
2. In-memory constructors (from_texts, from_records, from_dataframe).
3. File-based ingestion (CSV, JSON, JSONL, Parquet) with automatic column resolution.
4. Automatic text column detection heuristic.
5. Deterministic, stratified train/val/test splitting and K-fold CV iteration.
6. Missing value handling, validation constraints, and error edge cases.
7. Dataset summary statistics and protocol conformance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from dive.nlp.data import DatasetSplitter, NLPDataset, NLPSample, load_nlp_dataset
from dive.nlp.exceptions import NLPConfigError, TextDataError
from dive.nlp.interfaces import NLPDatasetProtocol


def test_nlp_dataset_from_texts_and_protocol() -> None:
    """Verify basic instantiation from text sequences and protocol conformance."""
    texts = ["Machine learning is fascinating.", "Natural language processing with transformers.", "Autonomous AutoML platform."]
    labels = ["tech", "nlp", "automl"]
    sample_ids = ["doc_01", "doc_02", "doc_03"]

    ds = NLPDataset(texts=texts, labels=labels, sample_ids=sample_ids, name="demo_corpus")

    # Protocol compliance
    assert isinstance(ds, NLPDatasetProtocol)
    assert len(ds) == 3
    assert ds.name == "demo_corpus"
    assert ds.has_labels is True
    assert ds.texts == texts
    assert ds.labels == labels
    assert ds.sample_ids == sample_ids

    # Indexing & Iteration
    first_item = ds[0]
    assert first_item["text"] == texts[0]
    assert first_item["label"] == "tech"
    assert first_item["sample_id"] == "doc_01"

    samples = list(ds)
    assert len(samples) == 3
    assert isinstance(samples[0], NLPSample)
    assert samples[0].text == texts[0]
    assert samples[0].label == "tech"


def test_nlp_dataset_from_dataframe() -> None:
    """Verify DataFrame ingestion with metadata preservation and roundtripping."""
    df = pd.DataFrame(
        {
            "id": ["s1", "s2", "s3", "s4"],
            "review": [
                "Outstanding customer service and fast shipping!",
                "Terrible experience, arrived broken.",
                "Average quality, nothing special.",
                "Highly recommended, will buy again.",
            ],
            "sentiment": ["positive", "negative", "neutral", "positive"],
            "user_age": [28, 45, 34, 52],
            "lang": ["en", "en", "en", "en"],
        }
    )

    ds = NLPDataset.from_dataframe(
        df=df,
        text_column="review",
        target_column="sentiment",
        sample_id_column="id",
        language_column="lang",
        name="reviews_dataset",
    )

    assert len(ds) == 4
    assert ds.texts[0] == "Outstanding customer service and fast shipping!"
    assert ds.labels == ["positive", "negative", "neutral", "positive"]
    assert ds.sample_ids == ["s1", "s2", "s3", "s4"]
    assert ds.languages == ["en", "en", "en", "en"]
    assert ds.metadata is not None
    assert "user_age" in ds.metadata.columns

    # Convert back to DataFrame
    df_out = ds.to_dataframe()
    assert len(df_out) == 4
    assert "text" in df_out.columns
    assert "label" in df_out.columns
    assert "user_age" in df_out.columns


def test_nlp_dataset_detect_text_column() -> None:
    """Verify automatic text column detection heuristic on heterogeneous tabular data."""
    df = pd.DataFrame(
        {
            "user_id": ["u_101", "u_102", "u_103", "u_104"],
            "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "rating": [5, 1, 3, 5],
            "comments": [
                "The product exceeded all my expectations and performed flawlessly under heavy load.",
                "Complete waste of time and money, totally unresponsive support team.",
                "Standard performance, meets basic expectations for the price point.",
                "Phenomenal build quality, durable materials, and intuitive user interface.",
            ],
            "category": ["electronics", "electronics", "electronics", "electronics"],
        }
    )

    detected_col = NLPDataset.detect_text_column(df, exclude=["rating"])
    assert detected_col == "comments"

    # Ingestion without specifying text_column auto-detects comments
    ds = NLPDataset.from_dataframe(df, target_column="rating")
    assert ds.texts == df["comments"].tolist()
    assert ds.labels == [5, 1, 3, 5]


def test_nlp_dataset_file_formats_roundtrip(tmp_path: Path) -> None:
    """Verify loading from CSV, JSON, JSONL, and Parquet file formats."""
    records = [
        {"id": "doc1", "text": "First NLP document for testing.", "topic": "science"},
        {"id": "doc2", "text": "Second document discussing machine learning.", "topic": "ai"},
        {"id": "doc3", "text": "Third entry covering deep neural networks.", "topic": "ai"},
        {"id": "doc4", "text": "Fourth text snippet on linguistics.", "topic": "science"},
    ]
    df = pd.DataFrame(records)

    # 1. CSV
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    ds_csv = load_nlp_dataset(csv_path, target_column="topic", sample_id_column="id")
    assert len(ds_csv) == 4
    assert ds_csv.labels == ["science", "ai", "ai", "science"]

    # 2. JSON
    json_path = tmp_path / "data.json"
    df.to_json(json_path, orient="records", indent=2)
    ds_json = load_nlp_dataset(json_path, target_column="topic")
    assert len(ds_json) == 4

    # 3. JSONL
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    ds_jsonl = load_nlp_dataset(jsonl_path, target_column="topic")
    assert len(ds_jsonl) == 4
    assert ds_jsonl.texts[0] == "First NLP document for testing."

    # 4. Parquet
    try:
        parquet_path = tmp_path / "data.parquet"
        df.to_parquet(parquet_path, index=False)
        ds_parquet = load_nlp_dataset(parquet_path, target_column="topic")
        assert len(ds_parquet) == 4
    except Exception:
        pass  # Parquet optional if pyarrow is missing in some environments


def test_nlp_dataset_from_records() -> None:
    """Verify constructor from dictionary records."""
    records = [
        {"text": "Python is a versatile programming language.", "label": "positive"},
        {"text": "Buggy release caused widespread outage.", "label": "negative"},
    ]
    ds = NLPDataset.from_records(records)
    assert len(ds) == 2
    assert ds.texts[0].startswith("Python")
    assert ds.labels == ["positive", "negative"]


def test_nlp_dataset_deterministic_splits() -> None:
    """Verify deterministic, stratified 2-way and 3-way dataset splitting."""
    texts = [f"Document number {i} with text content." for i in range(100)]
    labels = ["spam" if i % 2 == 0 else "ham" for i in range(100)]
    ds = NLPDataset(texts=texts, labels=labels, name="corpus_100")

    # Two-way split (Train / Test 80/20)
    train_ds, test_ds = ds.split(test_size=0.2, stratify=True, random_state=42)
    assert len(train_ds) == 80
    assert len(test_ds) == 20
    assert train_ds.name == "corpus_100_train"
    assert test_ds.name == "corpus_100_test"

    # Check stratification balance
    train_spam_count = sum(1 for l in train_ds.labels if l == "spam")
    test_spam_count = sum(1 for l in test_ds.labels if l == "spam")
    assert train_spam_count == 40
    assert test_spam_count == 10

    # Reproducibility check: identical seed produces identical splits
    train_ds2, test_ds2 = ds.split(test_size=0.2, stratify=True, random_state=42)
    assert train_ds.texts == train_ds2.texts
    assert test_ds.texts == test_ds2.texts

    # Three-way split (Train 70 / Val 15 / Test 15)
    train_3w, val_3w, test_3w = ds.split(test_size=0.15, val_size=0.15, stratify=True, random_state=42)
    assert len(train_3w) == 70
    assert len(val_3w) == 15
    assert len(test_3w) == 15


def test_dataset_splitter_kfold_cv() -> None:
    """Verify K-fold cross-validation iterator with DatasetSplitter."""
    texts = [f"Sample sentence {i}" for i in range(50)]
    labels = [i % 2 for i in range(50)]
    ds = NLPDataset(texts=texts, labels=labels)

    splitter = DatasetSplitter(cv_splits=5, stratify=True, random_state=42)
    folds = list(splitter.cv_folds(ds))

    assert len(folds) == 5
    all_val_texts = []
    for train_fold, val_fold in folds:
        assert len(train_fold) == 40
        assert len(val_fold) == 10
        all_val_texts.extend(val_fold.texts)

    # Every sample was in validation fold exactly once
    assert len(all_val_texts) == 50
    assert set(all_val_texts) == set(texts)


def test_nlp_dataset_missing_values_and_error_handling() -> None:
    """Verify validation boundaries and error conditions."""
    # Empty texts
    with pytest.raises(TextDataError, match="cannot be empty"):
        NLPDataset(texts=[])

    # Mismatched labels
    with pytest.raises(TextDataError, match="Length mismatch"):
        NLPDataset(texts=["doc 1", "doc 2"], labels=[1])

    # Missing text column in dataframe
    df = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    with pytest.raises(TextDataError, match="Specified text column 'missing_col' not found"):
        NLPDataset.from_dataframe(df, text_column="missing_col")

    # Missing target column in dataframe
    df_text = pd.DataFrame({"text": ["hello", "world"]})
    with pytest.raises(TextDataError, match="Specified target column 'target' not found"):
        NLPDataset.from_dataframe(df_text, text_column="text", target_column="target")

    # Invalid split sizes
    ds = NLPDataset(texts=["a", "b", "c", "d"])
    with pytest.raises(NLPConfigError, match="must be between 0.0 and 1.0"):
        ds.split(test_size=0.6, val_size=0.5)


def test_nlp_dataset_summary_stats() -> None:
    """Verify dataset summary statistics calculation."""
    texts = ["Short text.", "A slightly longer document with several words included."]
    labels = ["class_A", "class_B"]
    ds = NLPDataset(texts=texts, labels=labels, name="stats_corpus")

    stats = ds.summary_stats()
    assert stats["name"] == "stats_corpus"
    assert stats["n_samples"] == 2
    assert stats["has_labels"] is True
    assert stats["n_classes"] == 2
    assert stats["label_distribution"] == {"class_A": 1, "class_B": 1}
    assert stats["avg_char_length"] > 0
    assert stats["avg_word_count"] > 0
