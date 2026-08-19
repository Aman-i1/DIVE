"""Phase 3 Tests - DIVE NLP Dataset Profiling & Diagnostic Engine.

Verifies:
1. Document volume, character and token length distributions (percentiles, min/max/mean/std).
2. Detection of empty, whitespace-only, and duplicate documents with duplicate ratios.
3. Vocabulary size, type-token ratio (TTR), and top token frequency extraction.
4. Target label distribution, multi-class metrics, and imbalance warnings.
5. Contamination / label leakage auditing (identical texts with conflicting labels).
6. ASCII report rendering and dictionary serialization.
7. Profile input polymorphism (NLPDataset, list of strings, pandas DataFrame).
8. NLPProfilerProtocol compliance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dive.nlp.data import NLPDataset
from dive.nlp.exceptions import TextDataError
from dive.nlp.interfaces import NLPProfilerProtocol
from dive.nlp.profiling import NLPProfileReport, NLPProfiler, profile_nlp_dataset


def test_nlp_profiler_protocol_and_polymorphism() -> None:
    """Verify profiler protocol compliance and multiple input formats."""
    profiler = NLPProfiler()
    assert isinstance(profiler, NLPProfilerProtocol)

    texts = [
        "Machine learning models require clean text data.",
        "Deep learning architectures process sequential token streams.",
        "AutoML platforms automate algorithm and hyperparameter selection.",
    ]

    # 1. From list of texts
    rep1 = profiler.profile(texts, name="texts_list")
    assert isinstance(rep1, NLPProfileReport)
    assert rep1.n_samples == 3
    assert rep1.name == "texts_list"

    # 2. From NLPDataset
    ds = NLPDataset(texts=texts, name="custom_corpus")
    rep2 = profiler.profile(ds)
    assert rep2.n_samples == 3
    assert rep2.name == "custom_corpus"

    # 3. From pandas DataFrame
    df = pd.DataFrame({"body": texts, "target": [1, 0, 1]})
    rep3 = profile_nlp_dataset(df, text_column="body", target_column="target")
    assert rep3.n_samples == 3
    assert rep3.has_labels is True

    # 4. Empty input error
    with pytest.raises(TextDataError, match="empty"):
        profiler.profile([])


def test_nlp_profiler_length_and_vocabulary_statistics() -> None:
    """Verify exact character and token length distributions and vocabulary metrics."""
    texts = [
        "apple banana cherry",        # 3 words, 19 chars
        "apple banana",               # 2 words, 12 chars
        "apple cherry date elderberry", # 4 words, 28 chars
    ]
    report = profile_nlp_dataset(texts)

    # Length stats
    assert report.n_samples == 3
    assert report.n_empty == 0
    assert report.n_duplicates == 0
    assert report.char_stats["min"] == 12.0
    assert report.char_stats["max"] == 28.0
    assert report.token_stats["min"] == 2.0
    assert report.token_stats["max"] == 4.0
    assert report.token_stats["p50"] == 3.0

    # Vocabulary stats (apple, banana, cherry, date, elderberry -> 5 unique tokens)
    assert report.vocabulary_size == 5
    assert report.lexical_diversity == 5 / 9  # 5 unique / 9 total tokens

    top_tokens = dict(report.top_tokens)
    assert top_tokens["apple"] == 3
    assert top_tokens["banana"] == 2
    assert top_tokens["cherry"] == 2


def test_nlp_profiler_empty_whitespace_and_duplicates() -> None:
    """Verify detection of empty documents, whitespace-only documents, and duplicates."""
    texts = [
        "Unique document one.",
        "",                         # Empty
        "   \t\n  ",               # Whitespace only
        "Duplicate document.",
        "Duplicate document.",     # Duplicate
        "Duplicate document.",     # Duplicate
    ]
    report = profile_nlp_dataset(texts)

    assert report.n_samples == 6
    assert report.n_empty == 1
    assert report.n_whitespace_only == 1
    assert report.n_duplicates == 2  # 3 copies = 2 duplicates
    assert report.duplicate_ratio == 2 / 6

    # Warnings generated for duplicates and empty documents
    warning_text = " ".join(report.warnings)
    assert "High document duplication" in warning_text
    assert "empty or whitespace-only" in warning_text


def test_nlp_profiler_label_distribution_and_imbalance() -> None:
    """Verify multi-class label distribution and severe imbalance detection."""
    # Highly imbalanced dataset: 10 'spam' vs 1 'ham'
    texts = [f"Spam text document {i}" for i in range(10)] + ["Legitimate user communication."]
    labels = ["spam"] * 10 + ["ham"]
    ds = NLPDataset(texts=texts, labels=labels)

    report = profile_nlp_dataset(ds, imbalance_threshold=3.0)

    assert report.has_labels is True
    assert report.label_stats is not None
    assert report.label_stats["n_classes"] == 2
    assert report.label_stats["class_counts"] == {"spam": 10, "ham": 1}
    assert report.label_stats["imbalance_ratio"] == 10.0
    assert report.label_stats["is_imbalanced"] is True

    # Imbalance warning present
    assert any("Target class imbalance detected" in w for w in report.warnings)


def test_nlp_profiler_leakage_and_label_contamination() -> None:
    """Verify detection of label contamination / leakage (identical text with conflicting labels)."""
    texts = [
        "The battery life is exceptional.",
        "The battery life is exceptional.",  # Exactly identical text but conflicting label
        "Terrible customer service.",
    ]
    labels = [
        "positive",
        "negative",  # Conflict!
        "negative",
    ]
    ds = NLPDataset(texts=texts, labels=labels)

    report = profile_nlp_dataset(ds)

    assert len(report.leakage_risks) > 0
    leakage = report.leakage_risks[0]
    assert leakage["issue"] == "Conflicting Labels on Identical Text"
    assert leakage["count"] == 1
    assert any("Label contamination detected" in w for w in report.warnings)


def test_nlp_profiler_rendering_and_serialization() -> None:
    """Verify dictionary serialization and ASCII text rendering."""
    texts = ["First test sentence.", "Second test sentence with more details."]
    labels = ["A", "B"]
    ds = NLPDataset(texts=texts, labels=labels, name="render_test_ds")

    report = profile_nlp_dataset(ds)

    # 1. to_dict()
    d = report.to_dict()
    assert d["name"] == "render_test_ds"
    assert d["n_samples"] == 2
    assert "char_stats" in d
    assert "token_stats" in d
    assert "vocabulary_size" in d

    # 2. render()
    rendered = report.render()
    assert "DIVE NLP DATASET PROFILE REPORT: RENDER_TEST_DS" in rendered
    assert "Total Documents       : 2" in rendered
    assert "DOCUMENT LENGTH DISTRIBUTION" in rendered
    assert "TARGET LABEL DISTRIBUTION" in rendered
