"""NLP Dataset Profiler & Diagnostic Engine - `dive/nlp/profiling/profiler.py`.

Provides deterministic, high-efficiency statistical auditing of text datasets:
- Document volume & character/token length distributions (p50, p95, p99, min, max, std)
- Empty, whitespace-only, and duplicate document auditing
- Vocabulary size, lexical diversity (TTR), and top token frequencies
- Target label distribution, multi-class balance, and severe imbalance warnings
- Potential text leakage & label contamination (identical texts with conflicting labels)
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from dive.nlp.data.dataset import NLPDataset
from dive.nlp.exceptions import TextDataError
from dive.nlp.interfaces import NLPProfilerProtocol


# Simple, fast word tokenization regex (Unicode alphanumeric words)
_TOKEN_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class NLPProfileReport:
    """Diagnostic profile results for an NLP dataset."""

    name: str
    n_samples: int
    n_empty: int
    n_whitespace_only: int
    n_duplicates: int
    duplicate_ratio: float
    char_stats: Dict[str, float]
    token_stats: Dict[str, float]
    vocabulary_size: int
    lexical_diversity: float
    top_tokens: List[Tuple[str, int]]
    has_labels: bool
    label_stats: Optional[Dict[str, Any]] = None
    language_distribution: Optional[Dict[str, int]] = None
    leakage_risks: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_empty": self.n_empty,
            "n_whitespace_only": self.n_whitespace_only,
            "n_duplicates": self.n_duplicates,
            "duplicate_ratio": round(self.duplicate_ratio, 4),
            "char_stats": self.char_stats,
            "token_stats": self.token_stats,
            "vocabulary_size": self.vocabulary_size,
            "lexical_diversity": round(self.lexical_diversity, 4),
            "top_tokens": self.top_tokens,
            "has_labels": self.has_labels,
            "label_stats": self.label_stats,
            "language_distribution": self.language_distribution,
            "leakage_risks": self.leakage_risks,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        """Render a clean, human-readable ASCII diagnostic report."""
        lines = [
            "=" * 60,
            f"DIVE NLP DATASET PROFILE REPORT: {self.name.upper()}",
            "=" * 60,
            f"Total Documents       : {self.n_samples:,}",
            f"Empty / Whitespace    : {self.n_empty + self.n_whitespace_only} ({((self.n_empty + self.n_whitespace_only) / max(1, self.n_samples)) * 100:.1f}%)",
            f"Duplicate Documents   : {self.n_duplicates} ({self.duplicate_ratio * 100:.1f}%)",
            f"Vocabulary Size       : {self.vocabulary_size:,} unique tokens (TTR: {self.lexical_diversity:.3f})",
            "",
            "DOCUMENT LENGTH DISTRIBUTION",
            "----------------------------",
            f"Character Count  : min={self.char_stats.get('min', 0):.0f}, p50={self.char_stats.get('p50', 0):.0f}, p95={self.char_stats.get('p95', 0):.0f}, max={self.char_stats.get('max', 0):.0f}, mean={self.char_stats.get('mean', 0):.1f}",
            f"Token/Word Count : min={self.token_stats.get('min', 0):.0f}, p50={self.token_stats.get('p50', 0):.0f}, p95={self.token_stats.get('p95', 0):.0f}, max={self.token_stats.get('max', 0):.0f}, mean={self.token_stats.get('mean', 0):.1f}",
        ]

        if self.has_labels and self.label_stats:
            lines.extend(
                [
                    "",
                    "TARGET LABEL DISTRIBUTION",
                    "-------------------------",
                    f"Number of Classes: {self.label_stats.get('n_classes')}",
                    f"Imbalance Ratio  : {self.label_stats.get('imbalance_ratio', 1.0):.1f}:1 ({'IMBALANCED' if self.label_stats.get('is_imbalanced') else 'BALANCED'})",
                ]
            )
            for cls_name, count in self.label_stats.get("class_counts", {}).items():
                pct = (count / max(1, self.n_samples)) * 100
                lines.append(f"  * {cls_name:<20}: {count:>6,} ({pct:>5.1f}%)")

        if self.leakage_risks:
            lines.extend(
                [
                    "",
                    "LEAKAGE & CONTAMINATION RISKS",
                    "-----------------------------",
                ]
            )
            for lk in self.leakage_risks:
                lines.append(f"  [!] {lk.get('issue')}: {lk.get('description')}")

        if self.warnings:
            lines.extend(
                [
                    "",
                    "DATASET WARNINGS & RECOMMENDATIONS",
                    "----------------------------------",
                ]
            )
            for w in self.warnings:
                lines.append(f"  * {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


class NLPProfiler:
    """Autonomous profiler and statistical auditor for NLP datasets."""

    def __init__(self, imbalance_threshold: float = 3.0, top_k_tokens: int = 20) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.top_k_tokens = top_k_tokens

    def profile(
        self,
        dataset: Union[NLPDataset, Sequence[str], pd.DataFrame],
        text_column: Optional[str] = None,
        target_column: Optional[str] = None,
        name: Optional[str] = None,
    ) -> NLPProfileReport:
        """Execute comprehensive statistical profiling on the dataset."""
        # Standardize input into an NLPDataset
        if isinstance(dataset, NLPDataset):
            ds = dataset
        elif isinstance(dataset, pd.DataFrame):
            ds = NLPDataset.from_dataframe(
                df=dataset,
                text_column=text_column,
                target_column=target_column,
                name=name or "dataframe",
            )
        elif isinstance(dataset, Sequence):
            ds = NLPDataset.from_texts(texts=dataset, name=name or "text_sequence")
        else:
            raise TextDataError(f"Unsupported dataset type for profiling: {type(dataset).__name__}")

        n_samples = len(ds)
        if n_samples == 0:
            raise TextDataError("Cannot profile an empty dataset.")

        texts = ds.texts
        labels = ds.labels
        warnings: List[str] = []

        # 1. Document emptiness & whitespace
        n_empty = 0
        n_whitespace_only = 0
        char_lengths: List[int] = []
        token_lengths: List[int] = []
        token_counter: collections.Counter = collections.Counter()
        total_tokens = 0
        text_counter: collections.Counter = collections.Counter()

        for t in texts:
            if not t:
                n_empty += 1
                char_lengths.append(0)
                token_lengths.append(0)
                continue

            striped = t.strip()
            if not striped:
                n_whitespace_only += 1
                char_lengths.append(len(t))
                token_lengths.append(0)
                continue

            text_counter[striped] += 1
            char_lengths.append(len(t))

            # Tokenize words for vocabulary and token count
            doc_tokens = _TOKEN_REGEX.findall(striped.lower())
            doc_token_count = len(doc_tokens)
            token_lengths.append(doc_token_count)
            total_tokens += doc_token_count
            token_counter.update(doc_tokens)

        # 2. Duplicates
        n_unique_texts = len(text_counter)
        n_duplicates = sum(count - 1 for count in text_counter.values() if count > 1)
        duplicate_ratio = float(n_duplicates / n_samples)

        if duplicate_ratio > 0.10:
            warnings.append(
                f"High document duplication rate ({duplicate_ratio * 100:.1f}% duplicates). "
                "Consider deduplicating text before training."
            )

        if (n_empty + n_whitespace_only) > 0:
            warnings.append(
                f"Found {n_empty + n_whitespace_only} empty or whitespace-only document(s)."
            )

        # 3. Statistical length metrics
        char_stats = self._calc_distribution_stats(char_lengths)
        token_stats = self._calc_distribution_stats(token_lengths)

        if char_stats["max"] > 10_000:
            warnings.append(
                f"Extremely long document detected (max={char_stats['max']:.0f} chars). "
                "Ensure maximum sequence length truncation is configured."
            )

        # 4. Lexical diversity
        vocab_size = len(token_counter)
        lexical_diversity = float(vocab_size / total_tokens) if total_tokens > 0 else 0.0
        top_tokens = token_counter.most_common(self.top_k_tokens)

        # 5. Target label statistics
        has_labels = labels is not None
        label_stats: Optional[Dict[str, Any]] = None
        leakage_risks: List[Dict[str, Any]] = []

        if has_labels and labels:
            label_counts = collections.Counter(labels)
            n_classes = len(label_counts)
            counts = list(label_counts.values())
            max_c = max(counts)
            min_c = min(counts)
            imbalance_ratio = float(max_c / max(1, min_c))
            is_imbalanced = bool(imbalance_ratio >= self.imbalance_threshold)

            label_stats = {
                "n_classes": n_classes,
                "class_counts": {str(k): int(v) for k, v in label_counts.items()},
                "class_proportions": {
                    str(k): round(float(v / n_samples), 4) for k, v in label_counts.items()
                },
                "imbalance_ratio": round(imbalance_ratio, 2),
                "is_imbalanced": is_imbalanced,
            }

            if is_imbalanced:
                warnings.append(
                    f"Target class imbalance detected ({imbalance_ratio:.1f}:1 ratio). "
                    "Use balanced weighting or stratified sampling."
                )

            # 6. Leakage & Contamination Check (Conflicting labels on identical texts)
            text_to_labels: Dict[str, set] = collections.defaultdict(set)
            for t, l in zip(texts, labels):
                st = t.strip()
                if st:
                    text_to_labels[st].add(str(l))

            conflicts = {t: lbls for t, lbls in text_to_labels.items() if len(lbls) > 1}
            if conflicts:
                leakage_risks.append(
                    {
                        "issue": "Conflicting Labels on Identical Text",
                        "count": len(conflicts),
                        "description": f"{len(conflicts)} identical text document(s) have conflicting target labels.",
                        "examples": [
                            {"text": t[:80] + "...", "labels": list(lbls)}
                            for t, lbls in list(conflicts.items())[:3]
                        ],
                    }
                )
                warnings.append(
                    f"Label contamination detected: {len(conflicts)} exact text(s) have multiple contradictory labels."
                )

        # 7. Language distribution if present
        language_dist = None
        if ds.languages is not None:
            language_dist = dict(collections.Counter(ds.languages))

        return NLPProfileReport(
            name=ds.name,
            n_samples=n_samples,
            n_empty=n_empty,
            n_whitespace_only=n_whitespace_only,
            n_duplicates=n_duplicates,
            duplicate_ratio=duplicate_ratio,
            char_stats=char_stats,
            token_stats=token_stats,
            vocabulary_size=vocab_size,
            lexical_diversity=lexical_diversity,
            top_tokens=top_tokens,
            has_labels=has_labels,
            label_stats=label_stats,
            language_distribution=language_dist,
            leakage_risks=leakage_risks,
            warnings=warnings,
        )

    @staticmethod
    def _calc_distribution_stats(values: List[int]) -> Dict[str, float]:
        """Compute comprehensive percentiles and statistical moments."""
        if not values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        arr = np.array(values, dtype=np.float64)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }
