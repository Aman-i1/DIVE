"""Stage 1 - dataset profiling.

Ported from the ``DataIntelligence`` class in ``Automatic Machine Learning.ipynb``.
The detection rules are unchanged; what is new is that the profile also records a
few target-health facts (near-constant target, unique ratio) that the validation
engine surfaces as warnings rather than leaving them as unused metadata.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dive.exceptions import TargetError

# A class-frequency ratio above this is treated as imbalanced.
IMBALANCE_THRESHOLD = 3.0
# Object columns with more distinct values than this are "high cardinality".
HIGH_CARDINALITY_THRESHOLD = 20
# Fraction of values that must parse as dates before a column is treated as one.
DATETIME_PARSE_THRESHOLD = 0.8
# Column names that mark an identifier regardless of dtype.
_ID_NAMES = frozenset({"id", "index", "row_id", "rowid", "uuid", "guid", "key"})
_ID_SUFFIXES = ("_id", "_uuid", "_guid", "_key", "_no", "_number")


def looks_like_datetime(series: pd.Series, sample_size: int = 1000) -> bool:
    """True when most values in an object column parse as dates.

    Judged on a bounded sample so a large text column costs little, and with
    warnings silenced because pandas is noisy about ambiguous formats.
    """
    if series.dtype != object and not pd.api.types.is_string_dtype(series):
        return pd.api.types.is_datetime64_any_dtype(series)
    sample = series.dropna()
    if sample.empty:
        return False
    if len(sample) > sample_size:
        sample = sample.head(sample_size)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
    except Exception:
        return False
    return bool(parsed.notna().mean() > DATETIME_PARSE_THRESHOLD)


class DataIntelligence:
    """Profile a dataset before any modelling decisions are made.

    Detects problem type, class imbalance, missing values, high-cardinality
    columns, constant columns, and ID-like columns.
    """

    def __init__(self, target: str, random_state: int = 42) -> None:
        self.target = target
        self.random_state = random_state
        self.profile_: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return a profile dict describing ``df`` relative to the target."""
        if self.target not in df.columns:
            raise TargetError(
                f"Target column '{self.target}' is not present in the data.",
                f"Columns available: {', '.join(map(str, df.columns[:20]))}",
            )

        y = df[self.target]
        X = df.drop(columns=[self.target])
        n, p = X.shape

        problem_type = self._detect_problem(y)
        imbalance_ratio = (
            self._detect_imbalance(y) if problem_type == "classification" else None
        )

        # Datetime columns are identified before ID detection. A daily date
        # column has one distinct value per row, so the ID heuristic would
        # otherwise claim it and drop it before it can be expanded into
        # calendar features - silently discarding real signal.
        datetime_cols = [
            col
            for col in X.columns
            if pd.api.types.is_datetime64_any_dtype(X[col])
            or (X[col].dtype == object and looks_like_datetime(X[col]))
        ]

        # Build fine-grained semantic types dict
        semantic_types: Dict[str, List[str]] = {}
        for c in X.columns:
            tags: List[str] = []
            if c in datetime_cols:
                tags.append("datetime")
            elif c in self._detect_id_cols(X, n, exclude=datetime_cols):
                tags.append("identifier")
            elif X[c].nunique(dropna=True) <= 1:
                tags.append("constant")
            elif pd.api.types.is_numeric_dtype(X[c]):
                tags.append("numeric")
            elif X[c].nunique() > HIGH_CARDINALITY_THRESHOLD:
                tags.append("categorical")
                tags.append("high_cardinality")
            else:
                tags.append("categorical")

            if X[c].isnull().any():
                tags.append("nullable")
            if X[c].isnull().mean() > 0.5:
                tags.append("sparse")
            semantic_types[c] = tags

        # Infer dataset structure (IID vs Grouped vs Temporal vs Panel)
        id_like = self._detect_id_cols(X, n, exclude=datetime_cols)
        group_candidates = [
            c for c in X.select_dtypes(include=["object", "int64"]).columns
            if c not in id_like and 1 < X[c].nunique() <= (0.8 * n)
        ]
        is_grouped = len(group_candidates) > 0
        is_temporal = len(datetime_cols) > 0
        is_panel = is_grouped and is_temporal
        dup_count = int(df.duplicated().sum())

        dataset_structure = {
            "is_iid": not (is_grouped or is_temporal),
            "is_grouped": is_grouped,
            "is_temporal": is_temporal,
            "is_panel": is_panel,
            "group_candidates": group_candidates[:5],
            "duplicate_rows": dup_count,
        }

        self.profile_ = {
            "n_samples": n,
            "n_features": p,
            "problem_type": problem_type,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": (
                imbalance_ratio is not None and imbalance_ratio > IMBALANCE_THRESHOLD
            ),
            "missing_pct": (X.isnull().mean() * 100).to_dict(),
            "has_missing": bool(X.isnull().any().any()),
            "total_missing_pct": float(X.isnull().to_numpy().mean() * 100) if p else 0.0,
            "high_card_cols": [
                c
                for c in X.select_dtypes("object").columns
                if c not in datetime_cols
                and X[c].nunique() > HIGH_CARDINALITY_THRESHOLD
            ],
            "constant_cols": [c for c in X.columns if X[c].nunique(dropna=True) <= 1],
            "datetime_cols": datetime_cols,
            "id_like_cols": id_like,
            "semantic_types": semantic_types,
            "dataset_structure": dataset_structure,
            "n_numeric": int(X.select_dtypes(include=np.number).shape[1]),
            "n_categorical": int(X.select_dtypes(include="object").shape[1]),
            "n_classes": int(y.nunique()) if problem_type == "classification" else None,
            "class_distribution": (
                self._json_safe_counts(y) if problem_type == "classification" else None
            ),
            # -- target health, consumed by dive.validation ---------------
            "target_name": self.target,
            "target_dtype": str(y.dtype),
            "target_missing_pct": float(y.isnull().mean() * 100),
            "target_n_unique": int(y.nunique(dropna=True)),
            "target_unique_ratio": float(y.nunique(dropna=True) / max(len(y), 1)),
            "target_near_constant": self._is_near_constant(y, problem_type),
            "minority_class_count": (
                int(y.value_counts().min()) if problem_type == "classification" else None
            ),
        }
        return self.profile_

    # ------------------------------------------------------------------
    def _detect_problem(self, y: pd.Series) -> str:
        """Classification vs regression, using dtype then cardinality."""
        if y.dtype == object or str(y.dtype) == "category" or y.dtype == bool:
            return "classification"
        if y.nunique() <= HIGH_CARDINALITY_THRESHOLD and y.nunique() / max(len(y), 1) < 0.05:
            return "classification"
        return "regression"

    def _detect_imbalance(self, y: pd.Series) -> Optional[float]:
        """Ratio of the most common class to the least common class."""
        counts = y.value_counts()
        if len(counts) < 2:
            return None
        smallest = counts.iloc[-1]
        if smallest == 0:
            return None
        return float(counts.iloc[0] / smallest)

    def _detect_id_cols(
        self, X: pd.DataFrame, n: int, exclude: Optional[List[str]] = None
    ) -> List[str]:
        """Columns that look like row identifiers and carry no signal.

        ``exclude`` holds columns already claimed by another handler (datetime
        columns), which must not be dropped as IDs however unique they are.
        """
        skip = set(exclude or ())
        cols: List[str] = []
        for col in X.columns:
            if col in skip:
                continue
            series = X[col]
            n_unique = series.nunique(dropna=True)
            name = str(col).lower()
            named_like_id = name in _ID_NAMES or name.endswith(_ID_SUFFIXES)

            # A name-based match needs only near-total uniqueness.
            if named_like_id and n_unique > 0.9 * n:
                cols.append(col)
                continue

            # Uniqueness alone is never sufficient for a continuous float:
            # every sampled measurement is distinct, so treating "all values
            # unique" as an ID signature would discard real features.
            if n_unique != n:
                continue
            if pd.api.types.is_float_dtype(series):
                continue
            if pd.api.types.is_integer_dtype(series):
                # Row counters are consecutive; genuine integer measurements
                # such as prices or counts are not.
                ordered = series.dropna().sort_values()
                if len(ordered) > 1 and (ordered.diff().dropna() == 1).all():
                    cols.append(col)
                continue
            # Fully unique text: a code, hash, or free-text key.
            if series.dtype == object or pd.api.types.is_string_dtype(series):
                cols.append(col)
        return cols

    @staticmethod
    def _is_near_constant(y: pd.Series, problem_type: str) -> bool:
        """True when the target is so lopsided that a constant model would win.

        For classification: one class holds >= 99% of rows.
        For regression: the target has near-zero variance.
        """
        clean = y.dropna()
        if len(clean) == 0:
            return True
        if problem_type == "classification":
            return bool(clean.value_counts(normalize=True).iloc[0] >= 0.99)
        try:
            values = clean.astype(float)
            spread = float(values.std())
            scale = max(abs(float(values.mean())), 1e-12)
            return bool(spread / scale < 1e-6)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _json_safe_counts(y: pd.Series) -> Dict[str, int]:
        """Value counts with plain-``str`` keys so the profile stays serialisable."""
        return {str(key): int(value) for key, value in y.value_counts().items()}


def summarize_profile(profile: Dict[str, Any]) -> List[str]:
    """Render a profile as human-readable lines for the console and reports."""
    lines = [
        f"Rows                : {profile.get('n_samples')}",
        f"Features            : {profile.get('n_features')} "
        f"({profile.get('n_numeric')} numeric, {profile.get('n_categorical')} categorical)",
        f"Problem type        : {profile.get('problem_type')}",
    ]
    if profile.get("problem_type") == "classification":
        lines.append(f"Classes             : {profile.get('n_classes')}")
        ratio = profile.get("imbalance_ratio")
        if ratio:
            lines.append(
                f"Class imbalance     : {ratio:.1f}:1 "
                f"({'imbalanced' if profile.get('is_imbalanced') else 'acceptable'})"
            )
    lines.append(f"Missing values      : {profile.get('total_missing_pct', 0):.2f}% of cells")
    for label, key in (
        ("Constant columns", "constant_cols"),
        ("ID-like columns", "id_like_cols"),
        ("High-cardinality", "high_card_cols"),
    ):
        values = profile.get(key) or []
        if values:
            preview = ", ".join(map(str, values[:6]))
            more = f" (+{len(values) - 6} more)" if len(values) > 6 else ""
            lines.append(f"{label:<20}: {preview}{more}")
    return lines
