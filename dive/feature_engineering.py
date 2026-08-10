"""Stage 2 - feature engineering.

Ported from the ``FeatureEngineer`` class in ``Automatic Machine Learning.ipynb``.
The transformation logic is unchanged: drop junk columns, expand datetimes, clip
outliers at the 1st/99th percentile, bucket rare categories, frequency-encode,
target-encode, and cast to float32.

Two things were made robust for library/CLI use, where the same object is
pickled and reused at predict time:

* ``fit_transform`` resets fitted state first, so calling it twice on one
  instance cannot double-append to ``datetime_cols_``.
* Datetime sniffing is silenced and bounded so a text column of free-form
  strings cannot emit a wall of pandas warnings.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from dive.utils.optional import is_available, load_optional

# Fraction of values that must parse as dates before a column is treated as one.
DATETIME_PARSE_THRESHOLD = 0.8
# Parts extracted from every detected datetime column.
DATETIME_PARTS = ("year", "month", "day", "dayofweek", "quarter", "weekofyear")


class FeatureEngineer:
    """Deterministic, fit/transform-symmetric feature engineering.

    All state learned during ``fit_transform`` (drop lists, clip bounds, rare
    category sets, frequency maps, the fitted target encoder) is stored on the
    instance so that ``transform`` can be applied to new data without refitting.
    """

    def __init__(
        self,
        profile: Dict[str, Any],
        target: str,
        mode: str = "balanced",
        random_state: int = 42,
        use_target_encoding: bool = True,
        use_freq_encoding: bool = True,
        outlier_clip: bool = True,
        rare_threshold: float = 0.01,
    ) -> None:
        self.profile = profile
        self.target = target
        self.mode = mode
        self.random_state = random_state
        self.use_target_encoding = use_target_encoding
        self.use_freq_encoding = use_freq_encoding
        self.outlier_clip = outlier_clip
        self.rare_threshold = rare_threshold

        self.datetime_cols_: List[str] = []
        self.freq_maps_: Dict[str, Dict[Any, float]] = {}
        self.target_enc_: Any = None
        self.rare_maps_: Dict[str, Set[Any]] = {}
        self.clip_bounds_: Dict[str, Tuple[float, float]] = {}
        self.drop_cols_: List[str] = []
        # Raw input columns seen at fit time - used by the predict-time schema check.
        self.input_columns_: List[str] = []
        self.input_dtypes_: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Learn every transformation from ``df`` and return the transformed frame."""
        self._reset()
        self.input_columns_ = [str(c) for c in df.columns]
        self.input_dtypes_ = {str(c): str(dtype) for c, dtype in df.dtypes.items()}

        df = df.copy()
        df = self._drop_junk(df)
        df = self._parse_datetime(df, fit=True)
        df = self._clip_outliers_fit(df)
        df = self._rare_categories_fit(df)
        if self.use_freq_encoding:
            df = self._freq_encode_fit(df)
        if self.use_target_encoding and is_available("category_encoders"):
            df = self._target_encode_fit(df, y)
        return self._to_float32(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformations to new data. Never refits."""
        df = df.copy()
        df = df.drop(
            columns=[c for c in self.drop_cols_ if c in df.columns], errors="ignore"
        )
        df = self._parse_datetime(df, fit=False)
        df = self._clip_outliers_transform(df)
        df = self._rare_categories_transform(df)
        if self.use_freq_encoding:
            df = self._freq_encode_transform(df)
        if self.target_enc_ is not None:
            cat_cols = [c for c in self.target_enc_.cols if c in df.columns]
            if cat_cols:
                df[cat_cols] = self.target_enc_.transform(df[cat_cols])
        return self._to_float32(df)

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        """Clear fitted state so refitting one instance is idempotent."""
        self.datetime_cols_ = []
        self.freq_maps_ = {}
        self.target_enc_ = None
        self.rare_maps_ = {}
        self.clip_bounds_ = {}
        self.drop_cols_ = []

    @staticmethod
    def _to_float32(df: pd.DataFrame) -> pd.DataFrame:
        """Halve memory for float columns; boosters accept float32 natively."""
        for col in df.select_dtypes("float64").columns:
            df[col] = df[col].astype("float32")
        return df

    def _drop_junk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop constant and ID-like columns identified during profiling."""
        candidates = list(self.profile.get("constant_cols", [])) + list(
            self.profile.get("id_like_cols", [])
        )
        to_drop = [c for c in dict.fromkeys(candidates) if c in df.columns]
        self.drop_cols_ = to_drop
        return df.drop(columns=to_drop, errors="ignore")

    # ------------------------------------------------------------------
    def _parse_datetime(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Detect date-like object columns and expand them into calendar parts."""
        if fit:
            # Prefer the profile's list: DataIntelligence identifies datetime
            # columns before ID detection, so a unique-per-row date column is
            # still available here rather than having been dropped as an ID.
            from_profile = [
                col for col in (self.profile.get("datetime_cols") or []) if col in df.columns
            ]
            if from_profile:
                self.datetime_cols_ = list(from_profile)
            else:
                for col in df.select_dtypes("object").columns:
                    if self._looks_like_datetime(df[col]):
                        self.datetime_cols_.append(col)

        for col in self.datetime_cols_:
            if col not in df.columns:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = parsed.dt.year.astype("float32")
            df[f"{col}_month"] = parsed.dt.month.astype("float32")
            df[f"{col}_day"] = parsed.dt.day.astype("float32")
            df[f"{col}_dayofweek"] = parsed.dt.dayofweek.astype("float32")
            df[f"{col}_quarter"] = parsed.dt.quarter.astype("float32")
            df[f"{col}_weekofyear"] = parsed.dt.isocalendar().week.astype("float32")
            df.drop(columns=[col], inplace=True)
        return df

    @staticmethod
    def _looks_like_datetime(series: pd.Series, sample_size: int = 1000) -> bool:
        """True when most values parse as dates, judged on a bounded sample."""
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

    # ------------------------------------------------------------------
    def _clip_outliers_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn and apply 1st/99th percentile bounds per numeric column."""
        if not self.outlier_clip:
            return df
        for col in df.select_dtypes(include=np.number).columns:
            low, high = df[col].quantile(0.01), df[col].quantile(0.99)
            if pd.isna(low) or pd.isna(high):
                continue
            self.clip_bounds_[col] = (float(low), float(high))
            df[col] = df[col].clip(low, high)
        return df

    def _clip_outliers_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.outlier_clip:
            return df
        for col, (low, high) in self.clip_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)
        return df

    # ------------------------------------------------------------------
    def _rare_categories_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bucket categories below ``rare_threshold`` frequency into ``__rare__``."""
        for col in df.select_dtypes("object").columns:
            freq = df[col].value_counts(normalize=True)
            rare = set(freq[freq < self.rare_threshold].index)
            self.rare_maps_[col] = rare
            if rare:
                df[col] = df[col].map(lambda x: "__rare__" if x in rare else x)
        return df

    def _rare_categories_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, rare in self.rare_maps_.items():
            if col in df.columns and rare:
                df[col] = df[col].map(lambda x: "__rare__" if x in rare else x)
        return df

    # ------------------------------------------------------------------
    def _freq_encode_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``<col>_freq`` column holding each category's training frequency."""
        for col in df.select_dtypes("object").columns:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            self.freq_maps_[col] = freq_map
            df[f"{col}_freq"] = df[col].map(freq_map).astype("float32")
        return df

    def _freq_encode_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unseen categories get frequency 0, which is correct: never seen in training."""
        for col, freq_map in self.freq_maps_.items():
            if col in df.columns:
                df[f"{col}_freq"] = df[col].map(freq_map).fillna(0).astype("float32")
        return df

    # ------------------------------------------------------------------
    def _target_encode_fit(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Replace categoricals with smoothed target means (category_encoders)."""
        ce = load_optional("category_encoders")
        if ce is None:
            return df
        cat_cols = df.select_dtypes("object").columns.tolist()
        if not cat_cols:
            return df
        try:
            self.target_enc_ = ce.TargetEncoder(cols=cat_cols, smoothing=10)
            df[cat_cols] = self.target_enc_.fit_transform(df[cat_cols], y)
        except Exception:
            # Encoding is an enhancement; on failure keep the raw categoricals
            # and let the one-hot encoder in the preprocessor handle them.
            self.target_enc_ = None
        return df

    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """Return a human-readable record of what this engineer did."""
        return {
            "dropped_columns": self.drop_cols_,
            "datetime_columns": self.datetime_cols_,
            "datetime_parts": list(DATETIME_PARTS) if self.datetime_cols_ else [],
            "outlier_clip": self.outlier_clip,
            "n_clipped_columns": len(self.clip_bounds_),
            "clip_bounds_sample": {
                col: bounds for col, bounds in list(self.clip_bounds_.items())[:5]
            },
            "rare_category_threshold": self.rare_threshold,
            "rare_maps_sample": {
                col: sorted(map(str, values))[:5]
                for col, values in list(self.rare_maps_.items())[:3]
            },
            "frequency_encoded_cols": list(self.freq_maps_.keys()),
            "target_encoded_cols": (
                list(self.target_enc_.cols) if self.target_enc_ is not None else []
            ),
        }

    def missing_input_columns(self, df: pd.DataFrame) -> List[str]:
        """Return fit-time input columns absent from ``df``, ignoring dropped ones.

        Columns dropped during fitting (constant/ID-like) are excluded: their
        absence at predict time cannot change a prediction.
        """
        required = [c for c in self.input_columns_ if c not in self.drop_cols_]
        present = {str(c) for c in df.columns}
        return [c for c in required if c not in present]
