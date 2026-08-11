"""Controlled Feature Pruning & Mutual Information Selection - `dive/feature_selection.py`.

Prunes uninformative, redundant, and highly correlated features using mutual information,
variance thresholding, and correlation filtering to prevent combinatorial explosion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class FeaturePruner:
    """Prunes redundant, collinear, and low-information features."""

    def __init__(
        self,
        correlation_threshold: float = 0.98,
        min_variance: float = 1e-6,
        top_k: Optional[int] = None,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.min_variance = min_variance
        self.top_k = top_k
        self.selected_features_: List[str] = []
        self.pruned_features_: Dict[str, str] = {}

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
    ) -> pd.DataFrame:
        """Prune feature dataframe and return clean subset."""
        df_out = X.copy()
        numeric_df = df_out.select_dtypes(include=[np.number])

        # 1. Zero/Near-zero variance pruning
        for col in numeric_df.columns:
            if numeric_df[col].var() < self.min_variance:
                self.pruned_features_[col] = "near_zero_variance"
                if col in df_out.columns:
                    df_out = df_out.drop(columns=[col])

        # 2. High correlation / Collinearity pruning
        num_cols = df_out.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            corr_matrix = df_out[num_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
            for col in to_drop:
                self.pruned_features_[col] = f"collinear_correlation > {self.correlation_threshold}"
                if col in df_out.columns:
                    df_out = df_out.drop(columns=[col])

        # 3. Top K Mutual Information pruning if requested
        if self.top_k and len(df_out.columns) > self.top_k:
            clean_num = df_out.select_dtypes(include=[np.number]).fillna(0.0)
            if not clean_num.empty:
                score_func = mutual_info_classif if problem_type == "classification" else mutual_info_regression
                scores = score_func(clean_num, y)
                top_indices = np.argsort(scores)[::-1][: self.top_k]
                selected_num = set(clean_num.columns[top_indices])

                drop_cols = [c for c in clean_num.columns if c not in selected_num]
                for col in drop_cols:
                    self.pruned_features_[col] = "low_mutual_information"
                    if col in df_out.columns:
                        df_out = df_out.drop(columns=[col])

        self.selected_features_ = list(df_out.columns)
        return df_out
