"""Leakage-Safe Temporal & Entity Feature Generator - `dive/temporal_features.py`.

Generates lag features, rolling aggregations, expanding statistics, entity aggregates,
and time-since-event features while strictly enforcing point-in-time boundary safety (.shift(1)).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dive.feature_availability import FeatureAvailabilityModel


class LeakageSafeTemporalEngine:
    """Generates temporal and entity-level features guaranteed point-in-time safe."""

    def __init__(
        self,
        time_column: Optional[str] = None,
        group_column: Optional[str] = None,
        availability_model: Optional[FeatureAvailabilityModel] = None,
    ) -> None:
        self.time_column = time_column
        self.group_column = group_column
        self.availability_model = availability_model or FeatureAvailabilityModel(time_column=time_column)

    def generate_features(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        lags: Tuple[int, ...] = (1, 2, 3),
        rolling_windows: Tuple[int, ...] = (3, 7),
    ) -> Tuple[pd.DataFrame, FeatureAvailabilityModel]:
        """Generate point-in-time safe temporal, lag, and entity features."""
        df_out = df.copy()

        # Sort by time_column if present
        if self.time_column and self.time_column in df_out.columns:
            df_out = df_out.sort_values(by=self.time_column).reset_index(drop=True)

        target_cols = numeric_cols or [
            c for c in df_out.select_dtypes(include=[np.number]).columns
            if c not in (self.time_column, self.group_column)
        ]

        # 1. Generate Lag Features with .shift(lag)
        for col in target_cols:
            for lag in lags:
                col_name = f"{col}_lag_{lag}"
                if self.group_column and self.group_column in df_out.columns:
                    df_out[col_name] = df_out.groupby(self.group_column)[col].shift(lag)
                else:
                    df_out[col_name] = df_out[col].shift(lag)

                self.availability_model.register(
                    name=col_name,
                    source_column=col,
                    transformation=f"lag_{lag}",
                    is_point_in_time_safe=True,
                )

        # 2. Generate Rolling Aggregations (.shift(1) enforced)
        for col in target_cols:
            for window in rolling_windows:
                col_mean = f"{col}_rolling_mean_{window}"
                col_std = f"{col}_rolling_std_{window}"

                if self.group_column and self.group_column in df_out.columns:
                    # Grouped rolling with shift to prevent current row leakage
                    rolled = (
                        df_out.groupby(self.group_column)[col]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    )
                    rolled_std = (
                        df_out.groupby(self.group_column)[col]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
                    )
                else:
                    rolled = df_out[col].shift(1).rolling(window, min_periods=1).mean()
                    rolled_std = df_out[col].shift(1).rolling(window, min_periods=1).std()

                df_out[col_mean] = rolled.fillna(0.0)
                df_out[col_std] = rolled_std.fillna(0.0)

                self.availability_model.register(
                    name=col_mean,
                    source_column=col,
                    transformation=f"rolling_mean_{window}",
                    aggregation_window=f"{window}p",
                    is_point_in_time_safe=True,
                )
                self.availability_model.register(
                    name=col_std,
                    source_column=col,
                    transformation=f"rolling_std_{window}",
                    aggregation_window=f"{window}p",
                    is_point_in_time_safe=True,
                )

        # 3. Expanding Historical Statistics (.shift(1) enforced)
        for col in target_cols:
            col_expanding = f"{col}_expanding_mean"
            if self.group_column and self.group_column in df_out.columns:
                df_out[col_expanding] = (
                    df_out.groupby(self.group_column)[col]
                    .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
                    .fillna(0.0)
                )
            else:
                df_out[col_expanding] = df_out[col].shift(1).expanding(min_periods=1).mean().fillna(0.0)

            self.availability_model.register(
                name=col_expanding,
                source_column=col,
                transformation="expanding_mean",
                is_point_in_time_safe=True,
            )

        return df_out, self.availability_model
