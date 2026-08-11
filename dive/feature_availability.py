"""Explicit Feature Availability & Point-in-Time Metadata Model - `dive/feature_availability.py`.

Distinguishes information available at prediction time from information generated
after prediction, enforcing strict temporal point-in-time boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureMetadata:
    """Metadata tracking availability boundaries and transformation lineage for a feature."""

    name: str
    source_column: str
    available_time_col: Optional[str] = None
    is_point_in_time_safe: bool = True
    transformation: str = "raw"
    aggregation_window: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_column": self.source_column,
            "available_time_col": self.available_time_col,
            "is_point_in_time_safe": self.is_point_in_time_safe,
            "transformation": self.transformation,
            "aggregation_window": self.aggregation_window,
        }


class FeatureAvailabilityModel:
    """Registry tracking availability constraints across generated features."""

    def __init__(self, time_column: Optional[str] = None) -> None:
        self.time_column = time_column
        self.registry: Dict[str, FeatureMetadata] = {}

    def register(
        self,
        name: str,
        source_column: str,
        transformation: str = "raw",
        is_point_in_time_safe: bool = True,
        aggregation_window: Optional[str] = None,
    ) -> FeatureMetadata:
        """Register feature metadata and availability lineage."""
        meta = FeatureMetadata(
            name=name,
            source_column=source_column,
            available_time_col=self.time_column,
            is_point_in_time_safe=is_point_in_time_safe,
            transformation=transformation,
            aggregation_window=aggregation_window,
        )
        self.registry[name] = meta
        return meta

    def get_unsafe_features(self) -> List[str]:
        """Return list of feature names flagged as point-in-time unsafe."""
        return [name for name, meta in self.registry.items() if not meta.is_point_in_time_safe]
