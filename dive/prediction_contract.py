"""Prediction Contract Specification & Inference Engine - `dive/prediction_contract.py`.

Establishes a formal, unambiguous prediction contract before training:
- target: column to predict
- entity: primary unit of observation/entity grouping (or UNKNOWN)
- prediction_time: point in time when prediction is executed (or UNKNOWN)
- prediction_horizon: future duration/window being predicted (or UNKNOWN)
- allowed_information_until: cutoff point for feature availability (or UNKNOWN)
- problem_type: binary_classification, multiclass_classification, regression
- evaluation_metric: primary performance metric
- deployment_context: real-time, batch_daily, batch_hourly, embedded
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from dive.decisions import DecisionLogger


@dataclass
class PredictionContract:
    """Formal Prediction Contract governing model training and deployment validity."""

    target: str
    problem_type: str  # 'binary_classification', 'multiclass_classification', 'regression'
    entity: str = "UNKNOWN"
    prediction_time: str = "UNKNOWN"
    prediction_horizon: str = "UNKNOWN"
    allowed_information_until: str = "UNKNOWN"
    evaluation_metric: str = "AUTO"
    deployment_context: str = "batch"
    is_inferred: bool = True
    inferred_attributes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "problem_type": self.problem_type,
            "entity": self.entity,
            "prediction_time": self.prediction_time,
            "prediction_horizon": self.prediction_horizon,
            "allowed_information_until": self.allowed_information_until,
            "evaluation_metric": self.evaluation_metric,
            "deployment_context": self.deployment_context,
            "is_inferred": self.is_inferred,
            "inferred_attributes": self.inferred_attributes,
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Save contract to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PredictionContract":
        """Load contract from JSON file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def render(self) -> str:
        lines = [
            "FORMAL PREDICTION CONTRACT",
            "==========================",
            f"Target Column            : {self.target}",
            f"Problem Type             : {self.problem_type}",
            f"Entity / Group Column    : {self.entity}",
            f"Prediction Timestamp     : {self.prediction_time}",
            f"Prediction Horizon       : {self.prediction_horizon}",
            f"Allowed Info Until       : {self.allowed_information_until}",
            f"Evaluation Metric        : {self.evaluation_metric}",
            f"Deployment Context       : {self.deployment_context}",
            f"Contract Origin          : {'INFERRED STATISTICALLY' if self.is_inferred else 'USER SPECIFIED'}",
        ]
        if self.inferred_attributes:
            lines.append(f"Inferred Fields          : {', '.join(self.inferred_attributes)}")
        return "\n".join(lines)


class PredictionContractEngine:
    """Infers or establishes a PredictionContract from dataset analysis and user overrides."""

    def __init__(self, logger: Optional[DecisionLogger] = None) -> None:
        self.logger = logger or DecisionLogger()

    def infer_contract(
        self,
        df: pd.DataFrame,
        target: str,
        entity: Optional[str] = None,
        time_column: Optional[str] = None,
        horizon: Optional[str] = None,
        metric: Optional[str] = None,
        deployment_context: Optional[str] = None,
    ) -> PredictionContract:
        """Infer contract parameters with conservative defaults, flagging unknown items."""
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")

        inferred: List[str] = []

        # 1. Infer problem type
        y = df[target].dropna()
        n_unique = y.nunique()
        if n_unique == 2:
            problem_type = "binary_classification"
            default_metric = "roc_auc"
        elif n_unique > 2 and (y.dtype == object or pd.api.types.is_categorical_dtype(y) or n_unique <= 20):
            problem_type = "multiclass_classification"
            default_metric = "log_loss"
        else:
            problem_type = "regression"
            default_metric = "r2"

        # 2. Infer entity column if not supplied
        inferred_entity = "UNKNOWN"
        if entity:
            inferred_entity = entity
        else:
            # Check for candidate ID/group columns
            for col in df.columns:
                if col == target:
                    continue
                name_lower = col.lower()
                if any(kw in name_lower for kw in ("customer_id", "user_id", "patient_id", "device_id", "client_id", "account_id")):
                    inferred_entity = col
                    inferred.append(f"entity ({col})")
                    break

        # 3. Infer time column if not supplied
        inferred_time = "UNKNOWN"
        if time_column:
            inferred_time = time_column
        else:
            for col in df.columns:
                if col == target:
                    continue
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    inferred_time = col
                    inferred.append(f"time_column ({col})")
                    break
                name_lower = col.lower()
                if any(kw in name_lower for kw in ("timestamp", "event_time", "created_at", "date", "datetime")):
                    inferred_time = col
                    inferred.append(f"time_column ({col})")
                    break

        # 4. Prediction horizon and allowed info
        inferred_horizon = horizon if horizon else "UNKNOWN"
        inferred_allowed = f"cutoff <= {inferred_time}" if inferred_time != "UNKNOWN" else "UNKNOWN"

        contract = PredictionContract(
            target=target,
            problem_type=problem_type,
            entity=inferred_entity,
            prediction_time=inferred_time,
            prediction_horizon=inferred_horizon,
            allowed_information_until=inferred_allowed,
            evaluation_metric=metric or default_metric,
            deployment_context=deployment_context or "batch",
            is_inferred=bool(inferred),
            inferred_attributes=inferred,
        )

        self.logger.log(
            component="PredictionContractEngine",
            decision=f"Established prediction contract for target '{target}' ({problem_type})",
            reason=f"Inferred: entity={inferred_entity}, time={inferred_time}, horizon={inferred_horizon}",
            confidence=0.90 if inferred_entity != "UNKNOWN" else 0.70,
            evidence=contract.to_dict(),
        )

        return contract
