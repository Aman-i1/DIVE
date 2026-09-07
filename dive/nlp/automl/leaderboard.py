"""AutoNLP Trial Leaderboard & Ranking Engine - `dive/nlp/automl/leaderboard.py`.

Ranks candidate trials across accuracy, latency, and composite multi-objective metrics,
rendering structured terminal reports and DataFrames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from dive.nlp.automl.trial import NLPTrial
from dive.utils.report import TOP, ReportBuilder


class NLPLeaderboard:
    """Manages, ranks, and visualizes candidate trials evaluated during AutoNLP search."""

    def __init__(self, primary_metric: str = "macro_f1") -> None:
        self.primary_metric = primary_metric
        self.trials: List[NLPTrial] = []

    def add_trial(self, trial: NLPTrial) -> None:
        """Append a completed or failed trial."""
        self.trials.append(trial)

    @property
    def successful_trials(self) -> List[NLPTrial]:
        """Return list of successfully completed trials sorted by composite score descending."""
        successful = [t for t in self.trials if t.status == "SUCCESS"]
        return sorted(successful, key=lambda t: t.composite_score, reverse=True)

    @property
    def champion_trial(self) -> Optional[NLPTrial]:
        """Return top-ranked winning trial."""
        successful = self.successful_trials
        return successful[0] if successful else None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert leaderboard to pandas DataFrame."""
        rows = []
        for rank, trial in enumerate(self.successful_trials, start=1):
            rows.append(
                {
                    "Rank": rank,
                    "Model": trial.model_name,
                    "Representation": trial.representation_type,
                    f"Score ({self.primary_metric})": round(trial.primary_metric_score, 4),
                    "Composite Score": round(trial.composite_score, 4),
                    "Latency (ms)": round(trial.inference_latency_ms, 3),
                    "Train Time (ms)": round(trial.train_time_ms, 2),
                    "Status": trial.status,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return list of trial summary dictionaries."""
        return [t.to_dict() for t in self.successful_trials]

    def render(self) -> str:
        """Render the leaderboard through the shared platform renderer."""
        df = self.to_dataframe()
        if df.empty:
            return "AutoNLP Leaderboard: No successful trials completed."

        builder = ReportBuilder("DIVE AUTONLP MODEL SELECTION LEADERBOARD")
        score_column = f"Score ({self.primary_metric})"
        rows = [
            [
                row["Rank"],
                str(row["Model"]),
                str(row["Representation"]),
                f"{row[score_column]:.4f}",
                f"{row['Latency (ms)']:.3f}",
                f"{row['Train Time (ms)']:.2f}",
            ]
            for _, row in df.iterrows()
        ]
        builder.table(
            ["Rank", "Model", "Representation", "Score", "Latency (ms)", "Train (ms)"],
            rows,
            highlight_first=True,
        )

        if self.champion_trial:
            champ = self.champion_trial
            builder.blank()
            builder.status(
                TOP,
                f"Champion Model: {champ.model_name} + {champ.representation_type} "
                f"({self.primary_metric}: {champ.primary_metric_score:.4f}, "
                f"latency: {champ.inference_latency_ms:.2f}ms)",
            )

        return builder.build()
