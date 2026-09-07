"""AutoDL Trial Leaderboard & Ranking Engine - ``dive/dl/automl/leaderboard.py``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from dive.dl.automl.trial import DLTrial
from dive.utils.report import TOP, ReportBuilder


class DLLeaderboard:
    """Manages, ranks, and visualizes candidate deep learning trials."""

    def __init__(self, primary_metric: str = "accuracy") -> None:
        self.primary_metric = primary_metric
        self.trials: List[DLTrial] = []

    def add_trial(self, trial: DLTrial) -> None:
        self.trials.append(trial)

    @property
    def successful_trials(self) -> List[DLTrial]:
        """List of successfully completed trials sorted by composite score descending."""
        successful = [t for t in self.trials if t.status == "SUCCESS"]
        return sorted(successful, key=lambda t: t.composite_score, reverse=True)

    @property
    def champion_trial(self) -> Optional[DLTrial]:
        """Return the top-ranked winning trial."""
        successful = self.successful_trials
        return successful[0] if successful else None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert leaderboard to pandas DataFrame."""
        rows = []
        for rank, trial in enumerate(self.successful_trials, start=1):
            rows.append(
                {
                    "Rank": rank,
                    "Architecture": trial.model_name,
                    "Backend": trial.backend,
                    f"Score ({self.primary_metric})": round(trial.primary_metric_score, 4),
                    "Composite Score": round(trial.composite_score, 4),
                    "Latency (ms)": round(trial.inference_latency_ms, 3),
                    "Train Time (ms)": round(trial.train_time_ms, 2),
                    "Status": trial.status,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.successful_trials]

    def render(self) -> str:
        """Render the leaderboard through ReportBuilder."""
        df = self.to_dataframe()
        if df.empty:
            return "AutoDL Leaderboard: No successful trials completed."

        builder = ReportBuilder("DIVE AUTODL MODEL SELECTION LEADERBOARD")
        score_column = f"Score ({self.primary_metric})"
        rows = [
            [
                row["Rank"],
                str(row["Architecture"]),
                str(row["Backend"]),
                f"{row[score_column]:.4f}",
                f"{row['Latency (ms)']:.3f}",
                f"{row['Train Time (ms)']:.2f}",
            ]
            for _, row in df.iterrows()
        ]
        builder.table(
            ["Rank", "Architecture", "Backend", "Score", "Latency (ms)", "Train (ms)"],
            rows,
            highlight_first=True,
        )

        if self.champion_trial:
            champ = self.champion_trial
            builder.blank()
            builder.status(
                TOP,
                f"Champion Model: {champ.model_name} [{champ.backend}] "
                f"({self.primary_metric}: {champ.primary_metric_score:.4f}, "
                f"latency: {champ.inference_latency_ms:.2f}ms)",
            )

        return builder.build()
