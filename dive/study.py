"""Industrial API Entry Point - `dive/study.py`.

Provides fluent high-level `Study` API for Python power users and production deployments:
    study = dive.create_study("data.csv", target="churn", mode="fast")
    results = study.fit()
    predictions = study.predict(new_data)
    study.explain_decisions()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from dive.decisions import DecisionLogger
from dive.orchestration import StudyConfig, StudyOrchestrator
from dive.utils.io import load_dataframe
from dive.utils.logging import Console, get_console


class Study:
    """Industrial-grade autonomous Machine Learning Study domain entrypoint."""

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        target: str,
        mode: str = "fast",
        time_budget: str = "30m",
        memory_budget: str = "8GB",
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        console: Optional[Console] = None,
    ) -> None:
        self.raw_data = data
        self.target = target
        self.mode = mode
        self.output_dir = Path(output_dir) if output_dir else None
        self.console = console or get_console()

        # Parse time budget string like "30m", "1h", "1800s"
        secs = 1800.0
        tb = str(time_budget).lower()
        if tb.endswith("s"):
            secs = float(tb[:-1])
        elif tb.endswith("m"):
            secs = float(tb[:-1]) * 60.0
        elif tb.endswith("h"):
            secs = float(tb[:-1]) * 3600.0

        self.config = StudyConfig(
            target=target,
            mode=mode,
            time_budget_secs=secs,
            group_column=group_column,
            time_column=time_column,
        )

        self.orchestrator = StudyOrchestrator(self.config, console=self.console)
        self.result: Optional[Dict[str, Any]] = None

    def fit(self) -> Study:
        """Run complete autonomous study workflow."""
        if isinstance(self.raw_data, pd.DataFrame):
            df = self.raw_data.copy()
        else:
            df = load_dataframe(str(self.raw_data))

        self.result = self.orchestrator.run(df, output_dir=self.output_dir)
        return self

    def predict(self, data: Union[str, Path, pd.DataFrame]) -> pd.Series:
        """Predict on new dataset using the trained champion model."""
        if not self.result:
            raise RuntimeError("Study has not been fit yet. Call study.fit() first.")
        dive_engine = self.result["dive_engine"]
        df = load_dataframe(str(data)) if isinstance(data, (str, Path)) else data
        return dive_engine.predict(df)

    def explain_decisions(self) -> None:
        """Print human-readable explanation of all autonomous decisions made during the study."""
        if not self.result:
            self.console.warn("No study execution results available.")
            return
        self.console.print(self.orchestrator.logger.render_summary())

    @property
    def decisions(self) -> List[Dict[str, Any]]:
        """Return list of decision dictionaries."""
        return self.orchestrator.logger.to_list()


def create_study(
    data: Union[str, Path, pd.DataFrame],
    target: str,
    mode: str = "fast",
    time_budget: str = "30m",
    memory_budget: str = "8GB",
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Study:
    """Factory function creating a DIVE Autonomous Study instance."""
    return Study(
        data=data,
        target=target,
        mode=mode,
        time_budget=time_budget,
        memory_budget=memory_budget,
        group_column=group_column,
        time_column=time_column,
        output_dir=output_dir,
    )
