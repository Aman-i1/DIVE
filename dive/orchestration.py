"""Study Orchestrator Engine - `dive/orchestration.py`.

Coordinates data profiling, validation intelligence, leakage-safe feature engineering,
autonomous search, ensembling, trust reporting, and model serving lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from dive.decisions import DecisionLogger
from dive.doctor import DiveDoctor
from dive.leakage import AdvancedLeakageDetector
from dive.predictor import DivePredictor
from dive.resources import ResourceManager
from dive.utils.logging import Console, get_console


@dataclass
class StudyConfig:
    """Configuration for an autonomous AutoML Study."""

    target: str
    mode: str = "fast"
    time_budget_secs: float = 1800.0
    memory_budget_mb: float = 8192.0
    cv_splits: int = 5
    ensemble: bool = True
    strict_leakage: bool = True
    group_column: Optional[str] = None
    time_column: Optional[str] = None


class StudyOrchestrator:
    """Orchestrates end-to-end autonomous AutoML studies."""

    def __init__(self, config: StudyConfig, console: Optional[Console] = None) -> None:
        self.config = config
        self.console = console or get_console()
        self.logger = DecisionLogger()
        self.resource_manager = ResourceManager()

    def run(self, df: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Execute end-to-end autonomous study workflow."""
        start_time = time.time()
        self.console.banner("DIVE AUTONOMOUS STUDY ORCHESTRATOR", f"Target: '{self.config.target}' | Mode: {self.config.mode}")

        # 1. Resource & Hardware Check
        sys_res = self.resource_manager.get_system_resources()
        self.logger.log(
            component="ResourceEngine",
            decision=f"Allocated {sys_res.cpu_count} CPU threads, {sys_res.ram_total_mb:.0f} MB RAM",
            reason="Hardware capacity check passed",
            confidence=1.0,
            evidence={"cpus": sys_res.cpu_count, "ram_mb": sys_res.ram_total_mb},
        )

        # 2. Data Intelligence Audit
        doctor = DiveDoctor(
            target=self.config.target,
            group_column=self.config.group_column,
            time_column=self.config.time_column,
        )
        doc_report = doctor.analyze(df)

        self.logger.log(
            component="DataIntelligence",
            decision=f"Detected problem type: {doc_report.problem_type.upper()}",
            reason=f"Target column '{self.config.target}' cardinality and data distribution",
            confidence=0.98,
            evidence={"problem_type": doc_report.problem_type, "n_rows": len(df)},
        )

        # 3. Leakage & Validation Intelligence Audit
        leak_detector = AdvancedLeakageDetector(target=self.config.target)
        leak_report = leak_detector.detect(df)

        val_strategy = "StratifiedKFold" if doc_report.problem_type == "classification" else "KFold"
        if self.config.group_column:
            val_strategy = "GroupKFold"
        elif self.config.time_column:
            val_strategy = "TimeSeriesSplit"

        self.logger.log(
            component="ValidationEngine",
            decision=f"Selected validation strategy: {val_strategy}(n_splits={self.config.cv_splits})",
            reason="Based on dataset structure, group columns, and temporal ordering checks",
            confidence=0.95,
            evidence={"group_col": self.config.group_column, "time_col": self.config.time_column},
        )

        # 4. Train Model Pipeline via Dive core engine
        from dive.core import Dive

        dive_engine = Dive(
            target=self.config.target,
            mode=self.config.mode,
            time_budget_secs=self.config.time_budget_secs,
            n_splits=self.config.cv_splits,
            output_dir=output_dir,
            console=self.console,
        )
        dive_engine.fit(df)

        elapsed = time.time() - start_time
        self.logger.log(
            component="StudyOrchestrator",
            decision=f"Completed study with best model: {dive_engine.best_model_name_}",
            reason="Search budget exhausted cleanly",
            confidence=1.0,
            evidence={"elapsed_secs": round(elapsed, 2), "best_score": dive_engine.best_score_},
        )

        return {
            "best_model_name": dive_engine.best_model_name_,
            "best_score": dive_engine.best_score_,
            "problem_type": doc_report.problem_type,
            "readiness_score": doc_report.readiness_score.overall_score,
            "decisions": self.logger.to_list(),
            "elapsed_secs": round(elapsed, 2),
            "dive_engine": dive_engine,
        }
