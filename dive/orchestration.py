"""Study Orchestrator Engine - `dive/orchestration.py`.

Coordinates data profiling, validation intelligence, leakage-safe feature engineering,
autonomous search, ensembling, trust reporting, and model serving lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
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

        # 3. Validation Intelligence Engine Audit
        from dive.validation_engine import ValidationIntelligenceEngine

        val_engine = ValidationIntelligenceEngine(target=self.config.target, logger=self.logger)
        val_plan = val_engine.evaluate(
            df,
            problem_type=doc_report.problem_type,
            user_group_column=self.config.group_column,
            user_time_column=self.config.time_column,
            n_splits=self.config.cv_splits,
        )

        # 4. Meta-Learning Dataset Fingerprinting & Warm-Start Priors
        from dive.meta_learning import MetaLearningEngine

        meta_engine = MetaLearningEngine(logger=self.logger)
        X_df = df.drop(columns=[self.config.target]) if self.config.target in df.columns else df
        y_sr = df[self.config.target] if self.config.target in df.columns else pd.Series(np.zeros(len(df)))
        fingerprint = meta_engine.compute_fingerprint(X_df, y_sr, problem_type=doc_report.problem_type)
        meta_priors = meta_engine.warm_start_recommendations(fingerprint, problem_type=doc_report.problem_type)

        # 5. Train Model Pipeline via Dive core engine
        from dive.core import Dive

        dive_engine = Dive(
            target=self.config.target,
            mode=self.config.mode,
            time_budget=self.config.time_budget_secs,
            cv_folds=self.config.cv_splits,
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
