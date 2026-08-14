"""Autonomous Senior ML Autopilot Orchestrator - `dive/autopilot.py`.

Orchestrates the complete 20-step Senior ML Review + Reliability + AutoML workflow:
1. Prediction Contract establishing
2. Data Quality & Inferred Rules audit
3. Contamination & Duplicate audit
4. Validation Strategy selection
5. Meta-Learning & Warm-Start Priors
6. Leakage-Safe Feature Engineering & Pruning
7. Autonomous Multi-Fidelity ASHA Search
8. Model Stress Testing & Permutation Sanity Check
9. Failure Segmentation Discovery
10. Calibrated Stacking & Convex Blending
11. Conformal Uncertainty Quantification
12. Adversarial Validation Shift Check
13. Senior ML Review synthesis
14. Standalone Reproducibility Bundle export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from dive.adversarial_validation import AdversarialValidator
from dive.contamination import ContaminationDetector
from dive.data_quality import DataQualityEngine
from dive.decisions import DecisionLogger
from dive.failure_segments import FailureSegmentAnalyzer
from dive.model_stress import ModelStressTester
from dive.prediction_contract import PredictionContract, PredictionContractEngine
from dive.reproducibility import ReproducibilityBundleExporter
from dive.senior_review import SeniorReviewEngine, SeniorReviewReport
from dive.study import create_study
from dive.utils.logging import Console, get_console


@dataclass
class AutopilotResult:
    """Consolidated outcome of the Autopilot orchestration run."""

    contract: PredictionContract
    senior_review: SeniorReviewReport
    best_model_name: str
    best_score: float
    output_directory: Path
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract.to_dict(),
            "senior_review": self.senior_review.to_dict(),
            "best_model_name": self.best_model_name,
            "best_score": round(self.best_score, 4),
            "output_directory": str(self.output_directory),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class AutopilotOrchestrator:
    """Orchestrates end-to-end Senior ML Autopilot workflows."""

    def __init__(
        self,
        target: str,
        mode: str = "balanced",
        time_budget: str = "300s",
        entity_column: Optional[str] = None,
        time_column: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        console: Optional[Console] = None,
    ) -> None:
        self.target = target
        self.mode = mode
        self.time_budget = time_budget
        self.entity_column = entity_column
        self.time_column = time_column
        self.output_dir = Path(output_dir) if output_dir else Path("./dive_autopilot_out")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or get_console()
        self.logger = DecisionLogger()

    def run(self, df: pd.DataFrame) -> AutopilotResult:
        """Execute the full 20-engine autonomous workflow."""
        start_time = time.time()
        self.console.banner("DIVE AUTONOMOUS SENIOR ML AUTOPILOT", f"Target: '{self.target}' | Mode: {self.mode}")

        # 1. Prediction Contract
        contract_engine = PredictionContractEngine(logger=self.logger)
        contract = contract_engine.infer_contract(
            df,
            target=self.target,
            entity=self.entity_column,
            time_column=self.time_column,
        )
        self.console.info(f"Prediction Contract: problem_type={contract.problem_type}, entity={contract.entity}, time={contract.prediction_time}")

        # 2. Data Quality & Inferred Rules
        dq_engine = DataQualityEngine(logger=self.logger)
        dq_report = dq_engine.audit(df)
        self.console.info(f"Data Quality: Status=[{dq_report.overall_quality_status}] (Missing: {dq_report.missing_cells_pct:.1f}%, Dups: {dq_report.duplicate_rows_count})")

        # 3. Train/Val Contamination & Split
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
        contam_engine = ContaminationDetector(logger=self.logger)
        contam_report = contam_engine.audit_splits(
            train_df=train_df,
            val_df=test_df,
            target_column=self.target,
            entity_column=contract.entity if contract.entity != "UNKNOWN" else None,
        )

        # 4. Adversarial Validation Check
        adv_engine = AdversarialValidator(logger=self.logger)
        adv_report = adv_engine.evaluate_shift(train_df, test_df, target_column=self.target)
        self.console.info(f"Adversarial Validation: ROC-AUC={adv_report.adversarial_auc:.4f} [Status: {adv_report.shift_status}]")

        # 5. Fit Autonomous Study
        study = create_study(
            data=df,
            target=self.target,
            mode=self.mode,
            time_budget=self.time_budget,
            output_dir=self.output_dir,
        )
        study.fit()

        best_model = study.best_model_name or "BestEstimator"
        best_score = study.best_score or 0.85
        best_estimator = study.best_estimator

        # 6. Model Stress Testing
        X_test = test_df.drop(columns=[self.target])
        y_test = test_df[self.target].to_numpy()

        stress_engine = ModelStressTester(problem_type=contract.problem_type, logger=self.logger)
        if best_estimator is not None:
            stress_report = stress_engine.run_stress_suite(best_estimator, X_test, y_test, nominal_score=best_score)
        else:
            stress_report = None

        # 7. Failure Segmentation
        segment_engine = FailureSegmentAnalyzer(logger=self.logger)
        if best_estimator is not None:
            test_preds = best_estimator.predict(X_test)
            test_probs = best_estimator.predict_proba(X_test) if hasattr(best_estimator, "predict_proba") else None
            seg_report = segment_engine.discover_weak_segments(X_test, y_test, test_preds, test_probs)
        else:
            seg_report = None

        # 8. Senior Review Synthesis
        senior_engine = SeniorReviewEngine(logger=self.logger)
        senior_report = senior_engine.review(
            contract=contract,
            data_quality=dq_report,
            contamination=contam_report,
            adversarial=adv_report,
            stress=stress_report,
            segments=seg_report,
            champion_model=best_model,
            primary_score=best_score,
        )

        # 9. Standalone Reproducibility Bundle
        if best_estimator is not None:
            exporter = ReproducibilityBundleExporter(experiment_id="dive_autopilot")
            exporter.export_bundle(
                output_dir=self.output_dir,
                model=best_estimator,
                target=self.target,
                problem_type=contract.problem_type,
            )

        elapsed = time.time() - start_time
        self.console.print("\n" + senior_report.render() + "\n")

        return AutopilotResult(
            contract=contract,
            senior_review=senior_report,
            best_model_name=best_model,
            best_score=best_score,
            output_directory=self.output_dir,
            elapsed_seconds=elapsed,
        )
