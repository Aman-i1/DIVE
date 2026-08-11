"""Autonomous Multi-Fidelity Search Scheduler (ASHA / Hyperband) - `dive/search_scheduler.py`.

Implements Asynchronous Successive Halving (ASHA) multi-fidelity trial scheduling with
multi-objective trade-off scoring (accuracy, calibration, latency, memory) and early pruning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dive.decisions import DecisionLogger


@dataclass
class Trial:
    """Represents a hyperparameter search trial across multi-fidelity rungs."""

    trial_id: str
    model_name: str
    hyperparameters: Dict[str, Any]
    current_fidelity: float = 0.1  # 0.1 to 1.0 (sample fraction or iterations)
    current_rung: int = 0
    scores: Dict[int, float] = field(default_factory=dict)
    status: str = "PENDING"  # PENDING, RUNNING, PROMOTED, PRUNED, COMPLETED

    def multi_objective_score(
        self,
        primary_metric_score: float,
        calibration_error: float = 0.0,
        latency_secs: float = 0.0,
        memory_mb: float = 0.0,
    ) -> float:
        """Calculate composite multi-objective optimization score."""
        penalty = (calibration_error * 0.15) + (min(latency_secs, 10.0) * 0.02) + (min(memory_mb, 1000.0) * 0.0001)
        return primary_metric_score - penalty


@dataclass
class Rung:
    """Fidelity level rung in ASHA scheduler."""

    rung_index: int
    fidelity: float
    trials: List[Trial] = field(default_factory=list)


class ASHASearchScheduler:
    """Asynchronous Successive Halving Algorithm (ASHA) search scheduler."""

    def __init__(
        self,
        min_fidelity: float = 0.1,
        max_fidelity: float = 1.0,
        reduction_factor: int = 3,
        logger: Optional[DecisionLogger] = None,
    ) -> None:
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self.reduction_factor = reduction_factor
        self.logger = logger or DecisionLogger()

        # Build rungs: e.g. rung 0 (0.1), rung 1 (0.3), rung 2 (1.0)
        self.rungs: List[Rung] = []
        fidelity = min_fidelity
        rung_idx = 0
        while fidelity < max_fidelity:
            self.rungs.append(Rung(rung_index=rung_idx, fidelity=round(fidelity, 2)))
            fidelity *= reduction_factor
            rung_idx += 1
        if not self.rungs or self.rungs[-1].fidelity < max_fidelity:
            self.rungs.append(Rung(rung_index=len(self.rungs), fidelity=round(max_fidelity, 2)))

    def submit_trial(self, trial_id: str, model_name: str, hyperparameters: Dict[str, Any]) -> Trial:
        """Submit a new trial at initial minimum fidelity."""
        trial = Trial(
            trial_id=trial_id,
            model_name=model_name,
            hyperparameters=hyperparameters,
            current_fidelity=self.rungs[0].fidelity,
            current_rung=0,
            status="RUNNING",
        )
        self.rungs[0].trials.append(trial)
        return trial

    def report_trial_result(
        self,
        trial: Trial,
        primary_metric_score: float,
        calibration_error: float = 0.0,
        latency_secs: float = 0.0,
        memory_mb: float = 0.0,
    ) -> str:
        """Report metric score for trial and evaluate ASHA promotion or early pruning."""
        composite_score = trial.multi_objective_score(primary_metric_score, calibration_error, latency_secs, memory_mb)
        rung_idx = trial.current_rung
        trial.scores[rung_idx] = composite_score

        # Check if trial has reached top max fidelity
        if rung_idx >= len(self.rungs) - 1:
            trial.status = "COMPLETED"
            self.logger.log(
                component="SearchScheduler",
                decision=f"Trial '{trial.trial_id}' ({trial.model_name}) COMPLETED at max fidelity {trial.current_fidelity:.2f}",
                reason="Top rung achieved cleanly",
                confidence=1.0,
                evidence={"score": round(composite_score, 4)},
            )
            return "COMPLETED"

        # Check promotion threshold in current rung
        current_rung_trials = [t for t in self.rungs[rung_idx].trials if rung_idx in t.scores]
        current_rung_trials.sort(key=lambda t: t.scores[rung_idx], reverse=True)

        k_promoted = max(1, len(current_rung_trials) // self.reduction_factor)
        promoted_trials = current_rung_trials[:k_promoted]

        if trial in promoted_trials:
            next_rung_idx = rung_idx + 1
            next_fidelity = self.rungs[next_rung_idx].fidelity
            trial.current_rung = next_rung_idx
            trial.current_fidelity = next_fidelity
            trial.status = "PROMOTED"
            self.rungs[next_rung_idx].trials.append(trial)

            self.logger.log(
                component="SearchScheduler",
                decision=f"PROMOTED trial '{trial.trial_id}' ({trial.model_name}) to rung {next_rung_idx} (fidelity {next_fidelity:.2f})",
                reason=f"Top {1.0/self.reduction_factor:.0%} quantile score ({composite_score:.4f}) in rung {rung_idx}",
                confidence=0.95,
                evidence={"score": round(composite_score, 4), "next_fidelity": next_fidelity},
            )
            return "PROMOTED"
        else:
            trial.status = "PRUNED"
            self.logger.log(
                component="SearchScheduler",
                decision=f"PRUNED trial '{trial.trial_id}' ({trial.model_name}) at rung {rung_idx} (fidelity {trial.current_fidelity:.2f})",
                reason=f"Score ({composite_score:.4f}) fell below promotion quantile",
                confidence=0.95,
                evidence={"score": round(composite_score, 4)},
            )
            return "PRUNED"
