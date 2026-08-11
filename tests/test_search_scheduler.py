"""Unit & Integration tests for Phase 5: Autonomous Multi-Fidelity Search Scheduler (ASHA)."""

from __future__ import annotations

import pytest

from dive.search_scheduler import ASHASearchScheduler, Trial


def test_asha_search_scheduler_rungs() -> None:
    scheduler = ASHASearchScheduler(min_fidelity=0.1, max_fidelity=1.0, reduction_factor=3)
    assert len(scheduler.rungs) == 3
    assert scheduler.rungs[0].fidelity == 0.1
    assert scheduler.rungs[1].fidelity == 0.3
    assert scheduler.rungs[2].fidelity == 1.0


def test_asha_trial_promotion_and_pruning() -> None:
    scheduler = ASHASearchScheduler(min_fidelity=0.1, max_fidelity=1.0, reduction_factor=2)

    # Submit 2 trials
    t1 = scheduler.submit_trial("t1", "RandomForest", {"n_estimators": 50})
    t2 = scheduler.submit_trial("t2", "RandomForest", {"n_estimators": 10})

    # Report scores: t1 gets 0.90, t2 gets 0.60
    status1 = scheduler.report_trial_result(t1, primary_metric_score=0.90)
    status2 = scheduler.report_trial_result(t2, primary_metric_score=0.60)

    assert status1 == "PROMOTED"
    assert status2 == "PRUNED"
    assert t1.current_rung == 1
    assert t1.current_fidelity == 0.2
    assert t2.status == "PRUNED"
