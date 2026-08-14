"""Unit & Integration tests for Batch Inference, Observability Drift & Champion/Challenger Gate."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from dive.batch_inference import BatchInferenceEngine, BatchInferenceStats
from dive.champion_challenger import ChampionChallengerEvaluator, PromotionVerdict
from dive.observability import DriftMetricResult, ObservabilityEngine, ObservabilityReport


def test_batch_inference_file_streaming(tmp_path: Path) -> None:
    # Create sample CSV file with 50 rows
    df = pd.DataFrame({
        "feat1": np.random.normal(0, 1, 50),
        "feat2": np.random.normal(5, 2, 50),
        "target": np.random.choice([0, 1], 50),
    })
    in_csv = tmp_path / "input.csv"
    out_csv = tmp_path / "output.csv"
    df.to_csv(in_csv, index=False)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(df[["feat1", "feat2"]], df["target"])

    engine = BatchInferenceEngine(predictor=model, chunk_size=15)
    stats = engine.predict_file(
        input_path=in_csv,
        output_path=out_csv,
        target_column="target",
        include_probabilities=True,
    )

    assert stats.total_rows == 50
    assert stats.chunk_count == 4  # 15 + 15 + 15 + 5
    assert out_csv.exists()

    result_df = pd.read_csv(out_csv)
    assert len(result_df) == 50
    assert "prediction" in result_df.columns
    assert "prob_class_0" in result_df.columns


def test_observability_drift_and_retraining() -> None:
    np.random.seed(42)
    # Reference baseline data
    ref_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "salary": np.random.normal(50000, 10000, 200),
    })

    # Drifting current production data (salary shifted up, age stable)
    curr_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "salary": np.random.normal(85000, 15000, 200),  # Heavy drift
    })

    obs = ObservabilityEngine(psi_threshold=0.20)
    report = obs.audit_drift(ref_df, curr_df)

    assert isinstance(report, ObservabilityReport)
    assert report.features_monitored_count == 2
    assert "salary" in report.feature_metrics
    assert report.feature_metrics["salary"].drift_status == "SIGNIFICANT_DRIFT"
    assert report.feature_metrics["salary"].psi_score > 0.20
    assert report.retraining_alert_level in ("RETRAIN_RECOMMENDED", "RETRAIN_URGENT")
    assert report.retraining_urgency_score >= 40.0


def test_champion_challenger_promotion() -> None:
    # 10 cross-validation fold scores
    # Challenger consistently outperforms Champion across folds
    champ_cv = np.array([0.80, 0.81, 0.79, 0.82, 0.80, 0.81, 0.79, 0.80, 0.82, 0.81])
    chall_cv = np.array([0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.84, 0.85, 0.87, 0.86])

    evaluator = ChampionChallengerEvaluator(min_improvement_pct=0.01, alpha=0.05)
    verdict = evaluator.evaluate_promotion(
        champion_scores=champ_cv,
        challenger_scores=chall_cv,
        champion_name="RF_v1",
        challenger_name="LightGBM_v2",
        metric_name="ROC_AUC",
    )

    assert isinstance(verdict, PromotionVerdict)
    assert verdict.verdict == "APPROVED"
    assert verdict.is_statistically_significant is True
    assert verdict.metric_delta > 0.04

    # Now test a worse challenger
    worse_cv = np.array([0.70, 0.71, 0.69, 0.72, 0.70, 0.71, 0.69, 0.70, 0.72, 0.71])
    worse_verdict = evaluator.evaluate_promotion(
        champion_scores=champ_cv,
        challenger_scores=worse_cv,
        champion_name="RF_v1",
        challenger_name="Linear_v0",
    )
    assert worse_verdict.verdict == "REJECTED"
