"""Phase 12 Tests - DIVE NLP Production Drift Monitoring & Telemetry.

Verifies:
1. NLPDriftMonitor on stable text streams (no false positive drift alerts).
2. Token and character length distribution shift detection (KS-test and Wasserstein distance).
3. Out-Of-Vocabulary (OOV) rate shift and emergent vocabulary tracking.
4. Prediction class distribution drift via Population Stability Index (PSI).
5. NLPDriftReport dictionary serialization and ASCII terminal rendering.
6. Convenience functional helper monitor_nlp_drift().
"""

from __future__ import annotations

import pandas as pd
import pytest

from dive.nlp import (
    NLPDriftMonitor,
    NLPDriftReport,
    monitor_nlp_drift,
)


@pytest.fixture
def baseline_texts() -> List[str]:
    """Fixture providing reference corpus of customer reviews."""
    return [
        "Incredible product with top tier build quality and fast shipping.",
        "Loved everything about this device, exceeded my expectations!",
        "Fantastic customer support and seamless onboarding experience.",
        "Superb performance under heavy loads, highly recommended.",
        "Great value for money, absolutely delightful product overall.",
        "Worst purchase ever made, completely broken on arrival.",
        "Terrible quality, broke within two days of normal usage.",
        "Horrible customer service, completely unhelpful and rude.",
        "Do not buy this item, complete waste of money and frustration.",
        "Defective unit, useless customer support, refund was refused.",
    ]


def test_nlp_drift_monitor_stable_distribution(baseline_texts: List[str]) -> None:
    """Verify that similar text distributions do not trigger false positive drift."""
    monitor = NLPDriftMonitor(
        reference_texts=baseline_texts,
        reference_predictions=["pos", "pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg", "neg"],
    )

    # Current texts from same domain
    current_texts = [
        "Loved the build quality and fast shipping, great device.",
        "Terrible customer service and broken on arrival.",
    ]
    current_preds = ["pos", "neg"]

    report = monitor.check_drift(
        current_texts=current_texts,
        current_predictions=current_preds,
    )

    assert isinstance(report, NLPDriftReport)
    assert report.drift_detected is False
    assert len(report.alerts) == 0
    assert report.reference_samples == 10
    assert report.current_samples == 2


def test_nlp_drift_monitor_length_shift(baseline_texts: List[str]) -> None:
    """Verify detection of significant token and character length drift."""
    monitor = NLPDriftMonitor(reference_texts=baseline_texts)

    # Ingest massive multi-sentence paragraphs
    long_texts = [
        "This is an extraordinarily long text document containing an enormous amount of descriptive sentences, "
        "expanding multiple paragraphs and detailing every intricate subsystem, configuration file, database schema, "
        "and architectural pattern far beyond the baseline length distribution of standard reviews.",
        "Another extremely lengthy document with comprehensive explanations, detailed diagnostic checklists, "
        "benchmarking procedures, and extensive code documentation intended to test distribution shift.",
    ]

    report = monitor.check_drift(current_texts=long_texts)
    assert report.drift_detected is True
    assert report.length_drift["token_length"]["drift_detected"] is True
    assert any("Token length drift" in alert for alert in report.alerts)


def test_nlp_drift_monitor_oov_vocabulary_shift(baseline_texts: List[str]) -> None:
    """Verify Out-Of-Vocabulary (OOV) rate detection upon domain vocabulary shifts."""
    monitor = NLPDriftMonitor(reference_texts=baseline_texts, oov_threshold=0.20)

    # Inject out-of-domain medical / pharmacological vocabulary
    medical_texts = [
        "Patient presented with acute myocardial infarction and hypercholesterolemia requiring pharmacotherapy.",
        "Administered intravenous anticoagulant regimen following echocardiographic diagnostic assessment.",
    ]

    report = monitor.check_drift(current_texts=medical_texts)
    assert report.drift_detected is True
    voc = report.vocabulary_drift
    assert voc["drift_detected"] is True
    assert voc["oov_rate"] > 0.20
    assert "myocardial" in voc["top_emergent_words"] or "infarction" in voc["top_emergent_words"]
    assert any("Out-of-Vocabulary" in alert for alert in report.alerts)


def test_nlp_drift_monitor_prediction_distribution_shift(baseline_texts: List[str]) -> None:
    """Verify detection of prediction class distribution shift using Population Stability Index."""
    ref_preds = ["pos", "pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg", "neg"]  # 50/50 balanced
    monitor = NLPDriftMonitor(
        reference_texts=baseline_texts,
        reference_predictions=ref_preds,
        psi_threshold=0.20,
    )

    current_texts = ["Sample text"] * 10
    # Production stream is 90% negative
    curr_preds = ["neg"] * 9 + ["pos"]

    report = monitor.check_drift(
        current_texts=current_texts,
        current_predictions=curr_preds,
    )

    assert report.drift_detected is True
    assert report.prediction_drift is not None
    assert report.prediction_drift["drift_detected"] is True
    assert report.prediction_drift["psi"] >= 0.20
    assert any("Prediction class distribution drift" in alert for alert in report.alerts)


def test_nlp_drift_report_render_and_serialization(baseline_texts: List[str]) -> None:
    """Verify dictionary serialization and formatted ASCII terminal report rendering."""
    report = monitor_nlp_drift(
        reference_texts=baseline_texts,
        current_texts=["Sample query document"],
    )

    # 1. to_dict()
    d = report.to_dict()
    assert d["reference_samples"] == 10
    assert d["current_samples"] == 1
    assert "length_drift" in d
    assert "vocabulary_drift" in d

    # 2. render()
    rendered = report.render()
    assert "DIVE NLP DRIFT & MONITORING REPORT" in rendered
    assert "DOCUMENT LENGTH DRIFT" in rendered
    assert "VOCABULARY & OOV SHIFT" in rendered
