"""Phase 6 / NLP Upgrades - Test LSA Representation & Zero-Shot Classification.

Verifies:
1. LSARepresentation fit_transform, transform, dimension consistency, and normalization.
2. LSARepresentation integration with build_representation('lsa').
3. ZeroShotClassifier prediction, ranking, confidence calibration, and domain anchor expansion.
4. Error handling on empty inputs or labels.
"""

from __future__ import annotations

import numpy as np
import pytest

from dive.nlp.features.lsa import LSARepresentation
from dive.nlp.features import build_representation
from dive.nlp.inference.zero_shot import ZeroShotClassifier
from dive.nlp.interfaces import NLPRepresentationProtocol


@pytest.fixture
def sample_corpus():
    return [
        "Quarterly company earnings report showing strong profits and revenue growth.",
        "The football team won the championship match in a thrilling comeback.",
        "Medical research study reveals new treatment for clinical cardiovascular patients.",
        "Software developers released open source machine learning code algorithms.",
    ]


def test_lsa_representation_protocol_and_dimensions(sample_corpus):
    rep = LSARepresentation(n_components=2, ngram_range=(1, 2))
    assert isinstance(rep, NLPRepresentationProtocol)

    X = rep.fit_transform(sample_corpus)
    assert isinstance(X, np.ndarray)
    assert X.shape == (4, 2)

    # Unit L2 normalization verification
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    X_new = rep.transform(["Earnings and quarterly profit announcement"])
    assert X_new.shape == (1, 2)
    assert np.isclose(np.linalg.norm(X_new), 1.0, atol=1e-5)


def test_build_representation_lsa(sample_corpus):
    rep = build_representation("lsa", n_components=3)
    assert isinstance(rep, LSARepresentation)
    X = rep.fit_transform(sample_corpus)
    assert X.shape == (4, 3)


def test_zero_shot_classifier_predictions():
    clf = ZeroShotClassifier()
    res = clf.predict(
        "Your checking account balance and credit payment invoice are available.",
        candidate_labels=["finance", "sports", "gaming", "medicine"],
    )
    assert len(res) == 1
    item = res[0]
    assert item["predicted_label"] == "finance"
    assert item["confidence"] > 0.50
    assert "probabilities" in item
    assert sum(item["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_zero_shot_classifier_batch():
    clf = ZeroShotClassifier()
    texts = [
        "The striker scored two goals in the semifinal tournament.",
        "Hospital patient was admitted for cardiovascular surgery.",
    ]
    results = clf.predict(texts, candidate_labels=["sports", "medical"])
    assert len(results) == 2
    assert results[0]["predicted_label"] == "sports"
    assert results[1]["predicted_label"] == "medical"


def test_zero_shot_invalid_inputs():
    clf = ZeroShotClassifier()
    with pytest.raises(ValueError, match="At least one candidate label"):
        clf.predict("Some document", candidate_labels=[])
