"""Phase 6 Tests - DIVE NLP Feature Engine & Pluggable Representations.

Verifies:
1. Character n-gram feature extraction (subword/morphological signal, typo tolerance).
2. Word + character joint feature unions (horizontal sparse matrix concatenation).
3. BM25 probabilistic relevance representation (Okapi BM25 sparse scoring).
4. Pluggable representation integration with NLPPipeline.
5. End-to-end training with train_baseline(representation_type=...).
6. NLPRepresentationProtocol compliance across all representation engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from dive.nlp.features import (
    BM25Representation,
    CharNGramRepresentation,
    CountRepresentation,
    TFIDFRepresentation,
    WordCharUnionRepresentation,
    build_representation,
)
from dive.nlp.interfaces import NLPRepresentationProtocol
from dive.nlp.pipeline import NLPPipeline
from dive.nlp.models.baselines import build_baseline_model
from dive.nlp.training import train_baseline


@pytest.fixture
def mini_corpus() -> List[str]:
    return [
        "Machine learning platforms enable autonomous model discovery.",
        "Deep neural networks process complex natural language text.",
        "Statistical algorithms optimize classification accuracy and latency.",
        "Automated feature engineering extracts valuable numerical representations.",
    ]


def test_char_ngram_representation(mini_corpus: List[str]) -> None:
    """Verify character n-gram extraction, typo tolerance, and protocol compliance."""
    char_rep = CharNGramRepresentation(ngram_range=(3, 4), min_df=1)
    assert isinstance(char_rep, NLPRepresentationProtocol)

    X_char = char_rep.fit_transform(mini_corpus)
    assert sp.issparse(X_char)
    assert X_char.shape[0] == 4
    assert X_char.shape[1] > 20

    # Typo tolerance: 'machne lerning' has strong char n-gram overlap with 'machine learning'
    query_orig = char_rep.transform(["machine learning"])
    query_typo = char_rep.transform(["machne lerning"])

    # Dot product between original and typo representation is strictly positive
    overlap = float((query_orig @ query_typo.T).toarray()[0, 0])
    assert overlap > 0.3


def test_word_char_union_representation(mini_corpus: List[str]) -> None:
    """Verify joint word + character sparse matrix concatenation."""
    union_rep = WordCharUnionRepresentation(
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 4),
    )
    assert isinstance(union_rep, NLPRepresentationProtocol)

    X_union = union_rep.fit_transform(mini_corpus)
    assert sp.issparse(X_union)
    assert X_union.shape[0] == 4
    assert X_union.shape[1] == union_rep.n_features_

    # Verify feature names have word__ and char__ prefixes
    names = union_rep.feature_names_
    assert any(n.startswith("word__") for n in names)
    assert any(n.startswith("char__") for n in names)


def test_bm25_representation(mini_corpus: List[str]) -> None:
    """Verify Okapi BM25 probabilistic relevance representation."""
    bm25_rep = BM25Representation(k1=1.5, b=0.75, min_df=1)
    assert isinstance(bm25_rep, NLPRepresentationProtocol)

    X_bm25 = bm25_rep.fit_transform(mini_corpus)
    assert sp.issparse(X_bm25)
    assert X_bm25.shape[0] == 4
    assert X_bm25.nnz > 0

    # Rare terms receive higher BM25 weights than common terms
    X_arr = X_bm25.toarray()
    assert np.all(X_arr >= 0.0)


@pytest.mark.parametrize(
    "rep_type,rep_cls",
    [
        ("tfidf", TFIDFRepresentation),
        ("char_ngrams", CharNGramRepresentation),
        ("word_char_union", WordCharUnionRepresentation),
        ("bm25", BM25Representation),
        ("count", CountRepresentation),
    ],
)
def test_build_representation_factory(rep_type: str, rep_cls: Any) -> None:
    """Verify factory instantiation across all representation families."""
    rep = build_representation(representation_type=rep_type)
    assert isinstance(rep, rep_cls)
    assert isinstance(rep, NLPRepresentationProtocol)


def test_pipeline_with_custom_representations() -> None:
    """Verify NLPPipeline training with CharNGram and WordCharUnion representations."""
    texts = [
        "Incredible positive service and great experience",
        "Loved the great product quality, super fast",
        "Terrible and broken on arrival, worst item",
        "Awful experience and useless customer service",
    ]
    labels = ["pos", "pos", "neg", "neg"]

    # 1. Pipeline with WordCharUnion
    union_pipe = NLPPipeline(
        estimator=build_baseline_model("LogisticRegression"),
        representation=WordCharUnionRepresentation(),
        model_name="UnionPipeline",
    )
    union_pipe.fit(texts, labels)
    preds = union_pipe.predict(["Loved this great quality!"])
    assert preds[0] == "pos"

    # 2. Pipeline with BM25
    bm25_pipe = NLPPipeline(
        estimator=build_baseline_model("MultinomialNB"),
        representation=BM25Representation(min_df=1),
        model_name="BM25Pipeline",
    )
    bm25_pipe.fit(texts, labels)
    preds_bm25 = bm25_pipe.predict(["Awful and broken"])
    assert preds_bm25[0] == "neg"


@pytest.mark.parametrize("rep_type", ["tfidf", "char_ngrams", "word_char_union", "bm25"])
def test_train_baseline_with_all_representation_types(rep_type: str) -> None:
    """Verify end-to-end baseline training across all feature representations."""
    df = pd.DataFrame(
        {
            "text": [
                "Outstanding build quality, fast delivery, loved it!",
                "Great experience with customer support, loved the team.",
                "Awesome performance, superb durability, great product.",
                "Delightful purchase, loved everything, fast service.",
                "Worst purchase ever made, broken on arrival, awful.",
                "Terrible customer service, completely useless and broken.",
                "Horrible experience, awful quality, worst product.",
                "Broke within one hour, useless device, terrible.",
            ],
            "label": ["pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"],
        }
    )

    predictor, metrics = train_baseline(
        data=df,
        text_column="text",
        target_column="label",
        model_name="LogisticRegression",
        representation_type=rep_type,
        test_size=0.25,
        random_state=42,
    )

    assert predictor.model_name == "LogisticRegression"
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.5

    p = predictor.predict("Loved the great quality!")
    assert p[0] == "pos"
