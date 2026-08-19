"""Phase 4 Tests - DIVE NLP Text Preprocessing Pipeline.

Verifies:
1. Text normalization: Unicode decomposition, accent stripping, and whitespace handling.
2. HTML tag stripping and URL/email filtering.
3. Emoji and punctuation removal.
4. Sequence length truncation (character limits and word limits).
5. NLPPreprocessorProtocol compliance and fit/transform lifecycle.
6. Non-destructive raw passthrough mode (NLPPreprocessor.raw()) for transformer models.
7. Custom stop-word filtering and configuration integration with NLPPreprocessingConfig.
"""

from __future__ import annotations

import pytest

from dive.nlp.config import NLPPreprocessingConfig
from dive.nlp.interfaces import NLPPreprocessorProtocol
from dive.nlp.preprocessing import (
    NLPPreprocessor,
    TextNormalizer,
    build_nlp_preprocessor,
)


def test_text_normalizer_basic_cleaning() -> None:
    """Verify whitespace collapsing, trimming, and lowercasing."""
    normalizer = TextNormalizer(lowercase=True, collapse_whitespace=True)
    raw = "   This   is   a   MESSY \t\n  String.   "
    cleaned = normalizer.normalize_text(raw)
    assert cleaned == "this is a messy string."


def test_text_normalizer_accents_and_unicode() -> None:
    """Verify Unicode accent stripping and normalization."""
    normalizer = TextNormalizer(strip_accents="unicode", lowercase=True)
    raw = "Café au lait with naïve façade and crème brûlée"
    cleaned = normalizer.normalize_text(raw)
    assert cleaned == "cafe au lait with naive facade and creme brulee"


def test_text_normalizer_html_and_urls() -> None:
    """Verify stripping of HTML tags, entities, and URLs/emails."""
    normalizer = TextNormalizer(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        lowercase=True,
        collapse_whitespace=True,
    )
    raw = (
        "<div><h1>Release Notice</h1><p>Read docs at https://github.com/Aman-i1/DIVE "
        "or contact support@example.com for help &amp; assistance.</p></div>"
    )
    cleaned = normalizer.normalize_text(raw)
    assert "<div>" not in cleaned
    assert "https://" not in cleaned
    assert "support@example.com" not in cleaned
    assert "help & assistance" in cleaned
    assert cleaned.startswith("release notice")


def test_text_normalizer_punctuation_and_emojis() -> None:
    """Verify removal of punctuation and emojis."""
    normalizer = TextNormalizer(
        remove_emojis=True,
        remove_punctuation=True,
        lowercase=True,
        collapse_whitespace=True,
    )
    raw = "Incredible platform! 🚀 High throughput, zero latency! 💯🎉"
    cleaned = normalizer.normalize_text(raw)
    assert "🚀" not in cleaned
    assert "💯" not in cleaned
    assert "!" not in cleaned
    assert "," not in cleaned
    assert cleaned == "incredible platform high throughput zero latency"


def test_text_normalizer_length_truncation() -> None:
    """Verify character and word sequence length truncation."""
    # Word truncation
    word_norm = TextNormalizer(max_word_length=3)
    assert word_norm.normalize_text("one two three four five") == "one two three"

    # Char truncation
    char_norm = TextNormalizer(max_char_length=10)
    assert char_norm.normalize_text("1234567890abcdefgh") == "1234567890"


def test_nlp_preprocessor_protocol_and_lifecycle() -> None:
    """Verify NLPPreprocessorProtocol compliance and fit/transform lifecycle."""
    prep = build_nlp_preprocessor(lowercase=True, remove_html=True)
    assert isinstance(prep, NLPPreprocessorProtocol)

    texts = [
        "<p>Document ONE</p>",
        "<span>Document TWO</span>",
    ]
    transformed = prep.fit_transform(texts)
    assert prep.fitted_ is True
    assert transformed == ["document one", "document two"]


def test_nlp_preprocessor_raw_passthrough() -> None:
    """Verify non-destructive raw passthrough mode preserves casing, punctuation, and URLs."""
    raw_prep = NLPPreprocessor.raw()
    texts = [
        "CaseSensitive text with https://url.org and Emoji 😊! \n Next line.",
    ]
    transformed = raw_prep.transform(texts)
    # Exact original string preserved
    assert transformed == texts


def test_nlp_preprocessor_with_custom_stopwords() -> None:
    """Verify optional custom stopword filtering."""
    prep = NLPPreprocessor(
        lowercase=True,
        custom_stopwords=["is", "a", "the", "with"],
    )
    texts = ["This is a test document with sample words"]
    transformed = prep.transform(texts)
    assert transformed == ["this test document sample words"]


def test_nlp_preprocessor_with_config() -> None:
    """Verify initialization from declarative NLPPreprocessingConfig."""
    cfg = NLPPreprocessingConfig(
        lowercase=True,
        remove_urls=True,
        remove_html=True,
        max_seq_length=4,
    )
    prep = NLPPreprocessor(config=cfg)
    texts = ["<b>Click</b> https://example.com for more information on DIVE AutoML"]
    transformed = prep.transform(texts)
    assert "https://" not in transformed[0]
    assert "<b>" not in transformed[0]
    assert len(transformed[0].split()) <= 4
