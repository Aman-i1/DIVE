"""Text Normalization & Sanitization - `dive/nlp/preprocessing/normalizer.py`.

Provides modular, non-destructive text cleaning transformations:
- Unicode normalization (NFKD, NFC)
- Accent stripping (Unicode, ASCII)
- Whitespace collapsing & trimming
- Optional HTML tag and entity stripping
- Optional URL, email, and mention removal
- Optional punctuation removal
- Sequence length truncation (character and word limits)
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import List, Optional, Sequence


# Compiled regular expressions for fast, efficient text cleaning
_RE_HTML = re.compile(r"<[^>]+>", re.IGNORECASE)
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_RE_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_RE_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_RE_EMOJIS = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class TextNormalizer:
    """Configurable text normalizer that cleans and standardizes raw strings."""

    def __init__(
        self,
        lowercase: bool = True,
        unicode_form: Optional[str] = "NFKD",
        strip_accents: Optional[str] = "unicode",
        collapse_whitespace: bool = True,
        remove_html: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_emojis: bool = False,
        remove_punctuation: bool = False,
        max_char_length: Optional[int] = None,
        max_word_length: Optional[int] = None,
    ) -> None:
        self.lowercase = lowercase
        self.unicode_form = unicode_form
        self.strip_accents = strip_accents
        self.collapse_whitespace = collapse_whitespace
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_emojis = remove_emojis
        self.remove_punctuation = remove_punctuation
        self.max_char_length = max_char_length
        self.max_word_length = max_word_length

    def normalize_text(self, text: str) -> str:
        """Apply configured normalization transformations to a single string."""
        if not text or not isinstance(text, str):
            return ""

        s = text

        # 1. Unicode normalization
        if self.unicode_form:
            s = unicodedata.normalize(self.unicode_form, s)

        # 2. Accent stripping
        if self.strip_accents == "unicode":
            s = "".join(c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn")
        elif self.strip_accents == "ascii":
            s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")

        # 3. HTML stripping & unescaping
        if self.remove_html:
            s = html.unescape(s)
            s = _RE_HTML.sub(" ", s)

        # 4. URLs and Emails
        if self.remove_urls:
            s = _RE_URL.sub(" ", s)
        if self.remove_emails:
            s = _RE_EMAIL.sub(" ", s)

        # 5. Emojis
        if self.remove_emojis:
            s = _RE_EMOJIS.sub(" ", s)

        # 6. Punctuation
        if self.remove_punctuation:
            s = _RE_PUNCTUATION.sub(" ", s)

        # 7. Lowercasing
        if self.lowercase:
            s = s.lower()

        # 8. Whitespace collapsing
        if self.collapse_whitespace:
            s = _RE_WHITESPACE.sub(" ", s).strip()

        # 9. Length truncation
        if self.max_word_length is not None:
            words = s.split()
            if len(words) > self.max_word_length:
                s = " ".join(words[: self.max_word_length])

        if self.max_char_length is not None and len(s) > self.max_char_length:
            s = s[: self.max_char_length].rstrip()

        return s

    def transform(self, texts: Sequence[str]) -> List[str]:
        """Normalize a sequence of text strings."""
        return [self.normalize_text(t) for t in texts]
