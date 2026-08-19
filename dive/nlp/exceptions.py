"""Exception hierarchy for DIVE NLP.

All NLP exceptions inherit from :class:`NLPError`, which inherits from
:class:`dive.exceptions.DiveError`.
"""

from __future__ import annotations

from dive.exceptions import DiveError


class NLPError(DiveError):
    """Base exception for all DIVE NLP errors."""


class TextDataError(NLPError):
    """Raised when text input data is missing, empty, or unreadable."""


class NLPConfigError(NLPError):
    """Raised when an NLP configuration or parameter specification is invalid."""


class NLPModelError(NLPError):
    """Raised when an NLP model artifact is missing, unreadable, or invalid."""


class NLPTrainingError(NLPError):
    """Raised when NLP training fails or cannot produce a valid model."""


class NLPInferenceError(NLPError):
    """Raised when NLP inference or prediction encounters an error."""


class TokenizationError(NLPError):
    """Raised when tokenization fails on input text."""


class VocabularyError(NLPError):
    """Raised when vocabulary extraction, matching, or lookup fails."""


class TaskNotSupportedError(NLPError):
    """Raised when the requested NLP task type is unsupported or unrecognized."""
