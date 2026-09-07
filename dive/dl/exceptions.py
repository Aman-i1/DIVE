"""Exception hierarchy for DIVE Deep Learning.

Mirrors :mod:`dive.nlp.exceptions`: every error inherits from :class:`DLError`,
which inherits from :class:`dive.exceptions.DiveError`, so a caller can catch the
whole domain, one domain, or the whole platform.
"""

from __future__ import annotations

from dive.exceptions import DiveError


class DLError(DiveError):
    """Base exception for all DIVE deep-learning errors."""


class DLConfigError(DLError):
    """Raised when a deep-learning configuration or parameter is invalid."""


class DLDataError(DLError):
    """Raised when input data is missing, empty, unreadable, or the wrong shape."""


class DLModalityError(DLError):
    """Raised when a modality is unknown or cannot handle the given input."""


class DLBackendError(DLError):
    """Raised when no backend can serve the request.

    A *missing* backend is not this error: an absent ``torch`` degrades to the
    documented fallback. This is raised only when even the fallback cannot run -
    e.g. a modality whose fallback needs Pillow and Pillow is absent too.
    """


class DLTrainingError(DLError):
    """Raised when training fails or cannot produce a usable model."""


class DLInferenceError(DLError):
    """Raised when prediction or scoring fails."""
