"""Exception hierarchy for dive.

Every error raised deliberately by this package derives from :class:`DiveError`.
The CLI catches ``DiveError`` and prints ``str(exc)`` as a clean message rather
than a traceback, so error text is written for a human reading a terminal.
"""

from __future__ import annotations


class DiveError(Exception):
    """Base class for all expected, user-facing dive errors.

    Parameters
    ----------
    message:
        The primary, one-line problem statement.
    hint:
        Optional follow-up telling the user how to fix it. Rendered on its own
        line by the CLI.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        self.message = message
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n{self.hint}"
        return self.message


class DataError(DiveError):
    """The input dataset is missing, unreadable, empty, or malformed."""


class TargetError(DiveError):
    """The requested target column is absent or unusable for modelling."""


class SchemaError(DiveError):
    """Incoming data does not match the schema a model was trained on."""


class ConfigError(DiveError):
    """A config file or CLI option combination is invalid."""


class ModelError(DiveError):
    """A model artifact is missing, unreadable, or incompatible."""


class TrainingError(DiveError):
    """Training could not produce a usable model."""


class ValidationError(DiveError):
    """A validation/crosscheck run failed hard (used by the validate command)."""
