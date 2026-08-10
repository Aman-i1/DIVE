"""Shared helpers: console output, file I/O, and optional-dependency handling."""

from __future__ import annotations

from dive.utils.logging import Console, get_console
from dive.utils.optional import (
    OPTIONAL_PACKAGES,
    dependency_report,
    is_available,
    load_optional,
)

__all__ = [
    "Console",
    "get_console",
    "OPTIONAL_PACKAGES",
    "dependency_report",
    "is_available",
    "load_optional",
]
