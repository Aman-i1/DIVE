"""Console output that behaves identically on Windows, macOS, and Linux.

Two portability problems this module solves:

1. Windows terminals frequently use a legacy code page (cp1252) where the
   notebook's ``✓ ✗ ★ →`` characters raise ``UnicodeEncodeError`` mid-run. The
   :class:`Console` probes the active stream encoding once and falls back to
   ASCII equivalents when the fancy glyphs cannot be encoded.
2. Progress reporting during a long training run: ``Console.progress`` prints
   elapsed time against the time budget so the tool never goes silent.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional, TextIO

# Preferred glyph -> ASCII fallback.
_SYMBOLS: Dict[str, tuple] = {
    "ok": ("✓", "[ok]"),          # check mark
    "fail": ("✗", "[fail]"),      # ballot X
    "warn": ("⚠", "[warn]"),      # warning sign
    "up": ("↑", "^"),             # up arrow
    "arrow": ("→", "->"),         # right arrow
    "star": ("★", "*"),           # star
    "bullet": ("•", "-"),         # bullet
    "rule": ("─", "-"),           # box drawing horizontal
}


def _stream_supports_unicode(stream: TextIO) -> bool:
    """Return True when the stream's encoding can represent our glyph set."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        "".join(fancy for fancy, _ in _SYMBOLS.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class Console:
    """Minimal, dependency-free console writer with verbosity control."""

    def __init__(
        self,
        verbose: bool = True,
        quiet: bool = False,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        # Stored as given: when None, the stream is resolved at write time so
        # that redirections applied after construction (CliRunner, contextlib
        # .redirect_stdout, piping) are honoured.
        self._explicit_stream = stream
        self._start = time.time()

    @property
    def _stream(self) -> TextIO:
        return self._explicit_stream if self._explicit_stream is not None else sys.stdout

    @property
    def _unicode(self) -> bool:
        return _stream_supports_unicode(self._stream)

    # -- symbol handling ------------------------------------------------
    def symbol(self, key: str) -> str:
        """Return the best representation of a named symbol for this stream."""
        fancy, plain = _SYMBOLS.get(key, ("", ""))
        return fancy if self._unicode else plain

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text + "\n")
        except UnicodeEncodeError:
            # Last-resort guard: strip anything the terminal cannot render.
            encoding = getattr(self._stream, "encoding", "ascii") or "ascii"
            safe = text.encode(encoding, errors="replace").decode(encoding)
            self._stream.write(safe + "\n")
        self._stream.flush()

    # -- output levels --------------------------------------------------
    def print(self, message: str = "") -> None:
        """Print unconditionally unless quiet mode is active."""
        if not self.quiet:
            self._write(message)

    def info(self, message: str) -> None:
        """Print only in verbose mode."""
        if self.verbose and not self.quiet:
            self._write(message)

    def success(self, message: str) -> None:
        if not self.quiet:
            self._write(f"{self.symbol('ok')} {message}")

    def warn(self, message: str) -> None:
        """Warnings survive quiet mode - they are the point of quiet runs."""
        self._write(f"{self.symbol('warn')} {message}")

    def error(self, message: str) -> None:
        """Write an error to stderr.

        Routed through ``click.echo`` when click is importable so that output
        ordering is preserved under test runners and redirected streams; falls
        back to a direct stderr write otherwise.
        """
        text = f"{self.symbol('fail')} {message}"
        try:
            import click

            click.echo(text, err=True)
            return
        except ImportError:
            pass
        try:
            sys.stderr.write(text + "\n")
            sys.stderr.flush()
        except UnicodeEncodeError:
            sys.stderr.write(f"[fail] {message}\n")
            sys.stderr.flush()

    def rule(self, title: str = "", width: int = 66) -> None:
        """Print a horizontal rule, optionally with an inline title."""
        if self.quiet:
            return
        char = self.symbol("rule")
        if title:
            prefix = char * 3
            filler = max(0, width - len(title) - 5)
            self._write(f"{prefix} {title} {char * filler}")
        else:
            self._write(char * width)

    def step(self, index: int, total: int, message: str) -> None:
        """Print a numbered pipeline step, e.g. ``[2/6] Feature engineering``."""
        self.info(f"[{index}/{total}] {message}")

    def progress(self, message: str, elapsed: float, budget: Optional[float]) -> None:
        """Print a progress line carrying elapsed time against the budget."""
        if budget and budget > 0:
            self.info(f"    {message}  ({elapsed:.0f}s / {budget:.0f}s budget)")
        else:
            self.info(f"    {message}  ({elapsed:.0f}s elapsed)")

    def kv(self, key: str, value: Any, width: int = 22) -> None:
        """Print an aligned ``key : value`` pair."""
        self.info(f"  {str(key):<{width}}: {value}")

    def table(self, frame: Any, max_rows: int = 15) -> None:
        """Print a pandas DataFrame without truncating columns."""
        if self.quiet or frame is None:
            return
        try:
            import pandas as pd

            with pd.option_context(
                "display.max_columns", None,
                "display.width", 200,
                "display.float_format", lambda v: f"{v:.4f}",
            ):
                self._write(frame.head(max_rows).to_string(index=False))
        except Exception:
            self._write(str(frame))


_DEFAULT: Optional[Console] = None


def get_console(verbose: bool = True, quiet: bool = False) -> Console:
    """Return the process-wide console, creating it on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Console(verbose=verbose, quiet=quiet)
    else:
        _DEFAULT.verbose = verbose
        _DEFAULT.quiet = quiet
    return _DEFAULT
