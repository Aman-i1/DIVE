"""Console output that behaves identically on Windows, macOS, and Linux.

Three portability problems this module solves:

1. Windows terminals frequently use a legacy code page (cp1252) where the
   notebook's ``✓ ✗ ★ →`` characters raise ``UnicodeEncodeError`` mid-run. The
   :class:`Console` probes the active stream encoding once and falls back to
   ASCII equivalents when the fancy glyphs cannot be encoded.
2. Progress reporting during a long training run: ``Console.progress`` prints
   elapsed time against the time budget so the tool never goes silent.
3. Colour. ANSI escapes are emitted only when the destination is a real
   terminal that will interpret them, so piping to a file, redirecting into a
   test runner, or running under a dumb terminal all yield clean plain text.
   On Windows, virtual-terminal processing is enabled once at import so that
   escapes render instead of printing as literal ``←[38;5;209m`` garbage.
"""

from __future__ import annotations

import os
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


class Style:
    """ANSI style codes.

    256-colour codes rather than truecolour: every terminal that supports
    colour at all supports these, and they degrade gracefully in 16-colour
    terminals. Chosen to sit near the Claude Code palette - a coral accent
    against muted greys.
    """

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    ITALIC = "\x1b[3m"
    UNDERLINE = "\x1b[4m"

    ACCENT = "\x1b[38;5;209m"      # coral - headings, the dive brand colour
    ACCENT_DEEP = "\x1b[38;5;173m"  # muted coral - rules and dividers
    SUCCESS = "\x1b[38;5;42m"      # green
    WARN = "\x1b[38;5;214m"        # amber
    ERROR = "\x1b[38;5;203m"       # soft red
    INFO = "\x1b[38;5;39m"         # blue - keys and labels
    MUTED = "\x1b[38;5;245m"       # grey - timings, secondary detail
    BRIGHT = "\x1b[38;5;255m"      # near-white - values worth reading
    MAGENTA = "\x1b[38;5;176m"     # violet - model names


def _enable_windows_vt() -> None:
    """Turn on virtual-terminal processing so cmd.exe renders ANSI escapes.

    Windows 10+ consoles support ANSI but do not enable it by default for
    every host. Without this, escapes are printed literally. Any failure is
    ignored: the colour check below independently probes for a TTY, and a
    terminal that refuses VT mode simply keeps the plain-text path.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 = STD_OUTPUT_HANDLE, -12 = STD_ERROR_HANDLE
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


_enable_windows_vt()


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


def _stream_supports_color(stream: TextIO) -> bool:
    """Return True when ANSI escapes written to ``stream`` will be interpreted.

    Honours the two de-facto standards - ``NO_COLOR`` (any value disables) and
    ``FORCE_COLOR`` (any value enables, used by CI systems that render colour
    without presenting a TTY) - then falls back to a TTY check. Redirected
    output, pipes, and ``CliRunner`` in the test suite all land on plain text.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class Console:
    """Minimal, dependency-free console writer with verbosity and colour."""

    def __init__(
        self,
        verbose: bool = True,
        quiet: bool = False,
        stream: Optional[TextIO] = None,
        color: Optional[bool] = None,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        # Stored as given: when None, the stream is resolved at write time so
        # that redirections applied after construction (CliRunner, contextlib
        # .redirect_stdout, piping) are honoured.
        self._explicit_stream = stream
        # None means "decide from the stream"; True/False force the choice.
        self._color_override = color
        self._start = time.time()

    @property
    def _stream(self) -> TextIO:
        return self._explicit_stream if self._explicit_stream is not None else sys.stdout

    @property
    def _unicode(self) -> bool:
        return _stream_supports_unicode(self._stream)

    @property
    def color(self) -> bool:
        """True when this console is currently emitting ANSI escapes."""
        if self._color_override is not None:
            return self._color_override
        return _stream_supports_color(self._stream)

    # -- styling --------------------------------------------------------
    def paint(self, text: str, *styles: str) -> str:
        """Wrap ``text`` in ANSI styles, or return it unchanged without colour.

        Every colour decision in the package funnels through here so that a
        single TTY check governs the whole output surface.
        """
        if not text or not styles or not self.color:
            return text
        return f"{''.join(styles)}{text}{Style.RESET}"

    # -- symbol handling ------------------------------------------------
    def symbol(self, key: str) -> str:
        """Return the best representation of a named symbol for this stream.

        Deliberately uncoloured: callers embed the result in larger strings and
        several tests match on the surrounding text. Colour is applied by the
        level helpers (:meth:`success`, :meth:`warn`, :meth:`error`) and by
        :meth:`status_symbol` for callers that want a coloured mark inline.
        """
        fancy, plain = _SYMBOLS.get(key, ("", ""))
        return fancy if self._unicode else plain

    def status_symbol(self, key: str) -> str:
        """Return a symbol pre-coloured by its meaning."""
        tint = {
            "ok": Style.SUCCESS,
            "fail": Style.ERROR,
            "warn": Style.WARN,
            "up": Style.SUCCESS,
            "star": Style.ACCENT,
            "bullet": Style.ACCENT_DEEP,
            "arrow": Style.MUTED,
        }.get(key)
        symbol = self.symbol(key)
        return self.paint(symbol, tint) if tint else symbol

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
            self._write(f"{self.status_symbol('ok')} {self.paint(message, Style.SUCCESS)}")

    def warn(self, message: str) -> None:
        """Warnings survive quiet mode - they are the point of quiet runs."""
        self._write(f"{self.status_symbol('warn')} {self.paint(message, Style.WARN)}")

    def error(self, message: str) -> None:
        """Write an error to stderr.

        Routed through ``click.echo`` when click is importable so that output
        ordering is preserved under test runners and redirected streams; falls
        back to a direct stderr write otherwise.

        Colour is decided against stderr rather than the console's own stream,
        because that is where the text actually lands.
        """
        colorize = self._color_override
        if colorize is None:
            colorize = _stream_supports_color(sys.stderr)
        symbol = self.symbol("fail")
        if colorize:
            text = f"{Style.ERROR}{symbol} {message}{Style.RESET}"
        else:
            text = f"{symbol} {message}"
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
            self._write(
                f"{self.paint(prefix, Style.ACCENT_DEEP)} "
                f"{self.paint(title, Style.ACCENT, Style.BOLD)} "
                f"{self.paint(char * filler, Style.ACCENT_DEEP)}"
            )
        else:
            self._write(self.paint(char * width, Style.ACCENT_DEEP))

    def step(self, index: int, total: int, message: str) -> None:
        """Print a numbered pipeline step, e.g. ``[2/6] Feature engineering``."""
        marker = self.paint(f"[{index}/{total}]", Style.ACCENT, Style.BOLD)
        self.info(f"{marker} {self.paint(message, Style.BRIGHT)}")

    def progress(self, message: str, elapsed: float, budget: Optional[float]) -> None:
        """Print a progress line carrying elapsed time against the budget."""
        if budget and budget > 0:
            detail = f"({elapsed:.0f}s / {budget:.0f}s budget)"
        else:
            detail = f"({elapsed:.0f}s elapsed)"
        self.info(f"    {message}  {self.paint(detail, Style.MUTED)}")

    def kv(self, key: str, value: Any, width: int = 22) -> None:
        """Print an aligned ``key : value`` pair."""
        label = self.paint(f"{str(key):<{width}}", Style.INFO)
        self.info(f"  {label}: {self.paint(str(value), Style.BRIGHT)}")

    def model_result(
        self,
        index: int,
        total: int,
        name: str,
        score_label: str,
        score: float,
        took: float,
        elapsed: float,
        budget: float,
    ) -> None:
        """Print one leaderboard-bound model result during training."""
        self.info(
            f"  {self.status_symbol('ok')} "
            f"{self.paint(f'[{index}/{total}]', Style.MUTED)} "
            f"{self.paint(f'{name:<20}', Style.MAGENTA)} "
            f"{self.paint(f'{score_label}={score:.4f}', Style.SUCCESS, Style.BOLD)}  "
            f"{self.paint(f'({took:.1f}s, {elapsed:.0f}s/{budget:.0f}s budget)', Style.MUTED)}"
        )

    def model_failed(self, index: int, total: int, name: str, reason: str) -> None:
        """Print one model that could not be trained."""
        self.info(
            f"  {self.status_symbol('fail')} "
            f"{self.paint(f'[{index}/{total}]', Style.MUTED)} "
            f"{self.paint(f'{name:<20}', Style.MAGENTA)} "
            f"{self.paint('failed: ' + reason, Style.ERROR)}"
        )

    def table(self, frame: Any, max_rows: int = 15, highlight_first: bool = False) -> None:
        """Print a pandas DataFrame without truncating columns.

        With ``highlight_first`` the header is tinted and the top row - the
        winning model, since the leaderboard arrives pre-sorted - is bolded.
        """
        if self.quiet or frame is None:
            return
        try:
            import pandas as pd

            with pd.option_context(
                "display.max_columns", None,
                "display.width", 200,
                "display.float_format", lambda v: f"{v:.4f}",
            ):
                rendered = frame.head(max_rows).to_string(index=False)
        except Exception:
            self._write(str(frame))
            return

        if not (highlight_first and self.color):
            self._write(rendered)
            return

        lines = rendered.split("\n")
        if lines:
            lines[0] = self.paint(lines[0], Style.INFO, Style.BOLD)
        if len(lines) > 1:
            lines[1] = self.paint(lines[1], Style.SUCCESS, Style.BOLD)
        self._write("\n".join(lines))

    def banner(self, title: str, subtitle: str = "") -> None:
        """Print the branded header shown at the top of a command."""
        if self.quiet:
            return
        self._write("")
        self._write(f"  {self.paint(title, Style.ACCENT, Style.BOLD)}")
        if subtitle:
            self._write(f"  {self.paint(subtitle, Style.MUTED)}")
        self._write("")


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
