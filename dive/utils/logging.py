"""Console output with aesthetic ANSI styling, npm-style spinners, and cross-platform fallback.

Portability & aesthetics features:
1. Windows terminal Virtual Terminal Processing activation & ASCII fallback.
2. Animated npm-style loading spinners for long-running operations.
3. Rich 256-color aesthetic palette (Cyan, Violet, Emerald, Coral, Gold, Magenta).
4. Aligned key-value pairs, card layouts, and progress tracking.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import Any, Dict, List, Optional, TextIO

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
    "box_top": ("╔", "+"),
    "box_bot": ("╚", "+"),
    "box_vert": ("║", "|"),
    "sparkle": ("✨", "*"),
    "lightning": ("⚡", "!"),
}

# npm-style spinner frames
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_SPINNER_ASCII = ["|", "/", "-", "\\"]


class Style:
    """ANSI style codes - 256-color vibrant aesthetic palette."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    ITALIC = "\x1b[3m"
    UNDERLINE = "\x1b[4m"

    ACCENT = "\x1b[38;5;209m"       # Coral brand accent
    ACCENT_DEEP = "\x1b[38;5;173m"   # Deep coral divider
    CYAN = "\x1b[38;5;51m"          # Electric cyan
    BRIGHT_CYAN = "\x1b[38;5;87m"   # Bright mint cyan
    VIOLET = "\x1b[38;5;141m"       # Soft violet
    EMERALD = "\x1b[38;5;48m"       # Vibrant green
    SUCCESS = "\x1b[38;5;42m"       # Mint green
    WARN = "\x1b[38;5;214m"         # Amber gold
    ERROR = "\x1b[38;5;203m"        # Coral red
    INFO = "\x1b[38;5;39m"          # Sky blue
    GOLD = "\x1b[38;5;220m"         # Bright gold
    PINK = "\x1b[38;5;213m"         # Neon pink
    MUTED = "\x1b[38;5;245m"        # Medium grey
    BRIGHT = "\x1b[38;5;255m"       # Near-white
    MAGENTA = "\x1b[38;5;176m"      # Soft magenta


def _enable_windows_vt() -> None:
    """Turn on virtual-terminal processing on Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


_enable_windows_vt()


def _stream_supports_unicode(stream: TextIO) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        "".join(fancy for fancy, _ in _SYMBOLS.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _stream_supports_color(stream: TextIO) -> bool:
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


class Spinner:
    """npm-style background thread animated spinner context manager."""

    def __init__(self, console: Console, message: str = "Processing...") -> None:
        self.console = console
        self.message = message
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def _animate(self) -> None:
        frames = _SPINNER_FRAMES if self.console._unicode else _SPINNER_ASCII
        idx = 0
        while self.running:
            frame = frames[idx % len(frames)]
            idx += 1
            if self.console.color:
                colored_frame = f"{Style.CYAN}{Style.BOLD}{frame}{Style.RESET}"
                msg = f"{Style.BRIGHT}{self.message}{Style.RESET}"
                text = f"\r{colored_frame} {msg}"
            else:
                text = f"\r{frame} {self.message}"

            try:
                sys.stdout.write(text)
                sys.stdout.flush()
            except Exception:
                pass
            time.sleep(0.08)

        # Clear line on exit
        try:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()
        except Exception:
            pass

    def __enter__(self) -> "Spinner":
        if self.console.verbose and not self.console.quiet and _stream_supports_color(sys.stdout):
            self.running = True
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()
        else:
            self.console.info(f"... {self.message}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=0.2)


class Console:
    """Minimal console writer with verbosity, aesthetic colors & spinners."""

    def __init__(
        self,
        verbose: bool = True,
        quiet: bool = False,
        stream: Optional[TextIO] = None,
        color: Optional[bool] = None,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self._explicit_stream = stream
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
        if self._color_override is not None:
            return self._color_override
        return _stream_supports_color(self._stream)

    def paint(self, text: str, *styles: str) -> str:
        if not text or not styles or not self.color:
            return text
        return f"{''.join(styles)}{text}{Style.RESET}"

    def symbol(self, key: str) -> str:
        fancy, plain = _SYMBOLS.get(key, ("", ""))
        return fancy if self._unicode else plain

    def status_symbol(self, key: str) -> str:
        tint = {
            "ok": Style.EMERALD,
            "fail": Style.ERROR,
            "warn": Style.WARN,
            "up": Style.SUCCESS,
            "star": Style.GOLD,
            "bullet": Style.CYAN,
            "arrow": Style.VIOLET,
            "sparkle": Style.GOLD,
            "lightning": Style.CYAN,
        }.get(key)
        symbol = self.symbol(key)
        return self.paint(symbol, tint) if tint else symbol

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text + "\n")
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", "ascii") or "ascii"
            safe = text.encode(encoding, errors="replace").decode(encoding)
            self._stream.write(safe + "\n")
        self._stream.flush()

    def print(self, message: str = "") -> None:
        if not self.quiet:
            self._write(message)

    def info(self, message: str) -> None:
        if self.verbose and not self.quiet:
            self._write(message)

    def success(self, message: str) -> None:
        if not self.quiet:
            self._write(f"{self.status_symbol('ok')} {self.paint(message, Style.EMERALD, Style.BOLD)}")

    def warn(self, message: str) -> None:
        self._write(f"{self.status_symbol('warn')} {self.paint(message, Style.WARN)}")

    def error(self, message: str) -> None:
        colorize = self._color_override if self._color_override is not None else _stream_supports_color(sys.stderr)
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
        if self.quiet:
            return
        char = self.symbol("rule")
        if title:
            prefix = char * 3
            filler = max(0, width - len(title) - 5)
            self._write(
                f"{self.paint(prefix, Style.CYAN)} "
                f"{self.paint(title, Style.BRIGHT_CYAN, Style.BOLD)} "
                f"{self.paint(char * filler, Style.CYAN)}"
            )
        else:
            self._write(self.paint(char * width, Style.CYAN))

    def step(self, index: int, total: int, message: str) -> None:
        marker = self.paint(f"[{index}/{total}]", Style.VIOLET, Style.BOLD)
        self.info(f"{marker} {self.paint(message, Style.BRIGHT)}")

    def progress(self, message: str, elapsed: float, budget: Optional[float]) -> None:
        if budget and budget > 0:
            detail = f"({elapsed:.0f}s / {budget:.0f}s budget)"
        else:
            detail = f"({elapsed:.0f}s elapsed)"
        self.info(f"    {message}  {self.paint(detail, Style.MUTED)}")

    def kv(self, key: str, value: Any, width: int = 22) -> None:
        label = self.paint(f"{str(key):<{width}}", Style.VIOLET)
        self.info(f"  {label}: {self.paint(str(value), Style.BRIGHT)}")

    def spinner(self, message: str = "Processing...") -> Spinner:
        """Return an npm-style loading spinner context manager."""
        return Spinner(self, message=message)

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
        self.info(
            f"  {self.status_symbol('ok')} "
            f"{self.paint(f'[{index}/{total}]', Style.MUTED)} "
            f"{self.paint(f'{name:<20}', Style.VIOLET, Style.BOLD)} "
            f"{self.paint(f'{score_label}={score:.4f}', Style.EMERALD, Style.BOLD)}  "
            f"{self.paint(f'({took:.1f}s, {elapsed:.0f}s/{budget:.0f}s budget)', Style.MUTED)}"
        )

    def model_failed(self, index: int, total: int, name: str, reason: str) -> None:
        self.info(
            f"  {self.status_symbol('fail')} "
            f"{self.paint(f'[{index}/{total}]', Style.MUTED)} "
            f"{self.paint(f'{name:<20}', Style.MAGENTA)} "
            f"{self.paint('failed: ' + reason, Style.ERROR)}"
        )

    def table(self, frame: Any, max_rows: int = 15, highlight_first: bool = False) -> None:
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
            lines[0] = self.paint(lines[0], Style.CYAN, Style.BOLD)
        if len(lines) > 1:
            lines[1] = self.paint(lines[1], Style.EMERALD, Style.BOLD)
        self._write("\n".join(lines))

    def banner(self, title: str, subtitle: str = "") -> None:
        if self.quiet:
            return
        self._write("")
        self._write(f"  {self.status_symbol('lightning')} {self.paint(title, Style.CYAN, Style.BOLD)}")
        if subtitle:
            self._write(f"     {self.paint(subtitle, Style.MUTED)}")
        self._write("")


_DEFAULT: Optional[Console] = None


def get_console(verbose: bool = True, quiet: bool = False) -> Console:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Console(verbose=verbose, quiet=quiet)
    else:
        _DEFAULT.verbose = verbose
        _DEFAULT.quiet = quiet
    return _DEFAULT
