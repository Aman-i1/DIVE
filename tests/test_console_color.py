"""Tests for console colour gating.

Colour must never reach a non-terminal: piped output, redirected files, and the
CliRunner in this suite all have to receive clean text, or every downstream
string assertion becomes escape-sensitive.
"""

from __future__ import annotations

import io

import pytest

from dive.utils.logging import Console, Style, _stream_supports_color


class _FakeStream(io.StringIO):
    """StringIO with a controllable isatty() and a readable encoding."""

    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    @property
    def encoding(self) -> str:
        return "utf-8"

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def _clean_color_env(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)


def test_no_color_on_a_pipe():
    console = Console(stream=_FakeStream(tty=False))
    assert console.color is False
    console.success("done")
    assert "\x1b[" not in console._stream.getvalue()


def test_color_on_a_terminal():
    console = Console(stream=_FakeStream(tty=True))
    assert console.color is True
    console.success("done")
    assert "\x1b[" in console._stream.getvalue()


def test_no_color_env_var_wins(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _stream_supports_color(_FakeStream(tty=True)) is False


def test_force_color_enables_without_a_tty(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _stream_supports_color(_FakeStream(tty=False)) is True


def test_dumb_terminal_gets_no_color(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    assert _stream_supports_color(_FakeStream(tty=True)) is False


def test_paint_is_identity_without_color():
    console = Console(stream=_FakeStream(tty=False))
    assert console.paint("hello", Style.ACCENT) == "hello"


def test_paint_wraps_and_resets_with_color():
    console = Console(stream=_FakeStream(tty=True))
    painted = console.paint("hello", Style.ACCENT)
    assert painted.startswith(Style.ACCENT)
    assert painted.endswith(Style.RESET)
    assert "hello" in painted


def test_symbol_is_never_coloured():
    """Callers embed symbol() in larger strings; colour there would leak."""
    console = Console(stream=_FakeStream(tty=True))
    assert "\x1b[" not in console.symbol("ok")
    assert "\x1b[" in console.status_symbol("ok")


def test_plain_text_is_identical_with_and_without_color():
    """Stripping escapes from a coloured run must reproduce the plain run."""
    import re

    plain, colored = Console(stream=_FakeStream(False)), Console(stream=_FakeStream(True))
    for console in (plain, colored):
        console.rule("Training")
        console.step(1, 6, "Profiling dataset")
        console.kv("Rows", 150)
        console.success("Done")
        console.warn("Careful")

    stripped = re.sub(r"\x1b\[[0-9;]*m", "", colored._stream.getvalue())
    assert stripped == plain._stream.getvalue()
