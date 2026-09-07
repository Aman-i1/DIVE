"""Guards on the shared output grammar.

Two classes of bug shipped once and must not return:

1. **Rich markup printed literally.** ``dive/commands/nlp.py`` emitted 37 tags of
   the form ``[bold cyan]...[/bold cyan]`` although ``rich`` is not a dependency
   and :class:`~dive.utils.logging.Console` is a hand-written ANSI writer that
   does not parse markup, so users saw the tags.
2. **Raw Unicode glyphs mangled to ``?``.** Windows consoles here report
   ``cp1252``; a glyph embedded directly in an f-string bypasses the symbol
   table's ASCII fallback and ``Console._write`` rewrites it to ``?``. That is
   how ``dive ml info`` came to print its title box as ``????????????????``.

These tests assert the *mechanism*, not a golden string, so they keep working as
reports evolve.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import pytest

import dive
import dive.utils.logging as logging_module
from dive.data_intelligence import DataIntelligence, summarize_profile
from dive.info import DatasetInspector
from dive.resources import ResourceManager
from dive.utils.logging import Console
from dive.utils.report import CANONICAL_STATUSES, ReportBuilder, normalize_status

PACKAGE_ROOT = Path(dive.__file__).resolve().parent
SOURCES = sorted(PACKAGE_ROOT.rglob("*.py"))


# ----------------------------------------------------------------------
# 1. no Rich markup anywhere in the package
# ----------------------------------------------------------------------
_RICH_STYLES = (
    "bold", "dim", "italic", "underline", "blink", "reverse", "strike",
    "red", "green", "yellow", "blue", "magenta", "cyan", "white", "black",
    "bright_red", "bright_green", "bright_yellow", "bright_blue",
    "bright_magenta", "bright_cyan", "bright_white", "bright_black",
)
_RICH_TAG = re.compile(r"\[/?(?:" + "|".join(_RICH_STYLES) + r")(?:\s+[a-z_]+)*\]")


def test_sources_contain_no_rich_markup():
    offenders = []
    for source in SOURCES:
        text = source.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            match = _RICH_TAG.search(line)
            if match:
                relative = source.relative_to(PACKAGE_ROOT)
                offenders.append(f"{relative}:{number}: {match.group(0)}")
    assert not offenders, (
        "Rich-style markup is printed literally by dive.utils.logging.Console:\n"
        + "\n".join(offenders)
    )


# ----------------------------------------------------------------------
# 2. no raw glyphs outside the files allowed to hold them
# ----------------------------------------------------------------------
# Relative paths permitted to contain non-ASCII characters:
#   utils/logging.py - the glyph -> ASCII fallback registry and spinner frames
#   reporting.py, audit.py - HTML and reportlab PDF bodies, never a terminal
_GLYPH_ALLOWED = {
    Path("utils/logging.py"),
    Path("reporting.py"),
    Path("audit.py"),
}


def test_no_raw_glyphs_outside_the_symbol_registry():
    offenders = []
    for source in SOURCES:
        relative = Path(source.relative_to(PACKAGE_ROOT).as_posix())
        if relative in _GLYPH_ALLOWED:
            continue
        text = source.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if not line.isascii():
                glyphs = "".join(sorted({char for char in line if not char.isascii()}))
                offenders.append(f"{relative}:{number}: {glyphs!r}")
    assert not offenders, (
        "Register the glyph in dive.utils.logging._SYMBOLS and emit it via "
        "Console.symbol()/ReportBuilder instead of embedding it:\n" + "\n".join(offenders)
    )


# ----------------------------------------------------------------------
# 3. rendered reports survive a cp1252 console
# ----------------------------------------------------------------------
class _LegacyStream(io.StringIO):
    """A stream that claims the Windows cp1252 code page, like a real console."""

    encoding = "cp1252"

    def isatty(self) -> bool:
        return False


@pytest.fixture
def legacy_console(monkeypatch) -> Console:
    """Force every ``get_console()`` in the process onto a cp1252 stream.

    Renderers build their ``ReportBuilder`` without passing a console, so they
    resolve the module singleton; patching it is what makes this deterministic
    rather than dependent on whatever encoding pytest's capture layer uses.
    """
    console = Console(verbose=True, quiet=False, stream=_LegacyStream(), color=False)
    monkeypatch.setattr(logging_module, "_DEFAULT", console)
    return console


@pytest.fixture
def frame() -> pd.DataFrame:
    # 60 rows so the profiler's cardinality heuristic reads `label` as a
    # classification target (2 distinct / 60 rows is under its 5% ceiling).
    size = 60
    return pd.DataFrame(
        {
            "customer_id": list(range(1, size + 1)),
            "spend": [round(3.5 * index, 2) for index in range(size)],
            "tier": ["a", "b", "c"] * (size // 3),
            "label": [0, 1] * (size // 2),
        }
    )


def _assert_legacy_safe(text: str, what: str) -> None:
    """A report built for a cp1252 console must already be pure ASCII.

    Mangling happens in ``Console._write``, so anything non-ASCII surviving into
    ``build()`` is a glyph that skipped the symbol table.
    """
    assert text, f"{what} rendered nothing"
    text.encode("cp1252")  # raises UnicodeEncodeError if a glyph slipped through
    assert text.isascii(), f"{what} emitted non-ASCII on a cp1252 console"


def test_report_builder_is_ascii_on_a_legacy_console(legacy_console):
    """Every builder primitive must degrade, including the meter and the table."""
    builder = ReportBuilder("DIVE OUTPUT GRAMMAR", "subtitle", console=legacy_console)
    builder.kv("Key", "value")
    builder.kvs({"Another": 1, "Third": 2.5})
    builder.section("SECTION")
    for token in CANONICAL_STATUSES:
        builder.status(token, f"{token.lower()} message")
        builder.status_symbol(token, f"{token.lower()} with a glyph")
    builder.bullets(["first", "second"])
    builder.numbered(["one", "two"])
    builder.note("an aside")
    builder.bar("Meter", 0.42)
    builder.bar("Scaled", 3.0, 4.0, suffix="3/4")
    builder.table(["Column", "Role"], [["text", "TEXT"], ["label", "TARGET"]])
    builder.dataframe(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    builder.next_steps(["dive ml train data.csv --target label"])

    output = builder.build()
    _assert_legacy_safe(output, "ReportBuilder")
    for token in CANONICAL_STATUSES:
        assert f"[{token}]" in output


def test_renderers_are_ascii_on_a_legacy_console(legacy_console, frame):
    """Real renderers, resolving the singleton console, must degrade too."""
    inspection = DatasetInspector().inspect(frame)
    _assert_legacy_safe(inspection.render(), "DatasetInfoReport.render")

    profile = DataIntelligence(target="label").analyze(frame)
    _assert_legacy_safe("\n".join(summarize_profile(profile)), "summarize_profile")

    manager = ResourceManager(time_budget_sec=60.0)
    _assert_legacy_safe(
        manager.get_system_resources().render(), "SystemResources.render"
    )
    plan = manager.create_plan(frame, ["RandomForest", "KNN", "MLP"], mode="fast")
    _assert_legacy_safe(plan.render(), "AutoMLResourcePlan.render")
    _assert_legacy_safe(manager.assess(1.0, label="tiny").render(), "CapabilityVerdict")


def test_console_report_accepts_builders_and_strings(legacy_console):
    builder = ReportBuilder("TITLE", console=legacy_console)
    builder.kv("Key", "value")
    legacy_console.report(builder)
    legacy_console.report("plain string")

    written = legacy_console._stream.getvalue()
    assert "TITLE" in written
    assert "plain string" in written
    _assert_legacy_safe(written, "Console.report")


# ----------------------------------------------------------------------
# 4. one status vocabulary
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "legacy, canonical",
    [
        ("PASSED", "PASS"),
        ("SAFE", "PASS"),
        ("approved", "PASS"),
        ("WARNING", "WARN"),
        ("SIGNIFICANT_DRIFT", "INFO"),  # module-specific spellings are not guessed
        ("HIGH", "FAIL"),
        ("REJECTED", "FAIL"),
        ("REFUSE", "FAIL"),
        ("skipped", "SKIP"),
        ("CHAMPION", "TOP"),
        ("something nobody defined", "INFO"),
        (None, "INFO"),
    ],
)
def test_status_aliases_collapse_onto_the_canonical_vocabulary(legacy, canonical):
    assert normalize_status(legacy) == canonical
