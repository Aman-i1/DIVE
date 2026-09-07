"""Shared terminal-report rendering for every DIVE capability domain.

Every ``render() -> str`` in the package builds its report through
``ReportBuilder`` so that ``dive ml``, ``dive nlp``, ``dive dl`` and ``dive cv``
speak one visual grammar::

    -- TITLE -----------------------------------------------------------

      Key                   : value

    SECTION NAME
    ------------
      Column      Role      Dtype
      ---------------------------
      text        TEXT      object

      [PASS] a check that passed
      - a bullet
      Label            [####......]  42.0%

      Next: dive ml train data.csv

Two invariants make this worth centralising:

1. Every glyph resolves through :class:`~dive.utils.logging.Console`'s symbol
   table, so a legacy code page (cp1252 on Windows) degrades to ASCII instead of
   printing ``?``. Modules must never embed a raw glyph in an f-string.
2. Table column widths derive from content, so values are not silently truncated
   by a hardcoded ``:<22``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from dive.utils.logging import Console, Style, get_console

# Canonical status vocabulary. Before this existed the package emitted five
# competing sets ([PASS]/[HIGH]/[DRIFT]/[TOP]/[WARNING]/[APPROVED]/...), which is
# why two domains never looked alike. Tokens stay plain bracketed ASCII so that
# existing greps and test assertions keep matching.
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
INFO = "INFO"
TOP = "TOP"

CANONICAL_STATUSES = (PASS, WARN, FAIL, SKIP, INFO, TOP)

# Legacy spellings encountered across the codebase, mapped onto the canonical set
# so migration is a rename rather than a redesign.
_STATUS_ALIASES: Dict[str, str] = {
    "PASS": PASS, "PASSED": PASS, "OK": PASS, "SAFE": PASS, "GOOD": PASS,
    "APPROVED": PASS, "LOW": PASS, "STABLE": PASS, "BALANCED": PASS,
    "CLEAN": PASS, "HEALTHY": PASS, "NEGLIGIBLE": PASS,
    "WARN": WARN, "WARNING": WARN, "CAUTION": WARN, "MEDIUM": WARN,
    "DRIFT": WARN, "MODERATE": WARN, "REVIEW": WARN,
    "PASS_WITH_WARNINGS": WARN, "PASS WITH WARNINGS": WARN,
    "CONDITIONAL": WARN, "DEGRADE": WARN, "SIGNIFICANT": WARN,
    "FAIL": FAIL, "FAILED": FAIL, "HIGH": FAIL, "HIGH RISK": FAIL,
    "CRITICAL": FAIL, "REJECTED": FAIL, "UNSAFE": FAIL, "SEVERE": FAIL,
    "BLOCK": FAIL, "BLOCKED": FAIL, "REFUSE": FAIL, "ERROR": FAIL,
    "SKIP": SKIP, "SKIPPED": SKIP, "N/A": SKIP, "NA": SKIP, "NONE": SKIP,
    "NOT_RUN": SKIP, "UNAVAILABLE": SKIP,
    "INFO": INFO, "NOTE": INFO, "INCONCLUSIVE": INFO, "UNKNOWN": INFO,
    "TOP": TOP, "BEST": TOP, "CHAMPION": TOP,
}

_STATUS_TINTS: Dict[str, str] = {
    PASS: Style.EMERALD,
    WARN: Style.WARN,
    FAIL: Style.ERROR,
    SKIP: Style.MUTED,
    INFO: Style.INFO,
    TOP: Style.GOLD,
}

_STATUS_SYMBOLS: Dict[str, str] = {
    PASS: "ok",
    WARN: "warn",
    FAIL: "fail",
    SKIP: "bullet",
    INFO: "bullet",
    TOP: "star",
}

# Key column width for kv rows. Pinned at 22 because report text such as
# "Total Documents       : 2" is asserted verbatim by the test suite.
KEY_WIDTH = 22

# Longest a single table cell may grow before it is ellipsised.
_MAX_CELL = 44


def normalize_status(token: Any) -> str:
    """Map any legacy status spelling onto the canonical vocabulary.

    Unknown tokens fall through to :data:`INFO` rather than raising, so a report
    with an unexpected status still renders.
    """
    if token is None:
        return INFO
    key = str(token).strip().upper()
    return _STATUS_ALIASES.get(key, INFO)


def _truncate(text: str, limit: int = _MAX_CELL) -> str:
    if limit <= 3 or len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class ReportBuilder:
    """Accumulates report lines and renders them as one string.

    The public contract across the package is ``render() -> str``, so this builds
    text rather than writing to a stream. Pass it to ``Console.report(...)`` (or
    ``console.print(builder.build())``) to display it.

    Every mutator returns ``self`` so reports read as a single chained
    expression.
    """

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        console: Optional[Console] = None,
        width: int = 72,
    ) -> None:
        self.console = console if console is not None else get_console()
        self.width = max(24, int(width))
        self._lines: List[str] = []
        if title:
            self.title(title, subtitle)

    # -- primitives -----------------------------------------------------

    def _symbol(self, key: str) -> str:
        return self.console.symbol(key)

    def _paint(self, text: str, *styles: str) -> str:
        return self.console.paint(text, *styles)

    def raw(self, text: str = "") -> "ReportBuilder":
        """Append a line verbatim. Escape hatch; prefer the typed helpers."""
        self._lines.append(text)
        return self

    def blank(self) -> "ReportBuilder":
        """Append a blank line, collapsing consecutive blanks."""
        if self._lines and self._lines[-1] != "":
            self._lines.append("")
        return self

    # -- structure ------------------------------------------------------

    def title(self, text: str, subtitle: str = "") -> "ReportBuilder":
        """Render the report header: ``-- TITLE --------------``."""
        rule = self._symbol("rule")
        lead = rule * 2
        remaining = self.width - len(lead) - len(text) - 2
        trail = rule * max(3, remaining)
        self._lines.append(
            f"{self._paint(lead, Style.CYAN)} "
            f"{self._paint(text, Style.BRIGHT_CYAN, Style.BOLD)} "
            f"{self._paint(trail, Style.CYAN)}"
        )
        if subtitle:
            self._lines.append(f"  {self._paint(subtitle, Style.MUTED)}")
        return self

    def section(self, text: str) -> "ReportBuilder":
        """Start a section: a blank line, the title, then a dashed underline.

        The title is emitted verbatim (no case transformation) because several
        section names are asserted exactly by the test suite.
        """
        self.blank()
        self._lines.append(self._paint(text, Style.VIOLET, Style.BOLD))
        self._lines.append(self._paint(self._symbol("rule") * len(text), Style.MUTED))
        return self

    # -- content --------------------------------------------------------

    def kv(self, key: Any, value: Any, width: int = KEY_WIDTH) -> "ReportBuilder":
        """Append an aligned ``key : value`` row, indented two spaces."""
        label = f"{str(key):<{width}}"
        self._lines.append(
            f"  {self._paint(label, Style.VIOLET)}: {self._paint(str(value), Style.BRIGHT)}"
        )
        return self

    def kvs(self, pairs: Any, width: int = KEY_WIDTH) -> "ReportBuilder":
        """Append many kv rows from a mapping or an iterable of pairs."""
        items = pairs.items() if hasattr(pairs, "items") else pairs
        for key, value in items:
            self.kv(key, value, width=width)
        return self

    def status(self, token: Any, message: str) -> "ReportBuilder":
        """Append a ``[PASS] message`` row using the canonical vocabulary."""
        canonical = normalize_status(token)
        tag = self._paint(f"[{canonical}]", _STATUS_TINTS.get(canonical, Style.MUTED), Style.BOLD)
        self._lines.append(f"  {tag} {message}")
        return self

    def status_symbol(self, token: Any, message: str) -> "ReportBuilder":
        """Like :meth:`status` but leads with a glyph instead of a bracket tag."""
        canonical = normalize_status(token)
        glyph = self._symbol(_STATUS_SYMBOLS.get(canonical, "bullet"))
        tinted = self._paint(glyph, _STATUS_TINTS.get(canonical, Style.MUTED))
        self._lines.append(f"  {tinted} {message}")
        return self

    def bullet(self, text: str, indent: int = 2) -> "ReportBuilder":
        """Append a bulleted line using the symbol table's bullet."""
        self._lines.append(f"{' ' * indent}{self._symbol('bullet')} {text}")
        return self

    def bullets(self, items: Sequence[Any], indent: int = 2) -> "ReportBuilder":
        for item in items:
            self.bullet(str(item), indent=indent)
        return self

    def numbered(self, items: Sequence[Any], indent: int = 2) -> "ReportBuilder":
        for index, item in enumerate(items, start=1):
            self._lines.append(f"{' ' * indent}{index}. {item}")
        return self

    def note(self, text: str, indent: int = 2) -> "ReportBuilder":
        """Append a de-emphasised aside."""
        self._lines.append(f"{' ' * indent}{self._paint(text, Style.MUTED)}")
        return self

    def bar(
        self,
        label: str,
        value: float,
        total: float = 1.0,
        segments: int = 24,
        suffix: Optional[str] = None,
    ) -> "ReportBuilder":
        """Append a horizontal meter, e.g. ``label  [####....]  42.0%``.

        Replaces the hand-rolled block-glyph bars, which printed as ``?`` on a
        legacy code page.
        """
        try:
            fraction = 0.0 if not total else float(value) / float(total)
        except (TypeError, ValueError, ZeroDivisionError):
            fraction = 0.0
        fraction = min(1.0, max(0.0, fraction))
        filled = int(round(fraction * segments))
        meter = self._symbol("bar_fill") * filled + self._symbol("bar_empty") * (segments - filled)
        text = suffix if suffix is not None else f"{fraction * 100:.1f}%"
        self._lines.append(
            f"  {str(label):<{KEY_WIDTH}} "
            f"[{self._paint(meter, Style.CYAN)}] {self._paint(text, Style.BRIGHT)}"
        )
        return self

    def table(
        self,
        columns: Sequence[Any],
        rows: Sequence[Sequence[Any]],
        max_rows: Optional[int] = None,
        indent: int = 2,
        highlight_first: bool = False,
    ) -> "ReportBuilder":
        """Append a fixed-width table sized from its own content.

        Widths come from the widest cell in each column rather than a hardcoded
        pad, which is what used to truncate long column names mid-word.
        """
        headers = [str(column) for column in columns]
        body = [[_truncate(str(cell)) for cell in row] for row in rows]

        shown = body
        hidden = 0
        if max_rows is not None and len(body) > max_rows:
            shown = body[:max_rows]
            hidden = len(body) - max_rows

        widths = [len(header) for header in headers]
        for row in shown:
            for index, cell in enumerate(row[: len(widths)]):
                widths[index] = max(widths[index], len(cell))

        pad = " " * indent

        def compose(cells: Sequence[str]) -> str:
            parts = [
                f"{cell:<{widths[index]}}"
                for index, cell in enumerate(cells[: len(widths)])
            ]
            return (pad + "  ".join(parts)).rstrip()

        self._lines.append(self._paint(compose(headers), Style.CYAN, Style.BOLD))
        total = sum(widths) + 2 * max(0, len(widths) - 1)
        self._lines.append(pad + self._paint(self._symbol("rule") * total, Style.MUTED))

        for position, row in enumerate(shown):
            line = compose(row)
            if highlight_first and position == 0:
                line = self._paint(line, Style.EMERALD, Style.BOLD)
            self._lines.append(line)

        if hidden:
            self.note(f"... (+{hidden} more row{'s' if hidden != 1 else ''})", indent=indent)
        return self

    def dataframe(
        self,
        frame: Any,
        max_rows: int = 15,
        indent: int = 2,
        highlight_first: bool = False,
    ) -> "ReportBuilder":
        """Append a pandas DataFrame through :meth:`table`.

        Falls back to ``str(frame)`` if the object is not frame-like, so a report
        never fails to render because of its data.
        """
        if frame is None:
            return self
        try:
            columns = list(frame.columns)
            rows = frame.astype(object).where(frame.notna(), "-").values.tolist()
        except Exception:
            return self.raw(str(frame))
        formatted = [
            [f"{cell:.4f}" if isinstance(cell, float) else cell for cell in row]
            for row in rows
        ]
        return self.table(
            columns,
            formatted,
            max_rows=max_rows,
            indent=indent,
            highlight_first=highlight_first,
        )

    def next_steps(self, commands: Sequence[str]) -> "ReportBuilder":
        """Append the canonical footer.

        Replaces the four competing forms found across the package
        (``Next:`` / ``Tip:`` / ``RECOMMENDED NEXT STEPS:`` / ``ACTION PLAN:``).
        A single command renders inline; several render as a numbered list.
        """
        items = [str(command) for command in commands if str(command).strip()]
        if not items:
            return self
        self.blank()
        if len(items) == 1:
            self._lines.append(f"  {self._paint('Next:', Style.CYAN, Style.BOLD)} {items[0]}")
            return self
        self._lines.append(self._paint("Next steps:", Style.CYAN, Style.BOLD))
        return self.numbered(items, indent=4)

    # -- output ---------------------------------------------------------

    def build(self) -> str:
        """Return the assembled report, without a trailing newline."""
        while self._lines and self._lines[-1] == "":
            self._lines.pop()
        return "\n".join(self._lines)

    # Reports are frequently interpolated directly into f-strings.
    render = build
    __str__ = build


def report(
    title: str = "",
    subtitle: str = "",
    console: Optional[Console] = None,
    width: int = 72,
) -> ReportBuilder:
    """Convenience constructor: ``report("TITLE").kv(...).build()``."""
    return ReportBuilder(title=title, subtitle=subtitle, console=console, width=width)
