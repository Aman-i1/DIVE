"""Console resolution shared by every ``dive`` subcommand.

The root group stores the global flags on ``ctx.obj``; a subcommand that calls a
bare ``get_console()`` instead of reading them is how ``dive --quiet nlp info``
came to print everything anyway. One helper, used by every command module, so the
bug cannot come back one command at a time.

``ctx`` is duck-typed rather than annotated as ``click.Context`` so this module
stays importable without click in the dependency graph of library code.
"""

from __future__ import annotations

from typing import Any

from dive.utils.logging import Console, get_console


def console_from(ctx: Any) -> Console:
    """Return the process console configured by the root command's flags.

    Tolerates a missing or empty ``ctx.obj`` (direct invocation, tests) by
    falling back to the default verbose console.
    """
    obj = getattr(ctx, "obj", None) or {}
    quiet = bool(obj.get("quiet", False))
    return get_console(verbose=not quiet, quiet=quiet)


def traceback_requested(ctx: Any) -> bool:
    """True when the user asked for a full traceback on failure."""
    obj = getattr(ctx, "obj", None) or {}
    return bool(obj.get("traceback", False))
