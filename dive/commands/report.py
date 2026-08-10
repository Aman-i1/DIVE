"""Implementations of ``dive report``, ``dive explain``, and ``dive docs``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from dive.core import Dive
from dive.exceptions import DataError, ModelError
from dive.utils.io import ensure_dir, resolve_path
from dive.utils.logging import Console


def run_report(
    console: Console, model_path: str, output_path: str, open_browser: bool = False
) -> Path:
    """Render the full HTML report for a saved model."""
    from dive.reporting import build_html_report, generate_plots

    dive = Dive.load(model_path)
    console.rule("dive report")
    console.kv("Model", Path(str(model_path)).name)
    console.kv("Best model", dive.best_model_name_)

    validation = _load_sidecar_validation(model_path)
    if validation is not None:
        console.kv("Crosschecks", f"{len(validation.checks)} loaded from the training run")

    # A reloaded model has no holdout split in memory, so diagnostics plots that
    # need it are unavailable; plots saved during training are reused instead.
    plots_dir = resolve_path(model_path).parent / "plots"
    if not plots_dir.exists():
        plots_dir = None
        console.info(
            "      (no plots/ directory beside the model - the report will omit figures)"
        )

    written = build_html_report(
        dive, output_path, validation=validation, plots_dir=plots_dir, console=console
    )
    console.success(f"Report written to {written}")
    if open_browser:
        _open_in_browser(written, console)
    return written


def run_explain(
    console: Console,
    model_path: str,
    output_path: Optional[str] = None,
    open_browser: bool = False,
) -> Optional[Path]:
    """Explain a saved model: pipeline narrative, importances, repro code."""
    from dive.reporting import build_explanation, build_explanation_html

    dive = Dive.load(model_path)

    if output_path:
        written = build_explanation_html(dive, output_path)
        console.rule("dive explain")
        console.kv("Model", Path(str(model_path)).name)
        console.success(f"Explanation written to {written}")
        if open_browser:
            _open_in_browser(written, console)
        return written

    explanation = build_explanation(dive)
    console.rule(f"How {dive.best_model_name_} was built")
    for step in explanation["steps"]:
        console.print("")
        console.print(f"  {step['title']}")
        for line in step["lines"]:
            console.print(f"    {console.symbol('bullet')} {line}")

    importances = dive.feature_importances(top_n=15)
    if importances is not None and not importances.empty:
        console.print("")
        console.print("  Top features")
        maximum = float(importances["importance"].max()) or 1.0
        for _, row in importances.iterrows():
            bar = "#" * max(1, int(28 * float(row["importance"]) / maximum))
            console.print(f"    {str(row['feature'])[:34]:<34} {bar} {row['importance']:.4f}")

    console.print("")
    console.print(
        f"  Tip: dive explain --model {model_path} --output explanation.html"
    )
    console.print("       writes this plus standalone reproduction code as HTML.")
    return None


def run_docs(
    console: Console,
    output_dir: Optional[str] = None,
    serve: bool = False,
    port: int = 8000,
    no_browser: bool = False,
) -> Path:
    """Generate the documentation site, then open or serve it."""
    from dive.reporting import generate_docs

    destination = ensure_dir(output_dir or _default_docs_dir())
    pages = generate_docs(destination, console=console)

    console.rule("dive docs")
    console.kv("Location", destination)
    for page in pages:
        console.kv("  page", page.name)

    index = destination / "index.html"
    if serve:
        _serve(destination, port, index, console, open_browser=not no_browser)
    elif not no_browser:
        _open_in_browser(index, console)
    else:
        console.success(f"Documentation written to {index}")
    return destination


def _default_docs_dir() -> Path:
    """Prefer the repo's docs/ directory; fall back to a user-writable location.

    An installed package may live in a read-only site-packages tree, so writing
    next to the source is attempted first and abandoned quietly if not possible.
    """
    from dive import __file__ as package_file

    repo_docs = Path(package_file).resolve().parent.parent / "docs"
    try:
        repo_docs.mkdir(parents=True, exist_ok=True)
        probe = repo_docs / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return repo_docs
    except OSError:
        import tempfile

        return Path(tempfile.gettempdir()) / "dive-docs"


def _open_in_browser(path: Path, console: Console) -> None:
    """Open a local file in the default browser, cross-platform."""
    import webbrowser

    url = path.resolve().as_uri()
    try:
        if webbrowser.open(url):
            console.success(f"Opened {path.name} in your browser.")
            return
    except Exception:
        pass
    console.print(f"  Could not open a browser automatically. Open this file:\n  {path}")


def _serve(
    directory: Path, port: int, index: Path, console: Console, open_browser: bool
) -> None:
    """Serve the docs directory over HTTP until interrupted."""
    import functools
    import http.server
    import socketserver

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    try:
        with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
            url = f"http://127.0.0.1:{port}/index.html"
            console.success(f"Serving documentation at {url}")
            console.print("  Press Ctrl+C to stop.")
            if open_browser:
                import webbrowser

                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                console.print("")
                console.success("Stopped serving documentation.")
    except OSError as exc:
        raise DataError(
            f"Could not serve documentation on port {port}.",
            f"The operating system reported: {exc}. Try a different --port, "
            f"or open the file directly: {index}",
        ) from exc


def _load_sidecar_validation(model_path: str) -> Optional[Any]:
    """Rebuild the ValidationReport written next to the model during training."""
    from dive.validation import CheckResult, ValidationReport

    candidate = resolve_path(model_path).parent / "validation.json"
    if not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        report = ValidationReport(context=payload.get("context", {}))
        for entry in payload.get("checks", []):
            report.add(
                CheckResult(
                    name=entry.get("name", "?"),
                    status=entry.get("status", "SKIP"),
                    summary=entry.get("summary", ""),
                    details=list(entry.get("details", [])),
                    metrics=dict(entry.get("metrics", {})),
                )
            )
        return report
    except Exception:
        return None
