"""The ``dive`` command-line interface.

Design rules enforced throughout this module:

* Every command validates its inputs *before* doing expensive work.
* Expected failures raise :class:`dive.exceptions.DiveError` and are rendered
  by :func:`main` as a clean message plus a hint - never a traceback.
* Unexpected failures print a one-line summary and point at ``--traceback``.
* Exit codes: 0 success, 1 user/data error, 2 unexpected internal error.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import click

from dive import __version__
from dive.exceptions import DiveError, ConfigError, DataError, ModelError
from dive.utils.io import (
    ensure_dir,
    load_config,
    load_dataframe,
    resolve_path,
    save_dataframe,
    validate_target,
)
from dive.utils.logging import Console, get_console
from dive.utils.optional import dependency_report, missing_summary


def _quiet_third_party_warnings() -> None:
    """Silence library deprecation chatter that users cannot act on.

    Applied only at the CLI boundary - importing dive as a library leaves the
    caller's warning filters alone. Convergence and deprecation notices from
    scikit-learn during a 12-model sweep are noise: the run already reports
    which models succeeded and which failed.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass

MODES = ("fast", "balanced", "competition")

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 100,
}


# ----------------------------------------------------------------------
def _console(ctx: click.Context) -> Console:
    """Return the console configured by the root command's global flags."""
    obj = ctx.obj or {}
    return get_console(verbose=not obj.get("quiet", False), quiet=obj.get("quiet", False))


def _apply_config(
    config_path: Optional[str],
    overrides: Dict[str, Any],
    allowed: set,
) -> Dict[str, Any]:
    """Merge a YAML/JSON config under explicit CLI flags.

    Explicit CLI flags always win. Unknown keys raise rather than being ignored,
    so a typo in a config file is reported instead of silently doing nothing.
    """
    if not config_path:
        return overrides
    config = load_config(config_path)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown option(s) in config file: {', '.join(unknown)}",
            f"Valid keys are: {', '.join(sorted(allowed))}",
        )
    merged = dict(config)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    return merged


def _echo_optional_dependency_notice(console: Console) -> None:
    summary = missing_summary()
    if summary:
        console.warn(summary)
        console.print("")


# ----------------------------------------------------------------------
def _version_callback(ctx: click.Context, param: Any, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"dive, version {__version__}")
    ctx.exit()


def _traceback_callback(ctx: click.Context, param: Any, value: bool) -> None:
    if value:
        ctx.ensure_object(dict)
        ctx.obj["traceback"] = True


class DiveCommand(click.Command):
    """Subcommand that also accepts the global flags.

    The CLI spec calls for ``--version`` and ``--help`` on every command, and
    ``--traceback`` is far more useful typed after the subcommand than before
    it. Click only puts group-level options on the group, so they are added
    here per command as well.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params.append(
            click.Option(
                ["-V", "--version"],
                is_flag=True,
                expose_value=False,
                is_eager=True,
                callback=_version_callback,
                help="Show the version and exit.",
            )
        )
        self.params.append(
            click.Option(
                ["--traceback", "show_traceback"],
                is_flag=True,
                expose_value=False,
                is_eager=True,
                callback=_traceback_callback,
                help="Print the full Python traceback on failure.",
            )
        )


# ----------------------------------------------------------------------
class DiveGroup(click.Group):
    """Command group that renders DiveError as a clean message.

    Error handling lives here rather than only in :func:`main` so that any
    entry point gets identical behaviour - including ``CliRunner`` in the test
    suite and ``python -m dive.cli``, both of which bypass ``main``.
    """

    command_class = DiveCommand

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except DiveError as exc:
            console = get_console()
            console.print("")
            console.error(str(exc))
            if ctx.obj and ctx.obj.get("traceback"):
                raise
            ctx.exit(1)


@click.group(cls=DiveGroup, context_settings=CONTEXT_SETTINGS, invoke_without_command=False)
@click.version_option(__version__, "-V", "--version", prog_name="dive")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output; show only warnings and errors.")
@click.option("--traceback", "show_traceback", is_flag=True, help="Print the full Python traceback on failure.")
@click.pass_context
def cli(ctx: click.Context, quiet: bool, show_traceback: bool) -> None:
    """Automated machine learning for tabular data.

    Train a model zoo on a CSV, validate a dataset before training, score new
    rows with a saved model, and generate HTML reports - all from the terminal.

    \b
    Typical flow:
      dive validate --data data.csv --target label
      dive train    --data data.csv --target label --mode fast --output ./out
      dive predict  --model ./out/model.pkl --data new.csv --output preds.csv
      dive report   --model ./out/model.pkl --output report.html
    """
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["traceback"] = show_traceback


# ----------------------------------------------------------------------
@cli.command("train")
@click.option("--data", "data_path", default=None, type=click.Path(), help="Path to the training data. Any tabular format pandas can read (.csv, .tsv, .parquet, .json, .xlsx, .ods, .feather, .orc, .dta, .sav, .h5, .html, .xml, optionally .gz/.zip compressed). Quote paths containing spaces. Required unless supplied via --config.")
@click.option("--target", "target", default=None, help="Name of the column to predict. Defaults to the last column.")
@click.option("--mode", type=click.Choice(MODES), default=None, help="fast = small zoo, no tuning. balanced = full zoo + tuning + stacking. competition = adds feature selection and extra models. [default: balanced]")
@click.option("--time-budget", type=float, default=None, help="Wall-clock seconds for the whole run. [default: 1800]")
@click.option("--output", "output_dir", default=None, type=click.Path(), help="Directory for the model, leaderboard, plots, and report. [default: ./dive_output]")
@click.option("--config", "config_path", default=None, type=click.Path(), help="YAML or JSON file supplying any of these options.")
@click.option("--test-size", type=float, default=None, help="Fraction of rows held out for evaluation. [default: 0.2]")
@click.option("--cv-folds", type=int, default=None, help="Cross-validation folds. Defaults to an adaptive value based on dataset size.")
@click.option("--random-state", type=int, default=None, help="Random seed for reproducible runs. [default: 42]")
@click.option("--time-series", is_flag=True, default=None, help="Split chronologically instead of randomly (no shuffling).")
@click.option("--no-plots", is_flag=True, default=None, help="Skip PNG plot generation.")
@click.option("--no-report", is_flag=True, default=None, help="Skip the HTML report.")
@click.option("--skip-validation", is_flag=True, default=None, help="Do not run the crosscheck suite before training.")
@click.pass_context
def train_command(ctx: click.Context, data_path: str, target: Optional[str], config_path: Optional[str], output_dir: Optional[str], **options: Any) -> None:
    """Train a model zoo and write the best model plus a full report.

    \b
    Examples:
      dive train --data sales.csv --target revenue --mode fast --output ./out
      dive train --data churn.parquet --target churned --time-budget 600
      dive train --data data.csv --target y --config my_settings.yaml
    """
    from dive.commands.train import run_train

    console = _console(ctx)
    allowed = {
        "data", "target", "mode", "time_budget", "output", "test_size",
        "cv_folds", "random_state", "time_series", "no_plots", "no_report",
        "skip_validation",
    }
    raw = dict(options)
    raw["data"] = data_path
    raw["target"] = target
    raw["output"] = output_dir
    settings = _apply_config(config_path, raw, allowed)

    _echo_optional_dependency_notice(console)
    run_train(console=console, **_train_settings(settings))


def _train_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply defaults and coerce config values into run_train's signature."""
    return {
        "data_path": settings.get("data"),
        "target": settings.get("target"),
        "mode": settings.get("mode") or "balanced",
        "time_budget": float(settings.get("time_budget") or 1800),
        "output_dir": settings.get("output") or "dive_output",
        "test_size": float(settings.get("test_size") or 0.2),
        "cv_folds": settings.get("cv_folds"),
        "random_state": int(settings.get("random_state") or 42),
        "time_series": bool(settings.get("time_series") or False),
        "make_plots": not bool(settings.get("no_plots") or False),
        "make_report": not bool(settings.get("no_report") or False),
        "run_validation": not bool(settings.get("skip_validation") or False),
    }


# ----------------------------------------------------------------------
@cli.command("predict")
@click.option("--model", "model_path", required=True, type=click.Path(), help="Path to a .pkl written by `dive train`.")
@click.option("--data", "data_path", required=True, type=click.Path(), help="Rows to score. Must carry the same feature columns used in training.")
@click.option("--output", "output_path", default=None, type=click.Path(), help="Where to write predictions. [default: predictions.csv]")
@click.option("--proba", is_flag=True, help="Also emit one probability column per class (classification only).")
@click.option("--include-input", is_flag=True, help="Copy the input columns into the output file alongside predictions.")
@click.pass_context
def predict_command(ctx: click.Context, model_path: str, data_path: str, output_path: Optional[str], proba: bool, include_input: bool) -> None:
    """Score new data with a saved model.

    The incoming schema is checked against the training schema first: a missing
    feature column fails immediately rather than silently misaligning columns.

    \b
    Examples:
      dive predict --model ./out/model.pkl --data new_rows.csv
      dive predict --model ./out/model.pkl --data new.csv --proba --output scored.csv
    """
    from dive.commands.predict import run_predict

    run_predict(
        console=_console(ctx),
        model_path=model_path,
        data_path=data_path,
        output_path=output_path or "predictions.csv",
        with_proba=proba,
        include_input=include_input,
    )


# ----------------------------------------------------------------------
@cli.command("validate")
@click.option("--data", "data_path", required=True, type=click.Path(), help="Path to the data file. Any tabular format pandas can read. Quote paths containing spaces.")
@click.option("--target", "target", default=None, help="Column to predict. Without it, only structural checks run.")
@click.option("--test-size", type=float, default=0.2, show_default=True, help="Holdout fraction, used to reproduce the same split train will use.")
@click.option("--random-state", type=int, default=42, show_default=True, help="Seed for the reproducible split.")
@click.option("--time-series", is_flag=True, help="Split chronologically instead of randomly.")
@click.option("--output", "output_path", default=None, type=click.Path(), help="Also write the report as JSON.")
@click.option("--strict", is_flag=True, help="Return exit code 1 on warnings too (for CI gates).")
@click.pass_context
def validate_command(
    ctx: click.Context,
    data_path: str,
    target: Optional[str],
    test_size: float,
    random_state: int,
    time_series: bool,
    output_path: Optional[str],
    strict: bool,
) -> None:
    """Check a dataset for the failures that silently invalidate models.

    No model is trained. The suite covers target health, leakage, duplicate-row
    leakage, train/holdout drift, and missing data, and reports a PASS / WARN /
    FAIL verdict per check.

    \b
    Examples:
      dive validate --data data.csv --target diagnosis
      dive validate --data data.csv --target churned --strict --output checks.json
    """
    from dive.commands.validate import run_validate

    code = run_validate(
        console=_console(ctx),
        data_path=data_path,
        target=target,
        test_size=test_size,
        random_state=random_state,
        time_series=time_series,
        output_path=output_path,
        strict=strict,
    )
    ctx.exit(code)


# ----------------------------------------------------------------------
@cli.command("explain")
@click.option("--model", "model_path", required=True, type=click.Path(), help="Path to a .pkl written by `dive train`.")
@click.option("--output", "output_path", default=None, type=click.Path(), help="Write the explanation as a standalone HTML file.")
@click.option("--open", "open_browser", is_flag=True, help="Open the output in your browser after writing it.")
@click.pass_context
def explain_command(
    ctx: click.Context, model_path: str, output_path: Optional[str], open_browser: bool
) -> None:
    """Explain how the saved model was built, in plain English.

    Prints the pipeline stages and top features to the terminal. With
    ``--output`` it writes a standalone HTML file that also contains the exact
    Python needed to reproduce the model without dive.

    \b
    Examples:
      dive explain --model ./out/model.pkl
      dive explain --model ./out/model.pkl --output explanation.html
    """
    from dive.commands.report import run_explain

    run_explain(
        console=_console(ctx), model_path=model_path, output_path=output_path,
        open_browser=open_browser,
    )


# ----------------------------------------------------------------------
@cli.command("report")
@click.option("--model", "model_path", required=True, type=click.Path(), help="Path to a .pkl written by `dive train`.")
@click.option("--output", "output_path", default="report.html", type=click.Path(), show_default=True, help="Where to write the HTML report.")
@click.option("--open", "open_browser", is_flag=True, help="Open the report in your browser after writing it.")
@click.pass_context
def report_command(
    ctx: click.Context, model_path: str, output_path: str, open_browser: bool
) -> None:
    """Render the full HTML report for a saved model.

    The report is a single self-contained file: leaderboard, diagnostics plots,
    feature importance, data profile, and every crosscheck verdict from the
    training run (read from validation.json beside the model).

    \b
    Examples:
      dive report --model ./out/model.pkl
      dive report --model ./out/model.pkl --output report.html --open
    """
    from dive.commands.report import run_report

    run_report(
        console=_console(ctx), model_path=model_path, output_path=output_path,
        open_browser=open_browser,
    )


# ----------------------------------------------------------------------
@cli.command("docs")
@click.option("--output", "output_dir", default=None, type=click.Path(), help="Directory to write the docs into. Defaults to the package docs/ directory.")
@click.option("--serve", is_flag=True, help="Serve the docs over HTTP instead of just opening a file.")
@click.option("--port", type=int, default=8000, show_default=True, help="Port for --serve.")
@click.option("--no-browser", is_flag=True, help="Do not open a browser automatically.")
@click.pass_context
def docs_command(
    ctx: click.Context,
    output_dir: Optional[str],
    serve: bool,
    port: int,
    no_browser: bool,
) -> None:
    """Open or serve the HTML documentation site.

    \b
    Examples:
      dive docs
      dive docs --serve --port 9000
      dive docs --output ./my-docs --no-browser
    """
    from dive.commands.report import run_docs

    run_docs(
        console=_console(ctx),
        output_dir=output_dir,
        serve=serve,
        port=port,
        no_browser=no_browser,
    )


# ----------------------------------------------------------------------
@cli.command("deps")
@click.pass_context
def deps_command(ctx: click.Context) -> None:
    """Show which optional packages are installed and what each one enables."""
    console = _console(ctx)
    console.rule("Optional dependencies")
    for package in dependency_report():
        mark = (
            console.status_symbol("ok")
            if package.available
            else console.status_symbol("fail")
        )
        version = f"v{package.version}" if package.version else "not installed"
        console.print(f" {mark} {package.name:<20} {version:<16} {package.provides}")
    console.print("")
    summary = missing_summary()
    if summary:
        console.print(summary)
    else:
        console.success("All optional packages are installed.")


# ----------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    """Entry point installed as the ``dive`` console script."""
    args = list(sys.argv[1:] if argv is None else argv)
    show_traceback = "--traceback" in args
    if not show_traceback:
        _quiet_third_party_warnings()

    try:
        # standalone_mode=False makes click *return* the command's exit code
        # (from ctx.exit) instead of raising SystemExit. Propagate it, or every
        # failure would report success to the shell.
        rv = cli.main(args=args, prog_name="dive", standalone_mode=False)
        return rv if isinstance(rv, int) else 0
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.exceptions.Abort:
        get_console().error("Aborted.")
        return 1
    except click.exceptions.ClickException as exc:
        exc.show()
        return exc.exit_code
    except DiveError as exc:
        console = get_console()
        console.print("")
        console.error(str(exc))
        if show_traceback:
            import traceback

            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        get_console().error("Interrupted by user.")
        return 130
    except Exception as exc:  # unexpected - this is a bug, not user error
        console = get_console()
        console.print("")
        console.error(f"Unexpected internal error: {type(exc).__name__}: {exc}")
        if show_traceback:
            import traceback

            traceback.print_exc()
        else:
            console.print(
                "  Re-run with --traceback for the full stack trace, and please "
                "report this as a bug."
            )
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
