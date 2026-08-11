"""Cross-platform data loading, saving, and input validation.

Every path in this package flows through :func:`resolve_path`, which returns a
``pathlib.Path``. No string concatenation, no ``os.sep``, no shell calls - the
same code runs unchanged on Windows, macOS, and Linux.

Text files are written with ``newline=""``-safe helpers and read with explicit
UTF-8 handling so that CRLF/LF differences never corrupt output.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from dive.exceptions import ConfigError, DataError, ModelError, TargetError

# Extension -> reader family. Keys are lower-case, dot-prefixed.
# Covers every tabular format pandas ships a reader for. Formats whose reader
# needs a third-party engine (pyarrow, openpyxl, tables, pyreadstat, lxml) are
# listed here and produce an install hint at read time rather than an
# "unsupported file type" error, which would be misleading.
_TABULAR_SUFFIXES = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".dat": "csv",
    ".data": "csv",
    ".psv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".parq": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".xlsb": "excel",
    ".ods": "excel",
    ".feather": "feather",
    ".arrow": "feather",
    ".orc": "orc",
    ".h5": "hdf",
    ".hdf": "hdf",
    ".hdf5": "hdf",
    ".dta": "stata",
    ".sav": "spss",
    ".zsav": "spss",
    ".por": "spss",
    ".sas7bdat": "sas",
    ".xpt": "sas",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".pkl": "pickle",
    ".pickle": "pickle",
}

# Compression suffixes that pandas' text readers decompress transparently. A
# doubled extension such as data.csv.gz resolves against the inner suffix.
_COMPRESSION_SUFFIXES = {".gz", ".bz2", ".zip", ".xz", ".zst", ".zstd", ".tar"}

# Reader family -> install hint, used to turn an ImportError from a missing
# engine into a message naming the exact package to install.
_ENGINE_HINTS = {
    "parquet": "pip install pyarrow",
    "feather": "pip install pyarrow",
    "orc": "pip install pyarrow",
    "excel": "pip install openpyxl   (odfpy for .ods, xlrd for legacy .xls, pyxlsb for .xlsb)",
    "hdf": "pip install tables",
    "spss": "pip install pyreadstat",
    "html": "pip install lxml html5lib beautifulsoup4",
    "xml": "pip install lxml",
}

MODEL_SUFFIXES = (".pkl", ".pickle", ".joblib")


def _strip_quotes(raw: str) -> str:
    """Remove one layer of surrounding quotes from a user-supplied path.

    Paths containing spaces are routinely wrapped in quotes, and Windows
    Explorer's "Copy as path" emits a quoted string. When such a value reaches
    us through a config file, a copy-paste, or a shell that does not strip
    quotes itself, the quote characters become part of the filename and the
    open fails with a confusing "not found". Only a matched pair is removed, so
    a filename that legitimately contains a quote is left intact.
    """
    text = raw.strip()
    for quote in ('"', "'"):
        if len(text) >= 2 and text.startswith(quote) and text.endswith(quote):
            return text[1:-1].strip()
    return text


def resolve_path(path: Any, must_exist: bool = False, kind: str = "file") -> Path:
    """Expand and normalise a user-supplied path.

    Handles ``~`` expansion, surrounding quotes, and relative paths identically
    across platforms. When ``must_exist`` is set, raises :class:`DataError`
    naming the resolved absolute path so the user can see exactly where the
    tool looked.
    """
    if path is None:
        raise DataError("No path was provided.", "Expected a file or directory path.")
    try:
        resolved = Path(_strip_quotes(str(path))).expanduser()
    except Exception as exc:
        raise DataError(f"Could not interpret path: {path!r} ({exc})") from exc

    if must_exist and not resolved.exists():
        raise DataError(
            f"{kind.capitalize()} not found: {resolved}",
            f"Looked for an absolute path of: {resolved.resolve() if not resolved.is_absolute() else resolved}",
        )
    return resolved


def ensure_dir(path: Any) -> Path:
    """Create a directory (and parents) if needed and return it."""
    directory = resolve_path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataError(
            f"Could not create output directory: {directory}",
            f"The operating system reported: {exc}",
        ) from exc
    if not directory.is_dir():
        raise DataError(f"Output path exists but is not a directory: {directory}")
    return directory


def _payload_suffix(path: Path) -> str:
    """Return the format-bearing suffix, looking past a compression suffix.

    ``data.csv`` -> ``.csv`` and ``data.csv.gz`` -> ``.csv``.
    """
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        return ""
    if suffixes[-1] in _COMPRESSION_SUFFIXES and len(suffixes) >= 2:
        return suffixes[-2]
    return suffixes[-1]


def _detect_format(path: Path) -> str:
    """Map a filename to a reader family, seeing through compression suffixes."""
    # data.csv.gz / data.json.zip: the compression is transparent to pandas, so
    # the format is decided by the suffix underneath it.
    payload = _payload_suffix(path)
    if payload in _TABULAR_SUFFIXES:
        return _TABULAR_SUFFIXES[payload]

    # A file with no extension at all is almost always delimited text - UCI-style
    # dataset dumps and shell redirections both produce them - and the CSV
    # reader sniffs the delimiter, so this is a safe default. A file with an
    # extension we do not know stays an error: guessing there would turn a typo
    # into a confusing parser failure.
    if payload == "":
        return "csv"

    suffix = path.suffix.lower()
    supported = ", ".join(sorted(_TABULAR_SUFFIXES))
    raise DataError(
        f"Unsupported data file type: '{suffix or path.name}'",
        f"Supported extensions are: {supported}\n"
        "Text formats may additionally be compressed (.gz, .bz2, .zip, .xz, .zst).",
    )


def _read_by_format(fmt: str, resolved: Path, max_rows: Optional[int]) -> pd.DataFrame:
    """Dispatch to the pandas reader for one format family."""
    if fmt == "csv":
        # Extensions that name their own delimiter are read with it explicitly;
        # anything else is sniffed by the python engine, which covers comma,
        # semicolon, and whitespace-separated .dat/.txt files. Sniffing is not
        # safe for a known delimiter: header cells containing spaces can make
        # the sniffer pick the wrong character and silently split columns.
        explicit = {".tsv": "\t", ".psv": "|"}
        stem_suffix = _payload_suffix(resolved)
        separator = explicit.get(stem_suffix)
        return pd.read_csv(
            resolved,
            sep=separator,
            engine="python" if separator is None else "c",
            nrows=max_rows,
            encoding="utf-8",
            encoding_errors="replace",
        )
    if fmt == "parquet":
        return pd.read_parquet(resolved)
    if fmt == "json":
        return pd.read_json(resolved)
    if fmt == "jsonl":
        return pd.read_json(resolved, lines=True)
    if fmt == "excel":
        # sheet_name=0 keeps the return type a DataFrame; without it a
        # multi-sheet workbook would hand back a dict.
        return pd.read_excel(resolved, sheet_name=0, nrows=max_rows)
    if fmt == "feather":
        return pd.read_feather(resolved)
    if fmt == "orc":
        return pd.read_orc(resolved)
    if fmt == "hdf":
        frame = pd.read_hdf(resolved)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        return frame
    if fmt == "stata":
        return pd.read_stata(resolved)
    if fmt == "spss":
        return pd.read_spss(resolved)
    if fmt == "sas":
        return pd.read_sas(resolved)
    if fmt == "html":
        tables = pd.read_html(resolved)
        if not tables:
            raise DataError(
                f"No HTML tables found in {resolved}",
                "The file parsed, but contains no <table> element to read.",
            )
        return tables[0]
    if fmt == "xml":
        return pd.read_xml(resolved)
    if fmt == "pickle":
        obj = pd.read_pickle(resolved)
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        if not isinstance(obj, pd.DataFrame):
            raise DataError(
                f"The pickle at {resolved} does not contain a DataFrame.",
                f"It unpickled to a {type(obj).__name__}. To score rows with a "
                "saved model use --model, not --data.",
            )
        return obj
    raise DataError(f"Unhandled data format: {fmt}")  # pragma: no cover


def load_dataframe(path: Any, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load a tabular file into a DataFrame with specific, actionable errors.

    Parameters
    ----------
    path:
        File to read. Format is inferred from the extension, seeing through a
        trailing compression suffix.
    max_rows:
        Optional cap on rows read, used by fast-path sampling.
    """
    resolved = resolve_path(path, must_exist=True, kind="data file")

    if resolved.is_dir():
        raise DataError(
            f"Expected a data file but found a directory: {resolved}",
            "Point --data at a .csv/.parquet/.json file.",
        )
    try:
        if resolved.stat().st_size == 0:
            raise DataError(
                f"Data file is empty (0 bytes): {resolved}",
                "Check the file was written completely.",
            )
    except OSError as exc:
        raise DataError(f"Could not read file metadata for {resolved}: {exc}") from exc

    fmt = _detect_format(resolved)

    try:
        frame = _read_by_format(fmt, resolved, max_rows)
    except DataError:
        raise
    except ImportError as exc:
        hint = _ENGINE_HINTS.get(fmt)
        raise DataError(
            f"Reading {resolved.suffix} files needs an extra package that is not installed.",
            f"Underlying error: {exc}." + (f" Try: {hint}" if hint else ""),
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataError(
            f"Could not decode {resolved} as text.",
            f"The file may be binary or use a non-UTF-8 encoding ({exc}).",
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DataError(
            f"Data file contains no parseable rows: {resolved}",
            "The file has no header row or no records.",
        ) from exc
    except pd.errors.ParserError as exc:
        raise DataError(
            f"Could not parse {resolved} as a table.",
            f"Parser reported: {exc}. Check for inconsistent column counts or a wrong delimiter.",
        ) from exc
    except Exception as exc:
        raise DataError(f"Failed to read {resolved}: {type(exc).__name__}: {exc}") from exc

    # Readers without an nrows parameter are truncated after the fact.
    if max_rows is not None and fmt not in {"csv", "excel"}:
        frame = frame.head(max_rows)

    if frame.shape[0] == 0:
        raise DataError(
            f"Data file has a header but zero rows: {resolved}",
            "At least 2 rows are needed to fit and evaluate a model.",
        )
    if frame.shape[1] == 0:
        raise DataError(f"Data file has no columns: {resolved}")

    frame.columns = [str(col) for col in frame.columns]
    duplicates = _duplicate_names(frame.columns)
    if duplicates:
        raise DataError(
            f"Data file has duplicate column names: {', '.join(duplicates)}",
            "Rename the duplicates so every column is uniquely addressable.",
        )
    return frame


def _duplicate_names(columns: Iterable[str]) -> List[str]:
    seen, duplicates = set(), []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def _suggest(name: str, candidates: Sequence[str], limit: int = 3) -> List[str]:
    """Return close column-name matches for a typo-friendly error message."""
    import difflib

    close = difflib.get_close_matches(name, list(candidates), n=limit, cutoff=0.6)
    if close:
        return close
    lowered = name.lower()
    return [c for c in candidates if c.lower() == lowered][:limit]


def validate_target(
    frame: pd.DataFrame,
    target: str,
    min_rows: int = 10,
) -> None:
    """Verify a target column can actually be modelled; raise otherwise.

    Catches, with a specific message for each: absent column (plus close-name
    suggestions), all-missing target, a single unique value, and too few usable
    rows after dropping missing targets.
    """
    if target not in frame.columns:
        suggestions = _suggest(target, frame.columns)
        available = list(frame.columns)
        preview = ", ".join(available[:15]) + (" ..." if len(available) > 15 else "")
        hint = f"Available columns ({len(available)}): {preview}"
        if suggestions:
            hint = f"Did you mean: {', '.join(suggestions)}?\n{hint}"
        raise TargetError(f"Target column '{target}' is not in the data file.", hint)

    column = frame[target]
    non_null = column.dropna()

    if len(non_null) == 0:
        raise TargetError(
            f"Target column '{target}' is empty - every value is missing.",
            "A model cannot be trained without labels.",
        )

    missing_pct = 100.0 * (len(column) - len(non_null)) / max(len(column), 1)
    if missing_pct > 50:
        raise TargetError(
            f"Target column '{target}' is {missing_pct:.1f}% missing "
            f"({len(column) - len(non_null)} of {len(column)} rows).",
            "Rows with a missing target are dropped, leaving too little data to train on.",
        )

    unique = non_null.nunique(dropna=True)
    if unique < 2:
        only = non_null.iloc[0] if len(non_null) else "n/a"
        raise TargetError(
            f"Target column '{target}' has only one unique value ({only!r}).",
            "There is nothing to predict. Choose a target column that varies across rows.",
        )

    if len(non_null) < min_rows:
        raise TargetError(
            f"Only {len(non_null)} labelled rows available in '{target}'; "
            f"at least {min_rows} are needed.",
            "Provide more data, or reduce --test-size for a very small dataset.",
        )

    if frame.shape[1] < 2:
        raise DataError(
            "The data file contains only the target column - there are no features.",
            "Add at least one predictor column.",
        )


def save_dataframe(frame: pd.DataFrame, path: Any, index: bool = False) -> Path:
    """Write a DataFrame, choosing the writer from the file extension.

    CSV is written with ``lineterminator="\\n"`` so a file produced on Windows
    is byte-identical to one produced on Linux.
    """
    resolved = resolve_path(path)
    if resolved.parent and not resolved.parent.exists():
        ensure_dir(resolved.parent)

    suffix = resolved.suffix.lower()
    try:
        if suffix in {".csv", ".txt", ".dat", ".data", ""}:
            if suffix == "":
                resolved = resolved.with_suffix(".csv")
            frame.to_csv(resolved, index=index, encoding="utf-8", lineterminator="\n")
        elif suffix == ".tsv":
            frame.to_csv(
                resolved, index=index, sep="\t", encoding="utf-8", lineterminator="\n"
            )
        elif suffix == ".psv":
            frame.to_csv(
                resolved, index=index, sep="|", encoding="utf-8", lineterminator="\n"
            )
        elif suffix in {".parquet", ".pq", ".parq"}:
            frame.to_parquet(resolved, index=index)
        elif suffix == ".json":
            frame.to_json(resolved, orient="records", indent=2)
        elif suffix in {".jsonl", ".ndjson"}:
            frame.to_json(resolved, orient="records", lines=True)
        elif suffix in {".xlsx", ".xls", ".xlsm", ".ods"}:
            frame.to_excel(resolved, index=index)
        elif suffix in {".feather", ".arrow"}:
            frame.reset_index(drop=not index).to_feather(resolved)
        elif suffix == ".orc":
            frame.to_orc(resolved, index=index)
        elif suffix == ".dta":
            frame.to_stata(resolved, write_index=index)
        elif suffix in {".html", ".htm"}:
            frame.to_html(resolved, index=index)
        elif suffix == ".xml":
            frame.to_xml(resolved, index=index)
        elif suffix in {".pkl", ".pickle"}:
            frame.to_pickle(resolved)
        elif suffix in {".h5", ".hdf", ".hdf5"}:
            frame.to_hdf(resolved, key="data", mode="w")
        else:
            raise DataError(
                f"Cannot write output with extension '{suffix}'.",
                "Use .csv, .tsv, .psv, .parquet, .json, .jsonl, .xlsx, .ods, "
                ".feather, .orc, .dta, .html, .xml, .pkl, or .h5.",
            )
    except DataError:
        raise
    except ImportError as exc:
        raise DataError(
            f"Writing {suffix} files needs an extra package that is not installed.",
            f"Underlying error: {exc}",
        ) from exc
    except OSError as exc:
        raise DataError(
            f"Could not write to {resolved}",
            f"The operating system reported: {exc}",
        ) from exc
    return resolved


def write_text(path: Any, content: str) -> Path:
    """Write UTF-8 text with normalised newlines (identical bytes on all OSes)."""
    resolved = resolve_path(path)
    if resolved.parent and not resolved.parent.exists():
        ensure_dir(resolved.parent)
    try:
        with open(resolved, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except OSError as exc:
        raise DataError(
            f"Could not write file: {resolved}",
            f"The operating system reported: {exc}",
        ) from exc
    return resolved


def read_text(path: Any) -> str:
    """Read UTF-8 text, tolerating a BOM written by Windows editors."""
    resolved = resolve_path(path, must_exist=True)
    try:
        with open(resolved, "r", encoding="utf-8-sig") as handle:
            return handle.read()
    except OSError as exc:
        raise DataError(f"Could not read file: {resolved}", str(exc)) from exc


def save_pickle(obj: Any, path: Any) -> Path:
    """Pickle an object to disk with the highest available protocol."""
    resolved = resolve_path(path)
    if resolved.parent and not resolved.parent.exists():
        ensure_dir(resolved.parent)
    try:
        with open(resolved, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        raise ModelError(
            f"Could not write model file: {resolved}",
            f"The operating system reported: {exc}",
        ) from exc
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ModelError(
            f"Model could not be serialised to {resolved}.",
            f"Pickling failed with: {type(exc).__name__}: {exc}",
        ) from exc
    return resolved


def load_pickle(path: Any) -> Any:
    """Unpickle a model artifact, mapping every failure to a clear message."""
    resolved = resolve_path(path, must_exist=True, kind="model file")
    if resolved.is_dir():
        raise ModelError(
            f"Expected a model file but found a directory: {resolved}",
            "Point --model at the .pkl file written by `dive train`.",
        )
    try:
        with open(resolved, "rb") as handle:
            return pickle.load(handle)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ModelError(
            f"'{resolved}' is not a readable model file.",
            f"It may be truncated or not a pickle at all ({type(exc).__name__}).",
        ) from exc
    except ModuleNotFoundError as exc:
        raise ModelError(
            f"Model at '{resolved}' needs a package that is not installed: {exc.name}",
            f"This model was trained with {exc.name} available. Install it "
            f"(pip install {exc.name}) and retry.",
        ) from exc
    except AttributeError as exc:
        raise ModelError(
            f"Model at '{resolved}' was created by an incompatible version of dive.",
            f"Missing attribute: {exc}. Retrain with the current version.",
        ) from exc
    except OSError as exc:
        raise ModelError(f"Could not open model file: {resolved}", str(exc)) from exc


def load_config(path: Any) -> Dict[str, Any]:
    """Load a YAML (or JSON) config file into a flat dict of CLI overrides."""
    resolved = resolve_path(path, must_exist=True, kind="config file")
    text = read_text(resolved)
    suffix = resolved.suffix.lower()

    try:
        if suffix == ".json":
            import json

            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    except Exception as exc:
        raise ConfigError(
            f"Could not parse config file: {resolved}",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a mapping of option names to values: {resolved}",
            f"Found a {type(data).__name__} instead.",
        )
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def unique_path(path: Any) -> Path:
    """Return ``path``, or ``name_1.ext``/``name_2.ext`` if it already exists."""
    resolved = resolve_path(path)
    if not resolved.exists():
        return resolved
    stem, suffix, parent = resolved.stem, resolved.suffix, resolved.parent
    for index in range(1, 1000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    return resolved


def load_json(path: Any) -> Dict[str, Any]:
    """Read JSON from file path."""
    import json
    resolved = resolve_path(path, must_exist=True, kind="JSON file")
    text = read_text(resolved)
    return json.loads(text)


def save_json(path: Any, data: Any, indent: int = 2) -> Path:
    """Write data as formatted JSON to file path."""
    import json
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    text = json.dumps(data, indent=indent)
    write_text(resolved, text)
    return resolved

