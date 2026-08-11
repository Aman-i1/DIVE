"""Tests for tabular file-format coverage and path handling.

``load_dataframe`` dispatches on the file extension, and CLI paths arrive as
raw strings that may carry the quotes a user copied from their file browser.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dive.exceptions import DataError
from dive.utils.io import load_dataframe, resolve_path, save_dataframe

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
            "score": [0.5, 1.5, 2.5, 3.5],
        }
    )


@pytest.mark.parametrize(
    "suffix, writer",
    [
        (".csv", lambda f, p: f.to_csv(p, index=False)),
        (".tsv", lambda f, p: f.to_csv(p, sep="\t", index=False)),
        (".txt", lambda f, p: f.to_csv(p, index=False)),
        (".json", lambda f, p: f.to_json(p, orient="records")),
        (".xlsx", lambda f, p: f.to_excel(p, index=False)),
    ],
)
def test_round_trips_common_formats(frame, tmp_path, suffix, writer):
    path = tmp_path / f"data{suffix}"
    try:
        writer(frame, path)
    except ImportError as exc:  # optional engine (openpyxl etc.) not installed
        pytest.skip(str(exc))

    loaded = load_dataframe(path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)


@pytest.mark.parametrize("suffix", [".parquet", ".feather"])
def test_round_trips_binary_columnar_formats(frame, tmp_path, suffix):
    pytest.importorskip("pyarrow")
    path = tmp_path / f"data{suffix}"
    save_dataframe(frame, path)
    loaded = load_dataframe(path)
    assert loaded.equals(frame)


def test_semicolon_and_pipe_delimiters(frame, tmp_path):
    """European CSVs are sniffed; .psv declares its delimiter by extension."""
    scsv = tmp_path / "data.csv"
    frame.to_csv(scsv, sep=";", index=False)
    assert list(load_dataframe(scsv).columns) == list(frame.columns)

    psv = tmp_path / "data.psv"
    frame.to_csv(psv, sep="|", index=False)
    assert list(load_dataframe(psv).columns) == list(frame.columns)


def test_compressed_csv(frame, tmp_path):
    path = tmp_path / "data.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    assert len(load_dataframe(path)) == len(frame)


def test_unsupported_extension_names_the_supported_ones(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(DataError) as excinfo:
        load_dataframe(path)
    message = str(excinfo.value)
    assert ".bin" in message
    assert ".csv" in message and ".parquet" in message


def test_extensionless_file_is_read_as_delimited_text(frame, tmp_path):
    path = tmp_path / "export"
    frame.to_csv(path, index=False)
    assert list(load_dataframe(path).columns) == list(frame.columns)


@pytest.mark.parametrize("quote", ['"', "'"])
def test_resolve_path_strips_surrounding_quotes(frame, tmp_path, quote):
    """Users paste quoted paths; the quotes are not part of the filename."""
    path = tmp_path / "my data.csv"
    frame.to_csv(path, index=False)
    resolved = resolve_path(f"{quote}{path}{quote}", must_exist=True)
    assert resolved == path.resolve()


def test_loads_path_with_spaces_and_quotes(frame, tmp_path):
    path = tmp_path / "sales report Q1 2024.csv"
    frame.to_csv(path, index=False)
    assert len(load_dataframe(f'"{path}"')) == len(frame)


def test_strips_trailing_whitespace_and_stray_quote(frame, tmp_path):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    assert len(load_dataframe(f'  "{path}"  \n')) == len(frame)


def test_missing_file_still_reports_not_found(tmp_path):
    """Quote-stripping must not mask a genuinely absent file."""
    with pytest.raises(DataError) as excinfo:
        resolve_path(f'"{tmp_path / "absent.csv"}"', must_exist=True)
    assert "not found" in str(excinfo.value).lower()
    assert '"' not in str(excinfo.value).split("not found")[0][-3:]


def test_directory_rejected_with_clear_message(tmp_path):
    with pytest.raises(DataError):
        load_dataframe(tmp_path)
