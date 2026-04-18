"""Tests for src/connectors/base_connector.py.

Uses tmp_path fixture throughout — no writes to the real data/raw directory.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.connectors.base_connector import BaseConnector


# ---------------------------------------------------------------------------
# Minimal concrete subclass — satisfies the abstract interface for testing
# ---------------------------------------------------------------------------


class _ConcreteConnector(BaseConnector):
    """Stub connector used only in tests."""

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today_path(root: Path, source: str) -> Path:
    """Return the expected output directory for today's date."""
    today = date.today()
    return (
        root
        / str(today.year)
        / f"{today.month:02d}"
        / f"{today.day:02d}"
        / source
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cannot_instantiate_abstract_class() -> None:
    """BaseConnector must not be directly instantiable."""
    with pytest.raises(TypeError):
        BaseConnector()  # type: ignore[abstract]


def test_creates_partitioned_directories(tmp_path: Path) -> None:
    """save_raw creates YYYY/MM/DD/{source_name}/ directories."""
    connector = _ConcreteConnector(raw_data_root=tmp_path)
    connector.save_raw({"key": "value"}, source_name="test_source")

    expected_dir = _today_path(tmp_path, "test_source")
    assert expected_dir.is_dir()


def test_writes_correct_json(tmp_path: Path) -> None:
    """save_raw serialises the payload and the file round-trips correctly."""
    connector = _ConcreteConnector(raw_data_root=tmp_path)
    payload = {"series_id": "LNS14000000", "value": 3.7, "state": "CA"}

    path = connector.save_raw(payload, source_name="bls")

    with path.open(encoding="utf-8") as fh:
        loaded = json.load(fh)

    assert loaded == payload


def test_returns_correct_path(tmp_path: Path) -> None:
    """save_raw returns the exact path to the written data.json file."""
    connector = _ConcreteConnector(raw_data_root=tmp_path)
    returned_path = connector.save_raw({}, source_name="fred")

    expected_path = _today_path(tmp_path, "fred") / "data.json"
    assert returned_path == expected_path


def test_save_raw_idempotent(tmp_path: Path) -> None:
    """Calling save_raw twice for the same source on the same day does not raise."""
    connector = _ConcreteConnector(raw_data_root=tmp_path)
    connector.save_raw({"attempt": 1}, source_name="census")
    # Second call overwrites — must not raise FileExistsError
    connector.save_raw({"attempt": 2}, source_name="census")

    path = _today_path(tmp_path, "census") / "data.json"
    with path.open(encoding="utf-8") as fh:
        loaded = json.load(fh)

    assert loaded == {"attempt": 2}


def test_logger_named_after_concrete_class(tmp_path: Path) -> None:
    """Each connector instance logs under its own class name."""
    connector = _ConcreteConnector(raw_data_root=tmp_path)
    assert connector._logger.name == "_ConcreteConnector"
