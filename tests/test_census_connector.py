"""Tests for src/connectors/census_connector.py.

All HTTP calls are intercepted by the ``responses`` library — no live API
calls. Cached payload tests write directly into a ``tmp_path`` raw-data tree.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import responses as responses_lib

from src.connectors.census_connector import CensusConnector

CENSUS_URL = "https://api.census.gov/data/2023/acs/acs5"

HEADER = [
    "NAME",
    "B17001_002E",
    "B17001_001E",
    "B19013_001E",
    "B25070_010E",
    "B25070_001E",
    "zip code tabulation area",
]


def _raw_payload(*rows: list[str]) -> dict[str, object]:
    """Build the cached/raw payload wrapper used by the connector."""
    return {
        "dataset_year": 2023,
        "rows": [HEADER, *rows],
    }


def _write_cached_payload(
    raw_root: Path,
    cache_date: date,
    payload: dict[str, object],
) -> Path:
    """Write a cached Census payload into the expected date-partitioned path."""
    output_dir = (
        raw_root
        / str(cache_date.year)
        / f"{cache_date.month:02d}"
        / f"{cache_date.day:02d}"
        / "census"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return output_path


def test_raises_if_api_key_missing() -> None:
    """CensusConnector raises ValueError at construction if CENSUS_API_KEY is unset."""
    saved = os.environ.pop("CENSUS_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="CENSUS_API_KEY"):
            CensusConnector()
    finally:
        if saved is not None:
            os.environ["CENSUS_API_KEY"] = saved


@responses_lib.activate
def test_successful_fetch_returns_correct_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() returns the documented schema and numeric columns."""
    monkeypatch.setenv("CENSUS_API_KEY", "test_key")

    responses_lib.add(
        responses_lib.GET,
        CENSUS_URL,
        json=_raw_payload(
            ["ZCTA5 10001", "100", "1000", "75000", "25", "500", "10001"],
            ["ZCTA5 94105", "50", "200", "180000", "10", "100", "94105"],
        )["rows"],
        status=200,
    )

    connector = CensusConnector(raw_data_root=tmp_path)
    df = connector.fetch()

    assert list(df.columns) == [
        "zip_code",
        "poverty_rate",
        "median_income",
        "housing_cost_burden",
        "vintage_year",
        "geo_id",
        "geo_level",
        "geo_name",
        "data_note",
    ]
    assert len(df) == 2
    assert df["zip_code"].tolist() == ["10001", "94105"]
    assert pd.api.types.is_float_dtype(df["poverty_rate"])
    assert pd.api.types.is_float_dtype(df["median_income"])
    assert pd.api.types.is_float_dtype(df["housing_cost_burden"])
    assert pd.api.types.is_integer_dtype(df["vintage_year"])
    assert df.loc[0, "poverty_rate"] == pytest.approx(0.1)
    assert df.loc[0, "housing_cost_burden"] == pytest.approx(0.05)
    # geo columns: geo_id mirrors zip_code; level and name are constant formats
    assert (df["geo_id"] == df["zip_code"]).all()
    assert (df["geo_level"] == "zip").all()
    assert df.loc[0, "geo_name"] == "ZIP 10001"
    assert df.loc[1, "geo_name"] == "ZIP 94105"
    assert (df["data_note"] == "annual baseline only").all()


@responses_lib.activate
def test_uses_cache_within_30_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() uses a recent cached payload and does not make an HTTP request."""
    monkeypatch.setenv("CENSUS_API_KEY", "test_key")
    payload = _raw_payload(
        ["ZCTA5 10001", "100", "1000", "75000", "25", "500", "10001"]
    )
    _write_cached_payload(tmp_path, date.today() - timedelta(days=10), payload)

    connector = CensusConnector(raw_data_root=tmp_path)
    df = connector.fetch()

    assert len(responses_lib.calls) == 0
    assert len(df) == 1
    assert df.loc[0, "zip_code"] == "10001"


@responses_lib.activate
def test_stale_cache_refetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() refreshes the dataset when the newest cache is older than 30 days."""
    monkeypatch.setenv("CENSUS_API_KEY", "test_key")

    stale_payload = _raw_payload(
        ["ZCTA5 10001", "90", "900", "70000", "20", "400", "10001"]
    )
    _write_cached_payload(tmp_path, date.today() - timedelta(days=31), stale_payload)

    responses_lib.add(
        responses_lib.GET,
        CENSUS_URL,
        json=_raw_payload(
            ["ZCTA5 94105", "50", "200", "180000", "10", "100", "94105"]
        )["rows"],
        status=200,
    )

    connector = CensusConnector(raw_data_root=tmp_path)
    df = connector.fetch()

    assert len(responses_lib.calls) == 1
    assert df["zip_code"].tolist() == ["94105"]


@responses_lib.activate
def test_filters_by_state_fips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch(state_fips=...) filters ZIP results locally after loading data."""
    monkeypatch.setenv("CENSUS_API_KEY", "test_key")

    responses_lib.add(
        responses_lib.GET,
        CENSUS_URL,
        json=_raw_payload(
            ["ZCTA5 10001", "100", "1000", "75000", "25", "500", "10001"],
            ["ZCTA5 94105", "50", "200", "180000", "10", "100", "94105"],
            ["ZCTA5 30301", "80", "800", "65000", "40", "400", "30301"],
        )["rows"],
        status=200,
    )

    connector = CensusConnector(raw_data_root=tmp_path)
    df = connector.fetch(state_fips=["06"])

    assert df["zip_code"].tolist() == ["94105"]


@responses_lib.activate
def test_missing_sentinels_become_nan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Census missing sentinel values are converted to NaN without crashing."""
    monkeypatch.setenv("CENSUS_API_KEY", "test_key")

    responses_lib.add(
        responses_lib.GET,
        CENSUS_URL,
        json=_raw_payload(
            [
                "ZCTA5 94105",
                "-666666666",
                "1000",
                "-999999999",
                "10",
                "-888888888",
                "94105",
            ]
        )["rows"],
        status=200,
    )

    connector = CensusConnector(raw_data_root=tmp_path)
    df = connector.fetch()

    assert pd.isna(df.loc[0, "poverty_rate"])
    assert pd.isna(df.loc[0, "median_income"])
    assert pd.isna(df.loc[0, "housing_cost_burden"])
