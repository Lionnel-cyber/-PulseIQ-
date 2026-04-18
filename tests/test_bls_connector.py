"""Tests for src/connectors/bls_connector.py.

All HTTP calls are intercepted by the ``responses`` library — no live API calls.
``monkeypatch`` injects a dummy BLS_API_KEY for every test that instantiates the connector.
``patch("time.sleep")`` prevents tenacity from actually sleeping during retry tests.
"""

import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests
import responses as responses_lib

from src.connectors.bls_connector import BLSConnector, _derive_geo, _parse_date

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

MOCK_BLS_RESPONSE = {
    "status": "REQUEST_SUCCEEDED",
    "Results": {
        "series": [
            {
                "seriesID": "LNS14000000",
                "data": [
                    {
                        "year": "2024",
                        "period": "M01",
                        "periodName": "January",
                        "value": "3.7",
                    },
                    {
                        "year": "2024",
                        "period": "M02",
                        "periodName": "February",
                        "value": "3.9",
                    },
                ],
            }
        ]
    },
}

# ---------------------------------------------------------------------------
# Unit tests for module-level helpers (no connector, no HTTP)
# ---------------------------------------------------------------------------


def test_parse_date_monthly() -> None:
    """M01 in 2024 → 2024-01-01."""
    assert _parse_date("2024", "M01") == pd.Timestamp("2024-01-01")


def test_parse_date_monthly_december() -> None:
    """M12 in 2023 → 2023-12-01."""
    assert _parse_date("2023", "M12") == pd.Timestamp("2023-12-01")


def test_parse_date_weekly() -> None:
    """W01 in 2024 → Monday of ISO week 1, 2024 (2024-01-01)."""
    expected = pd.Timestamp(date.fromisocalendar(2024, 1, 1))
    assert _parse_date("2024", "W01") == expected


def test_parse_date_quarterly() -> None:
    """Q2 → first day of April."""
    assert _parse_date("2024", "Q02") == pd.Timestamp("2024-04-01")


def test_parse_date_annual() -> None:
    """A01 → January 1."""
    assert _parse_date("2024", "A01") == pd.Timestamp("2024-01-01")


def test_parse_date_invalid_prefix() -> None:
    """Unknown period prefix raises ValueError."""
    with pytest.raises(ValueError, match="Unrecognised BLS period prefix"):
        _parse_date("2024", "Z99")


def test_derive_geo_lasst_california() -> None:
    """LASST series with FIPS 06 → state geo tuple for California."""
    assert _derive_geo("LASST060000000000003") == ("06", "state", "California")


def test_derive_geo_lasst_unknown_fips() -> None:
    """LASST series with unknown FIPS falls back to 'State FIPS {code}'."""
    assert _derive_geo("LASST990000000000003") == ("99", "state", "State FIPS 99")


def test_derive_geo_national_series() -> None:
    """LNS (national) series → national geo tuple."""
    assert _derive_geo("LNS14000000") == ("US", "national", "United States")


def test_derive_geo_unknown_prefix() -> None:
    """Unrecognised prefix → national geo tuple."""
    assert _derive_geo("XYZABC123") == ("US", "national", "United States")


# ---------------------------------------------------------------------------
# Connector tests
# ---------------------------------------------------------------------------


def test_raises_if_api_key_missing() -> None:
    """BLSConnector raises ValueError at construction if BLS_API_KEY is unset."""
    saved = os.environ.pop("BLS_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="BLS_API_KEY"):
            BLSConnector()
    finally:
        if saved is not None:
            os.environ["BLS_API_KEY"] = saved


@responses_lib.activate
def test_successful_fetch_returns_correct_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() returns a DataFrame with documented columns and correct dtypes."""
    monkeypatch.setenv("BLS_API_KEY", "test_key")
    responses_lib.add(responses_lib.POST, BLS_URL, json=MOCK_BLS_RESPONSE, status=200)

    connector = BLSConnector(raw_data_root=tmp_path)
    df = connector.fetch(series_ids=["LNS14000000"])

    assert list(df.columns) == ["date", "series_id", "value", "geo_id", "geo_level", "geo_name"]
    assert len(df) == 2
    assert df["series_id"].iloc[0] == "LNS14000000"
    assert df["geo_id"].iloc[0] == "US"
    assert df["geo_level"].iloc[0] == "national"
    assert df["geo_name"].iloc[0] == "United States"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["value"])
    assert df["value"].iloc[0] == pytest.approx(3.7)


@responses_lib.activate
def test_date_parsing_in_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() correctly parses BLS M01 period into 2024-01-01."""
    monkeypatch.setenv("BLS_API_KEY", "test_key")
    responses_lib.add(responses_lib.POST, BLS_URL, json=MOCK_BLS_RESPONSE, status=200)

    connector = BLSConnector(raw_data_root=tmp_path)
    df = connector.fetch(series_ids=["LNS14000000"])

    assert df["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert df["date"].iloc[1] == pd.Timestamp("2024-02-01")


@responses_lib.activate
def test_weekly_period_parsed_in_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() correctly parses BLS W01 weekly period into ISO week 1 Monday."""
    monkeypatch.setenv("BLS_API_KEY", "test_key")

    weekly_response = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "LNS14000000",
                    "data": [
                        {
                            "year": "2024",
                            "period": "W01",
                            "periodName": "Week 1",
                            "value": "210000",
                        }
                    ],
                }
            ]
        },
    }
    responses_lib.add(responses_lib.POST, BLS_URL, json=weekly_response, status=200)

    connector = BLSConnector(raw_data_root=tmp_path)
    df = connector.fetch(series_ids=["LNS14000000"])

    expected = pd.Timestamp(date.fromisocalendar(2024, 1, 1))
    assert df["date"].iloc[0] == expected


@responses_lib.activate
def test_lasst_series_derives_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() derives the correct state abbreviation for LASST series."""
    monkeypatch.setenv("BLS_API_KEY", "test_key")

    lasst_response = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "LASST060000000000003",
                    "data": [
                        {
                            "year": "2024",
                            "period": "M01",
                            "periodName": "January",
                            "value": "5.1",
                        }
                    ],
                }
            ]
        },
    }
    responses_lib.add(responses_lib.POST, BLS_URL, json=lasst_response, status=200)

    connector = BLSConnector(raw_data_root=tmp_path)
    df = connector.fetch(series_ids=["LASST060000000000003"])

    assert df["geo_id"].iloc[0] == "06"
    assert df["geo_level"].iloc[0] == "state"
    assert df["geo_name"].iloc[0] == "California"


@responses_lib.activate
def test_raises_on_non_success_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() raises ValueError when the BLS API returns REQUEST_FAILED."""
    monkeypatch.setenv("BLS_API_KEY", "test_key")

    failed_response = {
        "status": "REQUEST_FAILED",
        "message": ["Series does not exist"],
        "Results": {"series": []},
    }
    responses_lib.add(responses_lib.POST, BLS_URL, json=failed_response, status=200)

    connector = BLSConnector(raw_data_root=tmp_path)
    with pytest.raises(ValueError, match="non-success status"):
        connector.fetch(series_ids=["INVALID_SERIES"])


@responses_lib.activate
def test_retries_on_429(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """fetch() retries after a 429 and returns data on the third attempt."""
    monkeypatch.setenv("BLS_API_KEY", "test_key")

    responses_lib.add(responses_lib.POST, BLS_URL, status=429)
    responses_lib.add(responses_lib.POST, BLS_URL, status=429)
    responses_lib.add(responses_lib.POST, BLS_URL, json=MOCK_BLS_RESPONSE, status=200)

    with patch("time.sleep"):
        connector = BLSConnector(raw_data_root=tmp_path)
        df = connector.fetch(series_ids=["LNS14000000"])

    assert len(df) == 2


@responses_lib.activate
def test_raises_after_max_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() raises HTTPError after all 3 retry attempts fail."""
    monkeypatch.setenv("BLS_API_KEY", "test_key")

    for _ in range(3):
        responses_lib.add(responses_lib.POST, BLS_URL, status=500)

    with patch("time.sleep"):
        connector = BLSConnector(raw_data_root=tmp_path)
        with pytest.raises(requests.exceptions.HTTPError):
            connector.fetch(series_ids=["LNS14000000"])
