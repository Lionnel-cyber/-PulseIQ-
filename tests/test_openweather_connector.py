"""Tests for src/connectors/openweather_connector.py.

All HTTP calls are intercepted by the ``responses`` library — no live API
calls. Tests cover batching, no-alert fallback rows, and schema guarantees.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import responses as responses_lib

from src.connectors.openweather_connector import OpenWeatherConnector

GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/zip"
ONE_CALL_URL = "https://api.openweathermap.org/data/3.0/onecall"


def _geocode_response(zip_code: str, lat: float, lon: float) -> dict[str, object]:
    """Build a geocoding response payload."""
    return {
        "zip": zip_code,
        "name": f"ZIP {zip_code}",
        "lat": lat,
        "lon": lon,
        "country": "US",
    }


def _one_call_response(
    temp: float = 22.5,
    description: str = "light rain",
    alerts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a One Call response payload."""
    payload: dict[str, object] = {
        "current": {
            "dt": 1704067200,
            "temp": temp,
            "weather": [{"description": description}],
        }
    }
    if alerts is not None:
        payload["alerts"] = alerts
    return payload


def test_raises_if_api_key_missing() -> None:
    """OpenWeatherConnector raises ValueError at construction if API key is unset."""
    saved = os.environ.pop("OPENWEATHER_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="OPENWEATHER_API_KEY"):
            OpenWeatherConnector()
    finally:
        if saved is not None:
            os.environ["OPENWEATHER_API_KEY"] = saved


@responses_lib.activate
def test_successful_fetch_returns_correct_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() returns the documented schema and data types."""
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")

    responses_lib.add(
        responses_lib.GET,
        GEOCODING_URL,
        json=_geocode_response("10001", 40.75, -73.99),
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        ONE_CALL_URL,
        json=_one_call_response(
            temp=21.0,
            description="overcast clouds",
            alerts=[{"event": "Flood Warning", "tags": ["Severe"]}],
        ),
        status=200,
    )

    connector = OpenWeatherConnector(raw_data_root=tmp_path)
    df = connector.fetch(["10001"])

    assert list(df.columns) == [
        "date",
        "event_type",
        "severity",
        "temp_celsius",
        "description",
        "geo_id",
        "geo_level",
        "geo_name",
    ]
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["temp_celsius"])
    assert "zip_code" not in df.columns
    assert df.loc[0, "event_type"] == "Flood Warning"
    assert df.loc[0, "severity"] == "Severe"
    assert df.loc[0, "description"] == "overcast clouds"
    assert df.loc[0, "temp_celsius"] == pytest.approx(21.0)
    # geo columns: name from geocode response, state abbr derived from ZIP prefix
    assert df.loc[0, "geo_id"] == "ZIP 10001-NY"
    assert df.loc[0, "geo_level"] == "city"
    assert df.loc[0, "geo_name"] == "ZIP 10001"


@responses_lib.activate
def test_no_alerts_still_returns_a_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ZIPs with no active alerts still produce a fallback output row."""
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")

    responses_lib.add(
        responses_lib.GET,
        GEOCODING_URL,
        json=_geocode_response("94105", 37.79, -122.39),
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        ONE_CALL_URL,
        json=_one_call_response(temp=18.0, description="clear sky"),
        status=200,
    )

    connector = OpenWeatherConnector(raw_data_root=tmp_path)
    df = connector.fetch(["94105"])

    assert len(df) == 1
    assert df.loc[0, "event_type"] == "none"
    assert df.loc[0, "severity"] == "none"
    assert df.loc[0, "description"] == "clear sky"
    assert df.loc[0, "geo_id"] == "ZIP 94105-CA"
    assert df.loc[0, "geo_level"] == "city"
    assert df.loc[0, "geo_name"] == "ZIP 94105"


@responses_lib.activate
def test_multiple_alerts_return_multiple_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each active alert expands to its own output row for the ZIP."""
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")

    responses_lib.add(
        responses_lib.GET,
        GEOCODING_URL,
        json=_geocode_response("30301", 33.75, -84.39),
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        ONE_CALL_URL,
        json=_one_call_response(
            alerts=[
                {"event": "Flood Warning", "tags": ["Severe"]},
                {"event": "Heat Advisory", "tags": ["Moderate"]},
            ]
        ),
        status=200,
    )

    connector = OpenWeatherConnector(raw_data_root=tmp_path)
    df = connector.fetch(["30301"])

    assert len(df) == 2
    assert set(df["event_type"]) == {"Flood Warning", "Heat Advisory"}
    assert set(df["severity"]) == {"Severe", "Moderate"}


@responses_lib.activate
def test_batching_processes_all_zip_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ZIPs are processed in batches while preserving one geocode and one weather call per ZIP."""
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")

    zip_codes = [f"{10000 + index}" for index in range(12)]
    for index, zip_code in enumerate(zip_codes):
        responses_lib.add(
            responses_lib.GET,
            GEOCODING_URL,
            json=_geocode_response(zip_code, 40.0 + index, -73.0 - index),
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            ONE_CALL_URL,
            json=_one_call_response(
                temp=15.0 + index,
                description=f"description-{index}",
            ),
            status=200,
        )

    connector = OpenWeatherConnector(raw_data_root=tmp_path)
    df = connector.fetch(zip_codes)

    assert len(df) == 12
    assert "zip_code" not in df.columns
    assert (df["geo_level"] == "city").all()
    assert len(responses_lib.calls) == 24


@responses_lib.activate
def test_empty_input_returns_empty_dataframe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty ZIP list returns an empty DataFrame without any HTTP calls."""
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")

    connector = OpenWeatherConnector(raw_data_root=tmp_path)
    df = connector.fetch([])

    assert list(df.columns) == [
        "date",
        "event_type",
        "severity",
        "temp_celsius",
        "description",
        "geo_id",
        "geo_level",
        "geo_name",
    ]
    assert df.empty
    assert len(responses_lib.calls) == 0
