"""Tests for src/connectors/fred_connector.py.

All HTTP calls are intercepted by the ``responses`` library — no live API calls.
``monkeypatch`` injects a dummy FRED_API_KEY for every test that needs one.
``patch("time.sleep")`` prevents tenacity from actually sleeping during retry tests.
"""

from unittest.mock import patch

import pandas as pd
import pytest
import requests
import responses as responses_lib

from src.connectors.fred_connector import FREDConnector

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"

MOCK_OBS_RESPONSE = {
    "observations": [
        {
            "date": "2024-01-01",
            "value": "314.5",
            "realtime_start": "2024-01-01",
            "realtime_end": "9999-12-31",
        },
        {
            "date": "2024-02-01",
            "value": "315.2",
            "realtime_start": "2024-01-01",
            "realtime_end": "9999-12-31",
        },
    ]
}

MOCK_SERIES_INFO_RESPONSE = {
    "seriess": [
        {
            "id": "CPIAUCSL",
            "units": "Index 1982-1984=100",
        }
    ]
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_raises_if_api_key_missing(tmp_path: object) -> None:
    """FREDConnector raises ValueError at construction if FRED_API_KEY is unset."""
    # Deliberately do NOT set FRED_API_KEY via monkeypatch
    import os
    saved = os.environ.pop("FRED_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="FRED_API_KEY"):
            FREDConnector()
    finally:
        if saved is not None:
            os.environ["FRED_API_KEY"] = saved


@responses_lib.activate
def test_successful_fetch_returns_correct_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """fetch() returns a DataFrame with the documented column schema and dtypes."""
    monkeypatch.setenv("FRED_API_KEY", "test_key")

    responses_lib.add(responses_lib.GET, FRED_SERIES_URL, json=MOCK_SERIES_INFO_RESPONSE, status=200)
    responses_lib.add(responses_lib.GET, FRED_OBS_URL, json=MOCK_OBS_RESPONSE, status=200)

    from pathlib import Path
    connector = FREDConnector(raw_data_root=Path(str(tmp_path)))
    df = connector.fetch(series_ids=["CPIAUCSL"])

    assert list(df.columns) == ["date", "series_id", "value", "unit"]
    assert len(df) == 2
    assert df["series_id"].iloc[0] == "CPIAUCSL"
    assert df["unit"].iloc[0] == "Index 1982-1984=100"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["value"])
    assert df["value"].iloc[0] == pytest.approx(314.5)


@responses_lib.activate
def test_drops_rows_where_value_is_dot(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Observations with value == '.' are excluded from the returned DataFrame."""
    monkeypatch.setenv("FRED_API_KEY", "test_key")

    obs_with_null = {
        "observations": [
            {"date": "2024-01-01", "value": "314.5", "realtime_start": "2024-01-01", "realtime_end": "9999-12-31"},
            {"date": "2024-02-01", "value": ".", "realtime_start": "2024-01-01", "realtime_end": "9999-12-31"},
            {"date": "2024-03-01", "value": "316.0", "realtime_start": "2024-01-01", "realtime_end": "9999-12-31"},
        ]
    }

    responses_lib.add(responses_lib.GET, FRED_SERIES_URL, json=MOCK_SERIES_INFO_RESPONSE, status=200)
    responses_lib.add(responses_lib.GET, FRED_OBS_URL, json=obs_with_null, status=200)

    from pathlib import Path
    connector = FREDConnector(raw_data_root=Path(str(tmp_path)))
    df = connector.fetch(series_ids=["CPIAUCSL"])

    assert len(df) == 2
    assert "." not in df["value"].astype(str).values


@responses_lib.activate
def test_retries_on_429(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """fetch() retries the observations call after a 429 and returns data on success."""
    monkeypatch.setenv("FRED_API_KEY", "test_key")

    responses_lib.add(responses_lib.GET, FRED_SERIES_URL, json=MOCK_SERIES_INFO_RESPONSE, status=200)
    # First two observations calls fail with 429, third succeeds
    responses_lib.add(responses_lib.GET, FRED_OBS_URL, status=429)
    responses_lib.add(responses_lib.GET, FRED_OBS_URL, status=429)
    responses_lib.add(responses_lib.GET, FRED_OBS_URL, json=MOCK_OBS_RESPONSE, status=200)

    from pathlib import Path
    with patch("time.sleep"):
        connector = FREDConnector(raw_data_root=Path(str(tmp_path)))
        df = connector.fetch(series_ids=["CPIAUCSL"])

    assert len(df) == 2


@responses_lib.activate
def test_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """fetch() raises HTTPError after all 3 retry attempts fail."""
    monkeypatch.setenv("FRED_API_KEY", "test_key")

    responses_lib.add(responses_lib.GET, FRED_SERIES_URL, json=MOCK_SERIES_INFO_RESPONSE, status=200)
    # All three attempts fail
    for _ in range(3):
        responses_lib.add(responses_lib.GET, FRED_OBS_URL, status=500)

    from pathlib import Path
    with patch("time.sleep"):
        connector = FREDConnector(raw_data_root=Path(str(tmp_path)))
        with pytest.raises(requests.exceptions.HTTPError):
            connector.fetch(series_ids=["CPIAUCSL"])
