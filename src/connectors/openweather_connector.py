"""OpenWeather connector for PulseIQ.

Fetches current weather conditions and active national alerts for ZIP codes
using OpenWeather's Geocoding API and One Call 3.0 API. Results are returned
as a tidy DataFrame ready for the dbt staging layer.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel

from src.connectors.base_connector import BaseConnector, http_retry

logger = logging.getLogger(__name__)

_GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/zip"
_ONE_CALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
_CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
_BATCH_SIZE = 10
_OUTPUT_COLUMNS: list[str] = [
    "date",
    "event_type",
    "severity",
    "temp_celsius",
    "description",
    "geo_id",
    "geo_level",
    "geo_name",
]

_ZIP3_TO_STATE_ABBR: list[tuple[int, int, str]] = [
    (6, 9, "PR"), (10, 27, "MA"), (28, 29, "RI"), (30, 38, "NH"),
    (39, 49, "ME"), (50, 59, "VT"), (60, 69, "CT"), (70, 89, "NJ"),
    (100, 149, "NY"), (150, 196, "PA"), (197, 199, "DE"), (200, 205, "DC"),
    (206, 219, "MD"), (220, 246, "VA"), (247, 268, "WV"), (270, 289, "NC"),
    (290, 299, "SC"), (300, 319, "GA"), (320, 349, "FL"), (350, 369, "AL"),
    (370, 385, "TN"), (386, 397, "MS"), (398, 399, "GA"), (400, 427, "KY"),
    (430, 459, "OH"), (460, 479, "IN"), (480, 499, "MI"), (500, 528, "IA"),
    (530, 549, "WI"), (550, 567, "MN"), (570, 577, "SD"), (580, 588, "ND"),
    (590, 599, "MT"), (600, 629, "IL"), (630, 658, "MO"), (660, 679, "KS"),
    (680, 693, "NE"), (700, 715, "LA"), (716, 729, "AR"), (730, 749, "OK"),
    (750, 799, "TX"), (800, 816, "CO"), (820, 831, "WY"), (832, 838, "ID"),
    (840, 847, "UT"), (850, 865, "AZ"), (870, 884, "NM"), (885, 885, "TX"),
    (889, 898, "NV"), (900, 961, "CA"), (967, 968, "HI"), (970, 979, "OR"),
    (980, 994, "WA"), (995, 999, "AK"),
]


def _zip_to_state_abbr(zip_code: str) -> str:
    """Return the two-letter state abbreviation for a ZIP code, or 'US' if unknown.

    Args:
        zip_code: Five-digit ZIP code string.

    Returns:
        Two-letter state abbreviation (e.g. ``"MI"``) or ``"US"`` if the ZIP
        prefix is not found in the bundled lookup table.
    """
    if len(zip_code) < 3 or not zip_code[:3].isdigit():
        return "US"
    zip3 = int(zip_code[:3])
    for start, end, abbr in _ZIP3_TO_STATE_ABBR:
        if start <= zip3 <= end:
            return abbr
    return "US"


def _chunk_zip_codes(zip_codes: list[str], size: int = _BATCH_SIZE) -> list[list[str]]:
    """Split ZIP codes into contiguous chunks.

    Args:
        zip_codes: ZIP codes to chunk.
        size: Maximum chunk size.

    Returns:
        List of ZIP-code batches preserving input order.
    """
    return [zip_codes[index : index + size] for index in range(0, len(zip_codes), size)]


class OpenWeatherGeoResponse(BaseModel):
    """Validated ZIP-to-coordinate response from OpenWeather geocoding."""

    zip: str
    lat: float
    lon: float
    name: str
    country: str


class OpenWeatherCondition(BaseModel):
    """A single weather condition description."""

    description: str


class OpenWeatherCurrent(BaseModel):
    """Current weather snapshot from OpenWeather One Call."""

    dt: int
    temp: float
    weather: list[OpenWeatherCondition]


class OpenWeatherAlert(BaseModel):
    """Validated active weather alert from OpenWeather One Call."""

    event: str
    tags: list[str] | None = None


class OpenWeatherOneCallResponse(BaseModel):
    """Validated One Call 3.0 payload with current weather and alerts."""

    current: OpenWeatherCurrent
    alerts: list[OpenWeatherAlert] | None = None


class OpenWeatherConnector(BaseConnector):
    """Fetch current weather and alerts from OpenWeather for ZIP codes.

    Args:
        raw_data_root: Root directory for raw data storage.
            Defaults to ``"data/raw"``. Pass ``tmp_path`` in tests.

    Raises:
        ValueError: If ``OPENWEATHER_API_KEY`` is not set.
    """

    def __init__(self, raw_data_root: str | Path = "data/raw") -> None:
        super().__init__(raw_data_root)
        self._api_key: str = os.getenv("OPENWEATHER_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "OPENWEATHER_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )

    @http_retry
    def _fetch_zip_geocode(self, zip_code: str) -> dict[str, Any]:
        """Fetch latitude/longitude for a ZIP code.

        Args:
            zip_code: ZIP code to resolve.

        Returns:
            Raw geocoding JSON response.

        Raises:
            requests.exceptions.HTTPError: On 4xx/5xx responses.
        """
        params: dict[str, str] = {
            "zip": f"{zip_code},US",
            "appid": self._api_key,
        }
        self._logger.debug("Resolving ZIP code %s to coordinates", zip_code)
        response = requests.get(_GEOCODING_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    @http_retry
    def _fetch_one_call(self, lat: float, lon: float) -> dict[str, Any]:
        """Fetch current weather and alerts for coordinates.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Raw One Call 3.0 JSON response.

        Raises:
            requests.exceptions.HTTPError: On 4xx/5xx responses.
        """
        params: dict[str, str | float] = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric",
        }
        self._logger.debug("Fetching OpenWeather One Call for lat=%s lon=%s", lat, lon)
        response = requests.get(_ONE_CALL_URL, params=params, timeout=30)
        if response.status_code == 401:
            self._logger.warning(
                "OpenWeather One Call 3.0 unauthorized for this API key; "
                "falling back to the free current weather endpoint."
            )
            fallback = requests.get(_CURRENT_WEATHER_URL, params=params, timeout=30)
            fallback.raise_for_status()
            payload = fallback.json()
            return {
                "current": {
                    "dt": payload.get("dt"),
                    "temp": payload.get("main", {}).get("temp"),
                    "weather": payload.get("weather", []),
                },
                "alerts": None,
            }
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def fetch(self, zip_codes: list[str]) -> pd.DataFrame:
        """Fetch current weather and active alerts for ZIP codes.

        ZIP codes are processed in sequential batches of 10 to reduce the risk
        of rate limiting. Each ZIP always yields at least one output row, even
        when there are no active alerts.

        Args:
            zip_codes: ZIP codes to fetch.

        Returns:
            DataFrame with columns:

            - ``date``         — ``datetime64[ns]``
            - ``event_type``   — ``str``
            - ``severity``     — ``str``
            - ``temp_celsius`` — ``float64``
            - ``description``  — ``str``
            - ``geo_id``       — ``str`` (e.g. ``"Detroit-MI"``)
            - ``geo_level``    — ``str`` (always ``"city"``)
            - ``geo_name``     — ``str`` (city name from geocoding response)

        Raises:
            requests.exceptions.HTTPError: After 3 failed HTTP attempts.
            pydantic.ValidationError: If the API responses fail schema validation.
            ValueError: If ``OPENWEATHER_API_KEY`` is not set (raised at init).
        """
        if not zip_codes:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

        all_raw: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []

        for batch in _chunk_zip_codes(zip_codes):
            self._logger.info("Processing OpenWeather batch of %d ZIP codes", len(batch))
            for zip_code in batch:
                geocode_raw = self._fetch_zip_geocode(zip_code)
                geocode = OpenWeatherGeoResponse.model_validate(geocode_raw)

                weather_raw = self._fetch_one_call(geocode.lat, geocode.lon)
                weather = OpenWeatherOneCallResponse.model_validate(weather_raw)

                all_raw[zip_code] = {
                    "geocode": geocode_raw,
                    "one_call": weather_raw,
                }

                description = (
                    weather.current.weather[0].description
                    if weather.current.weather
                    else ""
                )
                city_name = geocode.name
                state_abbr = _zip_to_state_abbr(zip_code)
                base_row = {
                    "date": pd.to_datetime(weather.current.dt, unit="s"),
                    "temp_celsius": float(weather.current.temp),
                    "description": description,
                    "geo_id": f"{city_name}-{state_abbr}",
                    "geo_level": "city",
                    "geo_name": city_name,
                }

                if weather.alerts:
                    for alert in weather.alerts:
                        rows.append(
                            {
                                **base_row,
                                "event_type": alert.event,
                                "severity": alert.tags[0] if alert.tags else "unknown",
                            }
                        )
                else:
                    rows.append(
                        {
                            **base_row,
                            "event_type": "none",
                            "severity": "none",
                        }
                    )

        self.save_raw(all_raw, source_name="openweather")
        return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
