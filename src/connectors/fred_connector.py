"""FRED API connector for PulseIQ.

Fetches economic time-series data from the Federal Reserve Economic Data (FRED)
API and returns a tidy DataFrame ready for the dbt staging layer.

Default series:
    CPIAUCSL   — Consumer Price Index for All Urban Consumers
    DRCCLACBS  — Delinquency Rate on Credit Card Loans, All Commercial Banks
"""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel

from src.connectors.base_connector import BaseConnector, http_retry

logger = logging.getLogger(__name__)

_FRED_BASE_URL = "https://api.stlouisfed.org/fred"


# ---------------------------------------------------------------------------
# Pydantic validation models
# ---------------------------------------------------------------------------


class FREDObservation(BaseModel):
    """A single FRED observation data point.

    All fields are kept as ``str`` exactly as returned by the API.
    Type conversion happens during DataFrame construction, not here.

    Attributes:
        date: Observation date in ``YYYY-MM-DD`` format.
        value: Observed value as a string. FRED uses ``"."`` to represent
            missing values.
        realtime_start: Start of the realtime period for this observation.
        realtime_end: End of the realtime period for this observation.
    """

    date: str
    value: str
    realtime_start: str
    realtime_end: str


class FREDResponse(BaseModel):
    """Validated response from the FRED series/observations endpoint.

    Attributes:
        observations: List of individual observation data points.
    """

    observations: list[FREDObservation]


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class FREDConnector(BaseConnector):
    """Fetches economic series data from the FRED API.

    Inherits retry logic and raw data persistence from ``BaseConnector``.
    Each call to ``fetch()`` retrieves observations for one or more series,
    validates them with Pydantic, persists the raw JSON, and returns a
    tidy DataFrame.

    Args:
        raw_data_root: Root directory for raw data storage.
            Defaults to ``"data/raw"``. Pass ``tmp_path`` in tests.

    Raises:
        ValueError: At construction time if ``FRED_API_KEY`` is not set.

    Example::

        connector = FREDConnector()
        df = connector.fetch()                          # uses DEFAULT_SERIES
        df = connector.fetch(["CPIAUCSL", "UNRATE"])   # custom series
    """

    DEFAULT_SERIES: list[str] = ["CPIAUCSL", "DRCCLACBS"]

    def __init__(self, raw_data_root: str | Path = "data/raw") -> None:
        super().__init__(raw_data_root)
        self._api_key: str = os.getenv("FRED_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "FRED_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )

    # ------------------------------------------------------------------
    # Private HTTP methods — each decorated with the shared retry policy
    # ------------------------------------------------------------------

    @http_retry
    def _fetch_observations(self, series_id: str) -> dict[str, Any]:
        """Fetch raw observation data for a single FRED series.

        Args:
            series_id: FRED series identifier (e.g. ``"CPIAUCSL"``).

        Returns:
            Raw JSON response dict from the observations endpoint.

        Raises:
            requests.exceptions.HTTPError: On 4xx/5xx responses.
                Tenacity retries this up to 3 times before re-raising.
        """
        url = f"{_FRED_BASE_URL}/series/observations"
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        self._logger.debug("Fetching FRED observations for %s", series_id)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    @http_retry
    def _fetch_series_unit(self, series_id: str) -> str:
        """Fetch the unit label for a single FRED series.

        The observations endpoint does not include unit metadata, so a
        separate call to ``/fred/series`` is required.

        Args:
            series_id: FRED series identifier (e.g. ``"CPIAUCSL"``).

        Returns:
            Unit string (e.g. ``"Index 1982-1984=100"``, ``"Percent"``).

        Raises:
            requests.exceptions.HTTPError: On 4xx/5xx responses.
            KeyError: If the response structure is unexpected.
        """
        url = f"{_FRED_BASE_URL}/series"
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        self._logger.debug("Fetching FRED series metadata for %s", series_id)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()["seriess"][0]["units"]  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, series_ids: list[str] | None = None) -> pd.DataFrame:
        """Fetch observations for one or more FRED series.

        For each series, this method:

        1. Fetches the unit label via ``_fetch_series_unit()``.
        2. Fetches observation data via ``_fetch_observations()``.
        3. Validates the response with ``FREDResponse``.
        4. Drops observations where ``value == "."`` (FRED's null sentinel).
        5. Casts ``date`` to ``datetime`` and ``value`` to ``float``.

        All raw observation payloads are collected and saved together via
        ``save_raw()`` before the DataFrame is returned.

        Args:
            series_ids: FRED series identifiers to fetch. Defaults to
                ``DEFAULT_SERIES`` (``["CPIAUCSL", "DRCCLACBS"]``) if
                ``None``.

        Returns:
            DataFrame with columns:

            - ``date``      — ``datetime64[ns]``
            - ``series_id`` — ``str``
            - ``value``     — ``float64``
            - ``unit``      — ``str``

        Raises:
            requests.exceptions.HTTPError: After 3 failed HTTP attempts.
            pydantic.ValidationError: If the API response fails schema
                validation.
            ValueError: If ``FRED_API_KEY`` is not set (raised at init).
        """
        ids: list[str] = series_ids if series_ids is not None else self.DEFAULT_SERIES

        all_raw: dict[str, Any] = {}
        series_units: dict[str, str] = {}
        series_observations: dict[str, FREDResponse] = {}

        for series_id in ids:
            unit = self._fetch_series_unit(series_id)
            series_units[series_id] = unit

            raw = self._fetch_observations(series_id)
            all_raw[series_id] = raw

            series_observations[series_id] = FREDResponse.model_validate(raw)
            self._logger.info("Fetched %s observations for %s", len(series_observations[series_id].observations), series_id)

        self.save_raw(all_raw, source_name="fred")

        frames: list[pd.DataFrame] = []
        for series_id, validated in series_observations.items():
            unit = series_units[series_id]
            rows = [
                {
                    "date": pd.to_datetime(obs.date),
                    "series_id": series_id,
                    "value": float(obs.value),
                    "unit": unit,
                }
                for obs in validated.observations
                if obs.value != "."
            ]
            if rows:
                frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(columns=["date", "series_id", "value", "unit"])

        return pd.concat(frames, ignore_index=True)
