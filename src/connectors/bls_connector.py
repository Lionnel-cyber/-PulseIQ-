"""BLS Public Data API v2 connector for PulseIQ.

Fetches economic time-series data (weekly initial jobless claims, unemployment
rates) from the Bureau of Labor Statistics Public Data API and returns a tidy
DataFrame ready for the dbt staging layer.

Key difference from other connectors: the BLS API accepts multiple series IDs
in a single POST request, so all series are fetched in one HTTP call.

Default series:
    LNS14000000  — Unemployment Rate (Seasonally Adjusted, national)
"""

import logging
import os
from datetime import date as _date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel

from src.connectors.base_connector import BaseConnector, http_retry

logger = logging.getLogger(__name__)

_BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# ---------------------------------------------------------------------------
# State FIPS → abbreviation lookup (used by _derive_state)
# ---------------------------------------------------------------------------

_FIPS_TO_STATE_NAME: dict[str, str] = {
    "01": "Alabama",              "02": "Alaska",
    "04": "Arizona",              "05": "Arkansas",
    "06": "California",           "08": "Colorado",
    "09": "Connecticut",          "10": "Delaware",
    "11": "District of Columbia", "12": "Florida",
    "13": "Georgia",              "15": "Hawaii",
    "16": "Idaho",                "17": "Illinois",
    "18": "Indiana",              "19": "Iowa",
    "20": "Kansas",               "21": "Kentucky",
    "22": "Louisiana",            "23": "Maine",
    "24": "Maryland",             "25": "Massachusetts",
    "26": "Michigan",             "27": "Minnesota",
    "28": "Mississippi",          "29": "Missouri",
    "30": "Montana",              "31": "Nebraska",
    "32": "Nevada",               "33": "New Hampshire",
    "34": "New Jersey",           "35": "New Mexico",
    "36": "New York",             "37": "North Carolina",
    "38": "North Dakota",         "39": "Ohio",
    "40": "Oklahoma",             "41": "Oregon",
    "42": "Pennsylvania",         "44": "Rhode Island",
    "45": "South Carolina",       "46": "South Dakota",
    "47": "Tennessee",            "48": "Texas",
    "49": "Utah",                 "50": "Vermont",
    "51": "Virginia",             "53": "Washington",
    "54": "West Virginia",        "55": "Wisconsin",
    "56": "Wyoming",
}

# Top-20 US metros by population (CBSA codes → full MSA name).
# Any CBSA code not listed here falls back to "MSA {cbsa}".
_MSA_CODE_TO_NAME: dict[str, str] = {
    "35620": "New York-Newark-Jersey City, NY-NJ-PA",
    "31080": "Los Angeles-Long Beach-Anaheim, CA",
    "16980": "Chicago-Naperville-Elgin, IL-IN-WI",
    "19100": "Dallas-Fort Worth-Arlington, TX",
    "26420": "Houston-The Woodlands-Sugar Land, TX",
    "47900": "Washington-Arlington-Alexandria, DC-VA-MD-WV",
    "33100": "Miami-Fort Lauderdale-Pompano Beach, FL",
    "37980": "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD",
    "12060": "Atlanta-Sandy Springs-Alpharetta, GA",
    "38060": "Phoenix-Mesa-Chandler, AZ",
    "14460": "Boston-Cambridge-Newton, MA-NH",
    "40140": "Riverside-San Bernardino-Ontario, CA",
    "42660": "Seattle-Tacoma-Bellevue, WA",
    "33460": "Minneapolis-St. Paul-Bloomington, MN-WI",
    "41740": "San Diego-Chula Vista-Carlsbad, CA",
    "45300": "Tampa-St. Petersburg-Clearwater, FL",
    "19740": "Denver-Aurora-Lakewood, CO",
    "41180": "St. Louis, MO-IL",
    "12580": "Baltimore-Columbia-Towson, MD",
    "36740": "Orlando-Kissimmee-Sanford, FL",
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_date(year: str, period: str) -> pd.Timestamp:
    """Convert a BLS year + period code into a ``pd.Timestamp``.

    BLS encodes dates as separate ``year`` and ``period`` fields. This function
    normalises all supported period types to the first day of the represented
    interval.

    Period formats handled:

    - ``M01``–``M12`` — monthly → first day of that month
    - ``W01``–``W53`` — weekly  → Monday of that ISO week
    - ``Q01``–``Q04`` — quarterly → first day of first month of the quarter
    - ``A01``         — annual  → January 1 of that year

    Args:
        year: Four-digit year string (e.g. ``"2024"``).
        period: BLS period code (e.g. ``"M01"``, ``"W03"``, ``"Q02"``).

    Returns:
        ``pd.Timestamp`` representing the start of the period.

    Raises:
        ValueError: If the period prefix is not one of ``M``, ``W``, ``Q``, ``A``.
    """
    prefix = period[0]
    num = int(period[1:])
    yr = int(year)

    if prefix == "M":
        return pd.Timestamp(f"{yr}-{num:02d}-01")
    if prefix == "W":
        # ISO week date: Monday (weekday=1) of the given week
        return pd.Timestamp(_date.fromisocalendar(yr, num, 1))
    if prefix == "Q":
        month = (num - 1) * 3 + 1
        return pd.Timestamp(f"{yr}-{month:02d}-01")
    if prefix == "A":
        return pd.Timestamp(f"{yr}-01-01")

    raise ValueError(f"Unrecognised BLS period prefix '{prefix}' in '{period}'")


def _derive_geo(series_id: str) -> tuple[str, str, str]:
    """Derive geo_id, geo_level, and geo_name from a BLS series ID.

    Rules applied in order:

    1. ``LASST`` prefix — Local Area Statistics, State level. Characters 5–6
       are the two-digit state FIPS code.
    2. ``LAUMT`` prefix — Local Area Unemployment Statistics, Metro level.
       Characters 5–9 are the five-digit CBSA code.
    3. All other series are treated as national.

    Args:
        series_id: BLS series identifier (e.g. ``"LASST060000000000003"``).

    Returns:
        Tuple of ``(geo_id, geo_level, geo_name)`` where:

        - ``geo_id``    — state FIPS, five-digit CBSA code, or ``"US"``
        - ``geo_level`` — ``"state"``, ``"metro"``, or ``"national"``
        - ``geo_name``  — full state name, resolved MSA name (or
          ``"MSA {cbsa}"`` for unlisted codes), or ``"United States"``
    """
    if series_id.startswith("LASST"):
        fips = series_id[5:7]
        name = _FIPS_TO_STATE_NAME.get(fips, f"State FIPS {fips}")
        return fips, "state", name
    if series_id.startswith("LAUMT"):
        cbsa = series_id[5:10]
        name = _MSA_CODE_TO_NAME.get(cbsa, f"MSA {cbsa}")
        return cbsa, "metro", name
    return "US", "national", "United States"


# ---------------------------------------------------------------------------
# Pydantic validation models
# ---------------------------------------------------------------------------


class BLSDataPoint(BaseModel):
    """A single BLS observation data point.

    All fields are kept as ``str`` exactly as returned by the API.
    Type conversion happens during DataFrame construction, not here.

    Attributes:
        year: Four-digit observation year.
        period: BLS period code (e.g. ``"M01"``, ``"W03"``).
        value: Observed value as a string.
        periodName: Human-readable period label (e.g. ``"January"``).
    """

    year: str
    period: str
    value: str
    periodName: str


class BLSSeries(BaseModel):
    """A single BLS time series with its data points.

    Attributes:
        seriesID: BLS series identifier (e.g. ``"LNS14000000"``).
        data: List of observation data points, newest-first as returned by the API.
    """

    seriesID: str
    data: list[BLSDataPoint]


class _BLSResults(BaseModel):
    series: list[BLSSeries]


class _BLSResponse(BaseModel):
    status: str
    Results: _BLSResults


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class BLSConnector(BaseConnector):
    """Fetches economic time-series data from the BLS Public Data API v2.

    Inherits retry logic and raw data persistence from ``BaseConnector``.
    All series are retrieved in a single POST request. The response is
    validated with Pydantic, persisted as raw JSON, and returned as a
    tidy DataFrame.

    Args:
        raw_data_root: Root directory for raw data storage.
            Defaults to ``"data/raw"``. Pass ``tmp_path`` in tests.

    Raises:
        ValueError: At construction time if ``BLS_API_KEY`` is not set.

    Example::

        connector = BLSConnector()
        df = connector.fetch()
        df = connector.fetch(["LNS14000000", "LASST060000000000003"])
    """

    DEFAULT_SERIES: list[str] = ["LNS14000000"]

    def __init__(self, raw_data_root: str | Path = "data/raw") -> None:
        super().__init__(raw_data_root)
        self._api_key: str = os.getenv("BLS_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "BLS_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )

    # ------------------------------------------------------------------
    # Private HTTP method
    # ------------------------------------------------------------------

    @http_retry
    def _fetch_series(self, series_ids: list[str]) -> dict[str, Any]:
        """POST all series IDs to the BLS API in a single request.

        The BLS v2 API accepts up to 50 series per request when a
        registration key is provided.

        Args:
            series_ids: List of BLS series identifiers to fetch.

        Returns:
            Raw JSON response dict from the API.

        Raises:
            requests.exceptions.HTTPError: On 4xx/5xx responses.
                Tenacity retries this up to 3 times before re-raising.
        """
        payload: dict[str, Any] = {
            "seriesid": series_ids,
            "registrationkey": self._api_key,
        }
        self._logger.debug("Fetching BLS series: %s", series_ids)
        response = requests.post(_BLS_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, series_ids: list[str] | None = None) -> pd.DataFrame:
        """Fetch observations for one or more BLS series.

        All series are requested in a single POST call. For each series
        this method:

        1. POSTs all series IDs to the BLS v2 endpoint.
        2. Checks that ``status == "REQUEST_SUCCEEDED"``; raises if not.
        3. Validates the response with ``_BLSResponse``.
        4. Persists the raw payload via ``save_raw()``.
        5. Derives a state label for each series via ``_derive_state()``.
        6. Parses BLS year + period codes into ``datetime`` via ``_parse_date()``.
        7. Casts ``value`` to ``float``.

        Args:
            series_ids: BLS series identifiers to fetch. Defaults to
                ``DEFAULT_SERIES`` (``["LNS14000000"]``) if ``None``.

        Returns:
            DataFrame with columns:

            - ``date``      — ``datetime64[ns]``
            - ``series_id`` — ``str``
            - ``value``     — ``float64``
            - ``geo_id``    — ``str`` (state FIPS, CBSA code, or ``"US"``)
            - ``geo_level`` — ``str`` (``"state"``, ``"metro"``, or ``"national"``)
            - ``geo_name``  — ``str`` (full state name, MSA name, or ``"United States"``)

        Raises:
            requests.exceptions.HTTPError: After 3 failed HTTP attempts.
            ValueError: If the API returns a non-success status, or if
                ``BLS_API_KEY`` is not set (raised at init).
            pydantic.ValidationError: If the response fails schema validation.
        """
        ids: list[str] = series_ids if series_ids is not None else self.DEFAULT_SERIES

        raw = self._fetch_series(ids)

        if raw.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(
                f"BLS API returned non-success status: {raw.get('status')!r}. "
                f"Messages: {raw.get('message', [])}"
            )

        self.save_raw(raw, source_name="bls")

        validated = _BLSResponse.model_validate(raw)
        self._logger.info(
            "Fetched %d BLS series", len(validated.Results.series)
        )

        frames: list[pd.DataFrame] = []
        for series in validated.Results.series:
            geo_id, geo_level, geo_name = _derive_geo(series.seriesID)
            rows: list[dict[str, Any]] = []
            skipped = 0
            for dp in series.data:
                try:
                    value = float(dp.value)
                except ValueError:
                    skipped += 1
                    continue

                rows.append(
                    {
                        "date": _parse_date(dp.year, dp.period),
                        "series_id": series.seriesID,
                        "value": value,
                        "geo_id": geo_id,
                        "geo_level": geo_level,
                        "geo_name": geo_name,
                    }
                )

            if skipped:
                self._logger.warning(
                    "Skipped %d non-numeric BLS observations for %s",
                    skipped,
                    series.seriesID,
                )
            if rows:
                frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(columns=["date", "series_id", "value", "geo_id", "geo_level", "geo_name"])

        return pd.concat(frames, ignore_index=True)
