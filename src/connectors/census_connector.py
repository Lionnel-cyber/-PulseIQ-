"""Census ACS 5-Year connector for PulseIQ.

Fetches ZIP Code Tabulation Area (ZCTA) estimates from the Census API and
returns a tidy DataFrame ready for the dbt staging layer.

The connector caches the raw national ACS payload in ``data/raw`` and only
re-fetches from the API when the newest cached payload is more than 30 days
old. Optional ``state_fips`` filtering is applied locally after the national
ZCTA payload is loaded.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, Field

from src.connectors.base_connector import BaseConnector, http_retry

logger = logging.getLogger(__name__)

_CENSUS_API_BASE_URL = "https://api.census.gov/data"
_CACHE_MAX_AGE_DAYS = 30
_DATASET_YEAR = 2023
_SOURCE_NAME = "census"
_ACS_VARIABLES: list[str] = [
    "NAME",
    "B17001_002E",
    "B17001_001E",
    "B19013_001E",
    "B25070_010E",
    "B25070_001E",
]
_GEOGRAPHY_FIELD = "zip code tabulation area"
_MISSING_SENTINELS: set[str] = {
    "",
    "-666666666",
    "-999999999",
    "-888888888",
    "null",
}

_STATE_ABBR_TO_FIPS: dict[str, str] = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
    "PR": "72",
}

_ZIP3_STATE_RANGES: list[tuple[int, int, str]] = [
    (6, 9, "PR"),
    (10, 27, "MA"),
    (28, 29, "RI"),
    (30, 38, "NH"),
    (39, 49, "ME"),
    (50, 59, "VT"),
    (60, 69, "CT"),
    (70, 89, "NJ"),
    (100, 149, "NY"),
    (150, 196, "PA"),
    (197, 199, "DE"),
    (200, 205, "DC"),
    (206, 219, "MD"),
    (220, 246, "VA"),
    (247, 268, "WV"),
    (270, 289, "NC"),
    (290, 299, "SC"),
    (300, 319, "GA"),
    (320, 349, "FL"),
    (350, 369, "AL"),
    (370, 385, "TN"),
    (386, 397, "MS"),
    (398, 399, "GA"),
    (400, 427, "KY"),
    (430, 459, "OH"),
    (460, 479, "IN"),
    (480, 499, "MI"),
    (500, 528, "IA"),
    (530, 549, "WI"),
    (550, 567, "MN"),
    (570, 577, "SD"),
    (580, 588, "ND"),
    (590, 599, "MT"),
    (600, 629, "IL"),
    (630, 658, "MO"),
    (660, 679, "KS"),
    (680, 693, "NE"),
    (700, 715, "LA"),
    (716, 729, "AR"),
    (730, 749, "OK"),
    (750, 799, "TX"),
    (800, 816, "CO"),
    (820, 831, "WY"),
    (832, 838, "ID"),
    (840, 847, "UT"),
    (850, 865, "AZ"),
    (870, 884, "NM"),
    (885, 885, "TX"),
    (889, 898, "NV"),
    (900, 961, "CA"),
    (967, 968, "HI"),
    (970, 979, "OR"),
    (980, 994, "WA"),
    (995, 999, "AK"),
]


def _normalise_numeric(value: str | None) -> float:
    """Convert Census numeric strings into floats, preserving missing values.

    Args:
        value: Raw string value returned by the Census API.

    Returns:
        Parsed float value or ``nan`` for Census missing sentinels.
    """
    if value is None:
        return float("nan")

    stripped = value.strip()
    if stripped in _MISSING_SENTINELS:
        return float("nan")

    return float(stripped)


def _safe_rate(numerator: float, denominator: float) -> float:
    """Compute a ratio while guarding against null or zero denominators."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return float("nan")
    return numerator / denominator


def _zip_to_state_fips(zip_code: str) -> str | None:
    """Infer state FIPS from a ZIP code using bundled ZIP3 ranges.

    Args:
        zip_code: Five-digit ZIP code string.

    Returns:
        State FIPS code if the ZIP prefix is recognised, otherwise ``None``.
    """
    if len(zip_code) < 3 or not zip_code[:3].isdigit():
        return None

    zip3 = int(zip_code[:3])
    for start, end, state_abbr in _ZIP3_STATE_RANGES:
        if start <= zip3 <= end:
            return _STATE_ABBR_TO_FIPS.get(state_abbr)
    return None


class CensusACSRow(BaseModel):
    """Validated ACS row for a single ZIP Code Tabulation Area."""

    name: str = Field(alias="NAME")
    poverty_count: str = Field(alias="B17001_002E")
    poverty_universe: str = Field(alias="B17001_001E")
    median_income: str = Field(alias="B19013_001E")
    housing_burden_50_plus: str = Field(alias="B25070_010E")
    housing_burden_universe: str = Field(alias="B25070_001E")
    zip_code: str = Field(alias=_GEOGRAPHY_FIELD)


class CensusACSResponse(BaseModel):
    """Validated tabular ACS payload parsed from the Census API response."""

    header: list[str]
    rows: list[CensusACSRow]

    @classmethod
    def from_api_payload(cls, payload: list[list[str]]) -> "CensusACSResponse":
        """Validate and parse the Census header-plus-rows array response.

        Args:
            payload: Raw JSON payload returned by the Census API.

        Returns:
            Parsed ``CensusACSResponse`` instance.

        Raises:
            ValueError: If the payload is malformed or missing required columns.
        """
        if not payload or not isinstance(payload, list):
            raise ValueError("Census API payload must be a non-empty list.")

        header = payload[0]
        if not isinstance(header, list):
            raise ValueError("Census API header row must be a list of strings.")

        required_columns = set(_ACS_VARIABLES + [_GEOGRAPHY_FIELD])
        missing_columns = required_columns.difference(header)
        if missing_columns:
            raise ValueError(
                "Census API payload is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        rows: list[CensusACSRow] = []
        for row in payload[1:]:
            if len(row) != len(header):
                raise ValueError(
                    "Census API row length does not match the header length."
                )
            rows.append(CensusACSRow.model_validate(dict(zip(header, row))))

        return cls(header=header, rows=rows)


class CensusConnector(BaseConnector):
    """Fetch ACS 5-Year ZCTA estimates from the Census API.

    The connector always caches and reuses the latest national ZIP-level ACS
    payload for up to 30 days. Optional ``state_fips`` filtering is applied
    after cache load or fresh fetch.

    Args:
        raw_data_root: Root directory for raw data storage.
            Defaults to ``"data/raw"``. Pass ``tmp_path`` in tests.

    Raises:
        ValueError: If ``CENSUS_API_KEY`` is not set.
    """

    DATASET_YEAR: int = _DATASET_YEAR

    def __init__(self, raw_data_root: str | Path = "data/raw") -> None:
        super().__init__(raw_data_root)
        self._api_key: str = os.getenv("CENSUS_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "CENSUS_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )

    @http_retry
    def _fetch_acs_data(self) -> dict[str, Any]:
        """Fetch national ACS 5-Year ZCTA data from the Census API.

        Returns:
            Raw payload wrapper containing the dataset year and API rows.

        Raises:
            requests.exceptions.HTTPError: On 4xx/5xx responses.
        """
        url = f"{_CENSUS_API_BASE_URL}/{self.DATASET_YEAR}/acs/acs5"
        params: dict[str, str] = {
            "get": ",".join(_ACS_VARIABLES),
            "for": f"{_GEOGRAPHY_FIELD}:*",
            "key": self._api_key,
        }

        self._logger.debug("Fetching Census ACS 5-Year ZCTA dataset for %s", self.DATASET_YEAR)
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return {
            "dataset_year": self.DATASET_YEAR,
            "rows": response.json(),
        }

    def _find_latest_cache_file(self) -> Path | None:
        """Find the newest cached raw Census payload in ``data/raw``."""
        latest_path: Path | None = None
        latest_date: date | None = None

        for path in self._raw_data_root.glob("*/*/*/census/data.json"):
            try:
                relative_parts = path.relative_to(self._raw_data_root).parts
                file_date = date(
                    int(relative_parts[0]),
                    int(relative_parts[1]),
                    int(relative_parts[2]),
                )
            except (ValueError, IndexError):
                continue

            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_path = path

        return latest_path

    def _is_cache_fresh(self, cache_path: Path) -> bool:
        """Check whether a cached raw Census payload is 30 days old or newer."""
        relative_parts = cache_path.relative_to(self._raw_data_root).parts
        cache_date = date(
            int(relative_parts[0]),
            int(relative_parts[1]),
            int(relative_parts[2]),
        )
        return date.today() - cache_date <= timedelta(days=_CACHE_MAX_AGE_DAYS)

    def _load_cached_payload(self, cache_path: Path) -> dict[str, Any]:
        """Load a cached raw Census payload from disk."""
        with cache_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)

        if not isinstance(payload, dict) or "rows" not in payload:
            raise ValueError(f"Cached Census payload is invalid: {cache_path}")

        return payload

    def _load_or_fetch_payload(self) -> dict[str, Any]:
        """Return a fresh or cached raw Census payload."""
        cache_path = self._find_latest_cache_file()
        if cache_path is not None and self._is_cache_fresh(cache_path):
            self._logger.info("Using cached Census data from %s", cache_path)
            return self._load_cached_payload(cache_path)

        payload = self._fetch_acs_data()
        validated = CensusACSResponse.from_api_payload(payload["rows"])
        self._logger.info("Fetched %d Census ZCTA rows", len(validated.rows))
        self.save_raw(payload, source_name=_SOURCE_NAME)
        return payload

    def _to_dataframe(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Transform a raw Census payload into the connector output schema."""
        validated = CensusACSResponse.from_api_payload(payload["rows"])
        dataset_year = int(payload.get("dataset_year", self.DATASET_YEAR))

        rows: list[dict[str, Any]] = []
        for row in validated.rows:
            zip_code = row.zip_code.zfill(5).strip()
            if not zip_code:
                continue

            poverty_count = _normalise_numeric(row.poverty_count)
            poverty_universe = _normalise_numeric(row.poverty_universe)
            median_income = _normalise_numeric(row.median_income)
            housing_burden_50_plus = _normalise_numeric(row.housing_burden_50_plus)
            housing_burden_universe = _normalise_numeric(row.housing_burden_universe)

            rows.append(
                {
                    "zip_code": zip_code,
                    "poverty_rate": _safe_rate(poverty_count, poverty_universe),
                    "median_income": median_income,
                    "housing_cost_burden": _safe_rate(
                        housing_burden_50_plus,
                        housing_burden_universe,
                    ),
                    "vintage_year": dataset_year,
                    "geo_id": zip_code,
                    "geo_level": "zip",
                    "geo_name": f"ZIP {zip_code}",
                    "data_note": "annual baseline only",
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
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
            )

        return pd.DataFrame(rows)

    def _filter_by_state_fips(
        self, df: pd.DataFrame, state_fips: list[str] | None
    ) -> pd.DataFrame:
        """Filter ZIP rows locally using bundled ZIP-prefix state lookup."""
        if state_fips is None:
            return df

        requested = {value.zfill(2) for value in state_fips}
        if not requested:
            return df

        state_codes = df["zip_code"].map(_zip_to_state_fips)
        return df.loc[state_codes.isin(requested)].reset_index(drop=True)

    def fetch(self, state_fips: list[str] | None = None) -> pd.DataFrame:
        """Fetch ACS 5-Year ZIP-level poverty, income, and housing data.

        The connector loads the newest cached national ACS payload if it is no
        more than 30 days old. Otherwise it fetches fresh data from the Census
        API, validates it, stores the raw payload, and transforms it into a
        tidy DataFrame. Optional ``state_fips`` filtering is applied locally.

        Args:
            state_fips: Optional list of state FIPS codes used to filter the
                national ZIP-level result after load. If ``None``, all ZIPs are
                returned.

        Returns:
            DataFrame with columns:

            - ``zip_code``            — ``str``
            - ``poverty_rate``        — ``float64``
            - ``median_income``       — ``float64``
            - ``housing_cost_burden`` — ``float64``
            - ``vintage_year``        — ``int64``
            - ``geo_id``              — ``str`` (same value as ``zip_code``)
            - ``geo_level``           — ``str`` (always ``"zip"``)
            - ``geo_name``            — ``str`` (e.g. ``"ZIP 48201"``)
            - ``data_note``           — ``str`` (always ``"annual baseline only"``)

        Raises:
            requests.exceptions.HTTPError: After 3 failed HTTP attempts.
            ValueError: If ``CENSUS_API_KEY`` is not set or cached/API payloads
                are malformed.
            pydantic.ValidationError: If row validation fails.
        """
        payload = self._load_or_fetch_payload()
        df = self._to_dataframe(payload)
        return self._filter_by_state_fips(df, state_fips)
