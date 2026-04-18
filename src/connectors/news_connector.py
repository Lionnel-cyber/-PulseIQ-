"""NewsAPI connector for PulseIQ.

Fetches news articles matching economic stress query terms from the NewsAPI
``/v2/everything`` endpoint, extracts city-level geography from article text,
and returns a tidy DataFrame ready for the dbt staging layer.

Default query terms:
    "unemployment layoffs", "food bank demand", "eviction crisis",
    "economic hardship", "job losses", "foreclosure"
"""

import logging
import os
import re
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, model_validator

from src.connectors.base_connector import BaseConnector, http_retry
from src.observability.metrics import IngestionMetrics, log_ingestion_metrics

logger = logging.getLogger(__name__)

_NEWSAPI_URL = "https://newsapi.org/v2/everything"

DEFAULT_QUERY_TERMS: list[str] = [
    "unemployment layoffs",
    "food bank demand",
    "eviction crisis",
    "economic hardship",
    "job losses",
    "foreclosure",
]

# Matches "in Detroit, MI" or "in New York, NY" (title-cased city, two-letter state)
_CITY_RE = re.compile(
    r"\bin ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*),\s*([A-Z]{2})\b"
)

# Expected output columns — used to build an empty DataFrame on failure
_OUTPUT_COLUMNS: list[str] = [
    "date",
    "geo_id",
    "geo_level",
    "geo_name",
    "headline",
    "description",
    "sentiment_score",
    "source_url",
]


# ---------------------------------------------------------------------------
# Pydantic validation models
# ---------------------------------------------------------------------------


class NewsArticle(BaseModel):
    """A single validated NewsAPI article.

    The NewsAPI response nests the source as ``{"id": ..., "name": ...}``.
    The ``_flatten_source`` validator hoists ``source.name`` to the top-level
    ``source_name`` field before field validation runs.

    Attributes:
        title: Article headline. May be ``None`` for some sources.
        description: Short article summary. May be ``None``.
        url: Canonical URL of the article.
        publishedAt: ISO 8601 publication timestamp (e.g.
            ``"2024-03-15T12:00:00Z"``).
        source_name: Human-readable source name (e.g. ``"BBC News"``).
    """

    title: str | None = None
    description: str | None = None
    url: str
    publishedAt: str
    source_name: str

    @model_validator(mode="before")
    @classmethod
    def _flatten_source(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Hoist ``source.name`` to top-level ``source_name``.

        Args:
            data: Raw dict from the NewsAPI response.

        Returns:
            Dict with ``source_name`` injected at the top level.
        """
        if isinstance(data.get("source"), dict):
            data["source_name"] = data["source"].get("name", "")
        return data


class NewsResponse(BaseModel):
    """Validated collection of NewsAPI articles.

    Attributes:
        articles: List of individual validated articles.
    """

    articles: list[NewsArticle]


# ---------------------------------------------------------------------------
# Geography extraction helper
# ---------------------------------------------------------------------------


def _extract_geo(text: str) -> tuple[str, str, str]:
    """Extract city-level geography from article text using a regex pattern.

    Searches for the pattern ``"in {City}, {ST}"`` (e.g. ``"in Detroit, MI"``).
    The city must start with an uppercase letter; the state must be exactly
    two uppercase letters.

    Args:
        text: Combined article title and description to search.

    Returns:
        A three-tuple of ``(geo_id, geo_level, geo_name)``:

        - If a city is found: ``("{City}-{ST}", "city", "{City}")``
        - If no city is found: ``("US", "national", "United States")``

    Examples:
        >>> _extract_geo("Layoffs spike in Detroit, MI")
        ('Detroit-MI', 'city', 'Detroit')
        >>> _extract_geo("Unemployment rises nationally")
        ('US', 'national', 'United States')
        >>> _extract_geo("Crisis in New York, NY worsens")
        ('New York-NY', 'city', 'New York')
    """
    match = _CITY_RE.search(text)
    if match:
        city = match.group(1).strip()
        state = match.group(2).strip()
        return f"{city}-{state}", "city", city
    return "US", "national", "United States"


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class NewsAPIConnector(BaseConnector):
    """Fetches economic stress news articles from the NewsAPI.

    Inherits retry logic and raw data persistence from ``BaseConnector``.
    Each call to ``fetch()`` queries all provided terms, de-duplicates
    articles by URL, validates with Pydantic, persists the raw JSON,
    writes ``IngestionMetrics``, and returns a tidy DataFrame.

    Args:
        raw_data_root: Root directory for raw data storage.
            Defaults to ``"data/raw"``. Pass ``tmp_path`` in tests.

    Raises:
        ValueError: At construction time if ``NEWS_API_KEY`` is not set.

    Example::

        connector = NewsAPIConnector()
        df = connector.fetch()                            # uses DEFAULT_QUERY_TERMS
        df = connector.fetch(["eviction crisis"])         # custom terms
    """

    def __init__(self, raw_data_root: str | Path = "data/raw") -> None:
        super().__init__(raw_data_root)
        self._api_key: str = os.getenv("NEWS_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "NEWS_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )

    # ------------------------------------------------------------------
    # Private HTTP methods
    # ------------------------------------------------------------------

    @http_retry
    def _fetch_articles(self, query: str) -> dict[str, Any]:
        """Fetch news articles for a single query term from NewsAPI.

        Args:
            query: Search string passed to the NewsAPI ``q`` parameter
                (e.g. ``"unemployment layoffs"``).

        Returns:
            Raw JSON response dict from the ``/v2/everything`` endpoint.

        Raises:
            requests.exceptions.HTTPError: On 4xx/5xx responses.
                Tenacity retries this up to 3 times before re-raising.
        """
        params: dict[str, str | int] = {
            "q": query,
            "apiKey": self._api_key,
            "language": "en",
            "pageSize": 100,
            "sortBy": "publishedAt",
        }
        self._logger.debug("Fetching NewsAPI articles for query: %r", query)
        response = requests.get(_NEWSAPI_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, query_terms: list[str] | None = None) -> pd.DataFrame:
        """Fetch economic stress articles for all query terms.

        For each query term this method:

        1. Calls ``_fetch_articles()`` (decorated with ``@http_retry``).
        2. Collects raw article dicts, de-duplicating by URL across terms.
        3. Validates all articles with ``NewsResponse``.
        4. Extracts city-level geography from each article's title and
           description via ``_extract_geo()``.
        5. Saves the full raw payload via ``save_raw()``.
        6. Writes ``IngestionMetrics`` via structured logging.

        Args:
            query_terms: List of search strings to query. Defaults to
                ``DEFAULT_QUERY_TERMS`` if ``None``.

        Returns:
            DataFrame with columns:

            - ``date``            — ``datetime64[ns]``, parsed from ``publishedAt``
            - ``geo_id``          — ``str``, ``"{City}-{ST}"`` or ``"US"``
            - ``geo_level``       — ``str``, ``"city"`` or ``"national"``
            - ``geo_name``        — ``str``, city name or ``"United States"``
            - ``headline``        — ``str``, article title (empty string if ``None``)
            - ``description``     — ``str``, article summary (empty string if ``None``)
            - ``sentiment_score`` — ``float``, placeholder ``0.0`` (computed in dbt)
            - ``source_url``      — ``str``, canonical article URL

        Raises:
            requests.exceptions.HTTPError: After 3 failed HTTP attempts per term.
            pydantic.ValidationError: If the API response fails schema validation.
            ValueError: If ``NEWS_API_KEY`` is not set (raised at init).
        """
        terms: list[str] = query_terms if query_terms is not None else DEFAULT_QUERY_TERMS
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        start_time = time.monotonic()
        error_message: str | None = None
        success = True

        # Collect and de-duplicate raw article dicts across all query terms
        seen_urls: set[str] = set()
        all_raw_articles: list[dict[str, Any]] = []
        queries_used: list[str] = []

        try:
            for term in terms:
                raw = self._fetch_articles(term)
                queries_used.append(term)
                for article in raw.get("articles", []):
                    url = article.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_raw_articles.append(article)

            validated = NewsResponse.model_validate({"articles": all_raw_articles})
            self._logger.info(
                "Fetched %d unique articles across %d query terms",
                len(validated.articles),
                len(terms),
            )

            self.save_raw(
                {"queries": queries_used, "articles": all_raw_articles},
                source_name="news",
            )

            rows: list[dict[str, Any]] = []
            for article in validated.articles:
                text = (article.title or "") + " " + (article.description or "")
                geo_id, geo_level, geo_name = _extract_geo(text)
                rows.append(
                    {
                        "date": pd.to_datetime(article.publishedAt, utc=True).tz_localize(None),
                        "geo_id": geo_id,
                        "geo_level": geo_level,
                        "geo_name": geo_name,
                        "headline": article.title or "",
                        "description": article.description or "",
                        "sentiment_score": 0.0,
                        "source_url": article.url,
                    }
                )

            df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS) if rows else pd.DataFrame(columns=_OUTPUT_COLUMNS)

        except Exception as exc:
            success = False
            error_message = str(exc)
            self._logger.error("NewsAPI fetch failed: %s", error_message)
            raise

        finally:
            completed_at = datetime.now(timezone.utc)
            latency = time.monotonic() - start_time
            log_ingestion_metrics(
                IngestionMetrics(
                    source="news",
                    run_date=date.today(),
                    run_id=run_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    records_fetched=len(all_raw_articles),
                    records_rejected=0,
                    records_suspect=0,
                    latency_seconds=round(latency, 3),
                    freshness_status="ok" if success else "unknown",
                    http_retries=0,
                    success=success,
                    error_message=error_message,
                ),
                self._logger,
            )

        return df
