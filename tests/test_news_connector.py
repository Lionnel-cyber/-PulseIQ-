"""Tests for src/connectors/news_connector.py.

All HTTP calls are intercepted by the ``responses`` library — no live API calls.
``monkeypatch`` injects a dummy NEWS_API_KEY for every test that needs one.
``patch("time.sleep")`` prevents tenacity from actually sleeping during retry tests.
"""

from unittest.mock import patch

import pandas as pd
import pytest
import requests
import responses as responses_lib

from src.connectors.news_connector import NewsAPIConnector, _extract_geo

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NEWSAPI_URL = "https://newsapi.org/v2/everything"

MOCK_CITY_ARTICLE = {
    "title": "Layoffs spike in Detroit, MI amid plant closures",
    "description": "Workers in Detroit, MI face economic hardship.",
    "url": "https://example.com/article1",
    "publishedAt": "2024-03-15T12:00:00Z",
    "source": {"id": "bbc", "name": "BBC News"},
}

MOCK_NATIONAL_ARTICLE = {
    "title": "Unemployment rises nationally",
    "description": "No city mentioned in this article.",
    "url": "https://example.com/article2",
    "publishedAt": "2024-03-15T11:00:00Z",
    "source": {"id": None, "name": "Reuters"},
}

MOCK_CITY_RESPONSE = {
    "status": "ok",
    "totalResults": 1,
    "articles": [MOCK_CITY_ARTICLE],
}

MOCK_NATIONAL_RESPONSE = {
    "status": "ok",
    "totalResults": 1,
    "articles": [MOCK_NATIONAL_ARTICLE],
}

MOCK_EMPTY_RESPONSE = {
    "status": "ok",
    "totalResults": 0,
    "articles": [],
}

# ---------------------------------------------------------------------------
# Unit tests — _extract_geo (no HTTP needed)
# ---------------------------------------------------------------------------


def test_extract_geo_city_found() -> None:
    """_extract_geo returns city geo tuple when 'in {City}, {ST}' is present."""
    geo_id, geo_level, geo_name = _extract_geo("Evictions rise in Chicago, IL this month")
    assert geo_id == "Chicago-IL"
    assert geo_level == "city"
    assert geo_name == "Chicago"


def test_extract_geo_no_match() -> None:
    """_extract_geo returns national fallback when no city pattern is found."""
    geo_id, geo_level, geo_name = _extract_geo("Unemployment rises across the country")
    assert geo_id == "US"
    assert geo_level == "national"
    assert geo_name == "United States"


def test_extract_geo_multi_word_city() -> None:
    """_extract_geo correctly parses multi-word city names."""
    geo_id, geo_level, geo_name = _extract_geo("Food banks strained in New York, NY")
    assert geo_id == "New York-NY"
    assert geo_level == "city"
    assert geo_name == "New York"


# ---------------------------------------------------------------------------
# Integration tests — NewsAPIConnector
# ---------------------------------------------------------------------------


def test_raises_if_api_key_missing() -> None:
    """NewsAPIConnector raises ValueError at construction if NEWS_API_KEY is unset."""
    import os

    saved = os.environ.pop("NEWS_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="NEWS_API_KEY"):
            NewsAPIConnector()
    finally:
        if saved is not None:
            os.environ["NEWS_API_KEY"] = saved


@responses_lib.activate
def test_correct_dataframe_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """fetch() returns a DataFrame with all 8 expected columns and correct dtypes."""
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, json=MOCK_CITY_RESPONSE, status=200)

    from pathlib import Path

    connector = NewsAPIConnector(raw_data_root=Path(str(tmp_path)))
    df = connector.fetch(query_terms=["unemployment layoffs"])

    assert list(df.columns) == [
        "date",
        "geo_id",
        "geo_level",
        "geo_name",
        "headline",
        "description",
        "sentiment_score",
        "source_url",
    ]
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["sentiment_score"].dtype == float
    assert df["sentiment_score"].iloc[0] == pytest.approx(0.0)
    assert df["source_url"].iloc[0] == "https://example.com/article1"


@responses_lib.activate
def test_city_extraction_from_article_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """fetch() sets geo_id, geo_level, and geo_name from city found in article text."""
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, json=MOCK_CITY_RESPONSE, status=200)

    from pathlib import Path

    connector = NewsAPIConnector(raw_data_root=Path(str(tmp_path)))
    df = connector.fetch(query_terms=["unemployment layoffs"])

    assert df["geo_id"].iloc[0] == "Detroit-MI"
    assert df["geo_level"].iloc[0] == "city"
    assert df["geo_name"].iloc[0] == "Detroit"


@responses_lib.activate
def test_fallback_to_national_when_no_city(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """fetch() falls back to national geo when no city pattern is found in article."""
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, json=MOCK_NATIONAL_RESPONSE, status=200)

    from pathlib import Path

    connector = NewsAPIConnector(raw_data_root=Path(str(tmp_path)))
    df = connector.fetch(query_terms=["unemployment layoffs"])

    assert df["geo_id"].iloc[0] == "US"
    assert df["geo_level"].iloc[0] == "national"
    assert df["geo_name"].iloc[0] == "United States"


@responses_lib.activate
def test_deduplication_across_query_terms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """fetch() returns each article URL only once even if it appears in multiple query results."""
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    # Both query terms return the same article URL
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, json=MOCK_CITY_RESPONSE, status=200)
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, json=MOCK_CITY_RESPONSE, status=200)

    from pathlib import Path

    connector = NewsAPIConnector(raw_data_root=Path(str(tmp_path)))
    df = connector.fetch(query_terms=["unemployment layoffs", "job losses"])

    assert len(df) == 1
    assert df["source_url"].iloc[0] == "https://example.com/article1"


@responses_lib.activate
def test_retries_on_429(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """fetch() retries after a 429 and returns data on subsequent success."""
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    # First two calls fail with 429, third succeeds
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, status=429)
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, status=429)
    responses_lib.add(responses_lib.GET, NEWSAPI_URL, json=MOCK_CITY_RESPONSE, status=200)

    from pathlib import Path

    with patch("time.sleep"):
        connector = NewsAPIConnector(raw_data_root=Path(str(tmp_path)))
        df = connector.fetch(query_terms=["unemployment layoffs"])

    assert len(df) == 1


@responses_lib.activate
def test_raises_after_max_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """fetch() raises HTTPError after all 3 retry attempts fail."""
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    for _ in range(3):
        responses_lib.add(responses_lib.GET, NEWSAPI_URL, status=500)

    from pathlib import Path

    with patch("time.sleep"):
        connector = NewsAPIConnector(raw_data_root=Path(str(tmp_path)))
        with pytest.raises(requests.exceptions.HTTPError):
            connector.fetch(query_terms=["unemployment layoffs"])
