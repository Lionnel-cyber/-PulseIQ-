"""Tests for src/rag/rss_ingest.py."""

from unittest.mock import MagicMock, patch

from src.rag.rss_ingest import (
    PULSEIQ_RSS_FEEDS,
    fetch_feed,
    ingest_rss_feeds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    link: str = "http://example.com/1",
    title: str = "Title 1",
    summary: str = "Summary 1",
    published: str = "Mon, 01 Jan 2026 00:00:00 +0000",
) -> MagicMock:
    """Return a mock feedparser entry that responds to .get() like a dict."""
    entry = MagicMock()
    data = {"link": link, "title": title, "summary": summary, "published": published}
    entry.get.side_effect = data.get
    return entry


def _make_parsed(entries: list, feed_title: str = "Mock Feed") -> MagicMock:
    """Return a mock feedparser result object."""
    parsed = MagicMock()
    parsed.feed.get.side_effect = {"title": feed_title}.get
    parsed.entries = entries
    return parsed


# ---------------------------------------------------------------------------
# fetch_feed
# ---------------------------------------------------------------------------


def test_fetch_feed_success():
    entries = [
        _make_entry("http://ex.com/1", "Title 1"),
        _make_entry("http://ex.com/2", "Title 2"),
        _make_entry("http://ex.com/3", "Title 3"),
    ]
    with patch("feedparser.parse", return_value=_make_parsed(entries)):
        result = fetch_feed("http://fake-feed.com/rss")

    assert len(result) == 3
    for article in result:
        assert set(article.keys()) == {"url", "title", "description", "publishedAt", "source"}


def test_fetch_feed_skips_empty_url():
    entries = [_make_entry(link="", title="Has Title But No URL")]
    with patch("feedparser.parse", return_value=_make_parsed(entries)):
        result = fetch_feed("http://fake-feed.com/rss")

    assert result == []


def test_fetch_feed_handles_exception():
    with patch("feedparser.parse", side_effect=Exception("network error")):
        result = fetch_feed("http://fake-feed.com/rss")

    assert result == []


# ---------------------------------------------------------------------------
# ingest_rss_feeds
# ---------------------------------------------------------------------------


def test_ingest_rss_feeds_returns_count():
    articles = [
        {"url": f"http://ex.com/{i}", "title": f"T{i}",
         "description": "", "publishedAt": "", "source": "X"}
        for i in range(5)
    ]
    with (
        patch("src.rag.rss_ingest.fetch_feed", return_value=articles),
        patch("src.rag.rss_ingest.ingest_news", return_value=5),
    ):
        result = ingest_rss_feeds(["http://fake.url"])

    assert isinstance(result, int)
    assert result > 0


def test_ingest_rss_feeds_uses_all_feeds_when_none():
    total_feeds = sum(len(v) for v in PULSEIQ_RSS_FEEDS.values())
    with (
        patch("src.rag.rss_ingest.fetch_feed", return_value=[]) as mock_fetch,
        patch("src.rag.rss_ingest.ingest_news", return_value=0),
    ):
        ingest_rss_feeds(None)

    assert mock_fetch.call_count == total_feeds


def test_ingest_rss_feeds_deduplication_handled_by_ingest():
    articles = [
        {"url": "http://ex.com/1", "title": "T",
         "description": "", "publishedAt": "", "source": "X"}
    ]
    with (
        patch("src.rag.rss_ingest.fetch_feed", return_value=articles),
        patch("src.rag.rss_ingest.ingest_news", return_value=0),
    ):
        result = ingest_rss_feeds(["http://fake.url"])

    assert result == 0
