"""
RSS feed ingestion for PulseIQ's RAG layer.

Fetches articles from a curated set of economic, labour-market, regional,
housing/debt, and food/poverty RSS feeds and ingests them into ChromaDB via
the existing :func:`~src.rag.ingest.ingest_news` pipeline.

Usage::

    from src.rag.rss_ingest import ingest_rss_feeds
    new_articles = ingest_rss_feeds()          # all configured feeds
    new_articles = ingest_rss_feeds([url, …])  # explicit list
"""

import logging
from typing import Any

import feedparser

from src.rag.ingest import ingest_news

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feed catalogue
# ---------------------------------------------------------------------------

PULSEIQ_RSS_FEEDS: dict[str, list[str]] = {
    "economic_national": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/companyNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
        "https://feeds.feedburner.com/businessinsider",
    ],
    "labour_market": [
        "https://www.bls.gov/feed/rss.xml",
        "https://www.dol.gov/rss/releases.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Jobs.xml",
    ],
    "regional": [
        "https://www.detroitnews.com/rss/",
        "https://www.chicagotribune.com/arcio/rss/",
        "https://www.houstonchronicle.com/rss/",
        "https://www.latimes.com/business/rss2.0.xml",
    ],
    "housing_debt": [
        "https://www.housingwire.com/feed/",
        "https://www.calculatedriskblog.com/feeds/posts/default",
    ],
    "food_poverty": [
        "https://feeds.reuters.com/reuters/domesticNews",
    ],
}

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def fetch_feed(url: str) -> list[dict[str, Any]]:
    """Parse a single RSS feed and return a list of article dicts.

    Uses feedparser to download and parse the feed at *url*. Each entry is
    mapped to the article dict shape expected by
    :func:`~src.rag.ingest.ingest_news`.  Entries whose ``url`` or ``title``
    field is empty are silently skipped.

    Any exception raised by feedparser is caught and logged at ERROR level;
    an empty list is returned so that one broken feed never interrupts the
    rest of the ingestion run.

    Args:
        url: The RSS/Atom feed URL to fetch and parse.

    Returns:
        A list of article dicts with keys ``url``, ``title``,
        ``description``, ``publishedAt``, and ``source``.
        Returns ``[]`` on parse error or if every entry is filtered out.
    """
    try:
        parsed = feedparser.parse(url)
        feed_title: str = parsed.feed.get("title", url)
        articles: list[dict[str, Any]] = []
        for entry in parsed.entries:
            article_url: str = entry.get("link", "")
            title: str = entry.get("title", "")
            if not article_url or not title:
                continue
            articles.append(
                {
                    "url": article_url,
                    "title": title,
                    "description": entry.get("summary", ""),
                    "publishedAt": entry.get("published", ""),
                    "source": feed_title,
                }
            )
        return articles
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse feed url=%s error=%s", url, exc)
        return []


def ingest_rss_feeds(feed_urls: list[str] | None = None) -> int:
    """Fetch all configured RSS feeds and ingest new articles into ChromaDB.

    If *feed_urls* is ``None``, every URL in :data:`PULSEIQ_RSS_FEEDS` is
    processed (all categories, flattened).  Otherwise only the supplied URLs
    are fetched.

    For each feed URL:

    1. :func:`fetch_feed` retrieves and parses entries.
    2. The article list is passed to :func:`~src.rag.ingest.ingest_news`,
       which deduplicates by URL and writes new documents to ChromaDB.
    3. Per-feed metrics (fetched count, newly-ingested count) are logged at
       INFO level.

    A summary line is logged after all feeds complete.

    Args:
        feed_urls: Optional list of RSS feed URLs to process.  Defaults to
            all URLs in :data:`PULSEIQ_RSS_FEEDS`.

    Returns:
        Total count of newly ingested articles across all processed feeds.
    """
    if feed_urls is None:
        feed_urls = [
            url
            for urls in PULSEIQ_RSS_FEEDS.values()
            for url in urls
        ]

    total_new: int = 0
    for url in feed_urls:
        articles = fetch_feed(url)
        fetched_count: int = len(articles)
        new_count: int = ingest_news(articles) if articles else 0
        total_new += new_count
        logger.info(
            "RSS feed processed url=%s fetched=%d new=%d",
            url,
            fetched_count,
            new_count,
        )

    logger.info(
        "RSS ingest complete feeds_processed=%d total_new_articles=%d",
        len(feed_urls),
        total_new,
    )
    return total_new
