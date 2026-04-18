"""Semantic news retrieval for PulseIQ RAG explanations.

Wraps ChromaDB's vector search to surface the most relevant recent news
articles for a given geography. The returned ``RetrievedSource`` objects
plug directly into the ``Explanation`` contract.

Queries use ``embed_query`` (BGE asymmetric prefix) while ingested documents
use ``embed_document`` (no prefix) - this asymmetry is what makes BGE
retrieval work correctly.

Typical usage::

    from src.rag.retriever import NewsRetriever

    retriever = NewsRetriever()
    docs = retriever.get_relevant_docs(geo_id="Detroit-MI", geo_name="Detroit")
    for doc in docs:
        print(doc.title, doc.url)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import chromadb

from src.contracts import RetrievedSource
from src.rag.ingest import COLLECTION_NAME, _embedding_function, embed_query

logger = logging.getLogger(__name__)


def _is_placeholder_url(url: str) -> bool:
    """Return True when a URL is a placeholder rather than a real source."""
    host = urlparse(url).netloc.lower()
    return not host or "example.com" in host


def _metadata_matches_geo(meta: dict[str, Any], geo_id: str) -> bool:
    """Prefer documents explicitly tagged to the requested geography."""
    meta_geo_id = str(meta.get("geo_id") or "").strip()
    return bool(meta_geo_id) and meta_geo_id == geo_id


class NewsRetriever:
    """Semantic search over ingested news articles in ChromaDB.

    Queries the ``pulseiq_news`` collection for articles semantically
    similar to a geography + economic-stress query string, optionally
    filtered to a recent time window.

    Queries are embedded with the BGE asymmetric prefix via ``embed_query``
    and passed as ``query_embeddings`` - this bypasses the collection's
    document embedding function and ensures the correct asymmetric encoding.
    """

    def __init__(
        self,
        chroma_path: str | None = None,
        _client: chromadb.ClientAPI | None = None,
    ) -> None:
        if _client is not None:
            self._client: chromadb.ClientAPI = _client
        else:
            resolved_path = (
                chroma_path
                or os.getenv("CHROMADB_PATH")
                or "data/processed/chroma/"
            )
            self._client = chromadb.PersistentClient(path=resolved_path)

        self._collection: chromadb.Collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    def get_relevant_docs(
        self,
        geo_id: str,
        geo_name: str,
        days_back: int = 7,
        top_k: int = 5,
    ) -> list[RetrievedSource]:
        """Retrieve the most semantically relevant recent news articles.

        The retriever now fetches a wider candidate set than ``top_k`` so it can
        discard placeholder URLs such as ``example.com`` before returning the
        final results. If a narrow time window produces no real articles, it
        falls back to a 30-day window.
        """
        try:
            if self._collection.count() == 0:
                logger.debug(
                    "NewsRetriever: collection is empty; skipping query for geo_id=%s",
                    geo_id,
                )
                return []
        except Exception as exc:
            logger.debug(
                "NewsRetriever: could not inspect collection size for geo_id=%s: %s",
                geo_id,
                exc,
            )

        query = f"economic stress {geo_name} unemployment poverty jobless claims"
        candidate_count = max(top_k * 5, 10)
        search_windows = [days_back]
        if days_back < 30:
            search_windows.append(30)

        for window_days in search_windows:
            cutoff_ts = int(
                (datetime.now(tz=timezone.utc) - timedelta(days=window_days)).timestamp()
            )

            try:
                results = self._collection.query(
                    query_embeddings=[embed_query(query)],
                    n_results=candidate_count,
                    where={"published_at_ts": {"$gte": cutoff_ts}},
                    include=["metadatas", "distances"],
                )
            except ValueError as exc:
                logger.error("NewsRetriever: invalid query embeddings - %s", exc)
                raise
            except Exception as exc:
                logger.debug(
                    "NewsRetriever: ChromaDB query returned no results: %s", exc
                )
                continue

            metadatas: list[dict[str, Any]] = results["metadatas"][0]
            distances: list[float] = results["distances"][0]
            if not metadatas:
                continue

            filtered = [
                (meta, dist)
                for meta, dist in zip(metadatas, distances)
                if int(meta.get("published_at_ts") or 0) >= cutoff_ts
                and not _is_placeholder_url(str(meta.get("url") or ""))
            ]
            if not filtered:
                continue

            filtered.sort(
                key=lambda item: (
                    0 if _metadata_matches_geo(item[0], geo_id) else 1,
                    float(item[1]),
                )
            )

            sources: list[RetrievedSource] = []
            for meta, dist in filtered[:top_k]:
                relevance = max(0.0, min(1.0, 1.0 - float(dist)))

                published_at: datetime | None = None
                ts = meta.get("published_at_ts")
                if ts:
                    try:
                        published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    except (ValueError, OSError):
                        published_at = None

                sources.append(
                    RetrievedSource(
                        url=str(meta.get("url", "")),
                        title=str(meta.get("title", "")),
                        published_at=published_at,
                        relevance_score=relevance,
                    )
                )

            logger.debug(
                "NewsRetriever: returned %d docs for geo_id=%s days_back=%d",
                len(sources), geo_id, window_days,
            )
            return sources

        logger.debug(
            "NewsRetriever: returned 0 docs for geo_id=%s days_back=%d",
            geo_id, days_back,
        )
        return []
