"""Tests for src/rag/retriever.py.

Seeds a ``PersistentClient`` (tmp_path-scoped) collection with known documents
and timestamps, then verifies ``NewsRetriever`` returns correctly filtered and
mapped ``RetrievedSource`` objects.

Each test gets its own isolated ChromaDB directory via ``tmp_path``.

The ``_mock_model`` autouse fixture prevents downloading ``BAAI/bge-large-en-v1.5``
during tests by returning a deterministic mock with 1024-dim unit vectors.
This covers both ``embed_document`` (called during collection seeding) and
``embed_query`` (called inside ``get_relevant_docs``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import chromadb
import pytest

from src.contracts import RetrievedSource
from src.rag.ingest import COLLECTION_NAME, _embedding_function, _url_to_id
from src.rag.retriever import NewsRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_model():
    """Prevent downloading BAAI/bge-large-en-v1.5 during tests.

    Returns 1024-dim unit vectors consistently for both document and query
    embeddings so ChromaDB's HNSW index remains valid across add/query calls.
    """
    mock = MagicMock()

    def _encode(texts, normalize_embeddings=True):
        if isinstance(texts, str):
            vec = np.ones(1024, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            return vec  # 1-D array, matching real SentenceTransformer behaviour
        n = len(texts)
        vecs = np.ones((n, 1024), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    mock.encode.side_effect = _encode
    with patch("src.rag.ingest._get_model", return_value=mock):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_collection(
    tmp_path: Path,
    *,
    published_at_ts: int,
    n_docs: int = 3,
) -> chromadb.ClientAPI:
    """Return a PersistentClient with ``n_docs`` seeded documents.

    Args:
        tmp_path: Test-scoped temp directory for the ChromaDB files.
        published_at_ts: Unix timestamp to assign to all documents.
        n_docs: Number of documents to insert.

    Returns:
        Populated PersistentClient.
    """
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ids = [_url_to_id(f"https://example.com/doc-{i}") for i in range(n_docs)]
    documents = [
        f"Economic stress report {i}: unemployment and poverty in Detroit Michigan."
        for i in range(n_docs)
    ]
    metadatas = [
        {
            "url": f"https://example.com/doc-{i}",
            "title": f"Economic Report {i}",
            "published_at_ts": published_at_ts,
            "source": "AP",
            "geo_id": "Detroit-MI",
            "geo_level": "city",
        }
        for i in range(n_docs)
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return client


def _recent_ts() -> int:
    """Unix timestamp for yesterday."""
    return int((datetime.now(tz=timezone.utc) - timedelta(days=1)).timestamp())


def _old_ts() -> int:
    """Unix timestamp for 60 days ago (outside any 7-day window)."""
    return int((datetime.now(tz=timezone.utc) - timedelta(days=60)).timestamp())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_relevant_docs_returns_list_of_retrieved_sources(
    tmp_path: Path,
) -> None:
    """Results must be a list of ``RetrievedSource`` instances."""
    client = _seed_collection(tmp_path, published_at_ts=_recent_ts())
    retriever = NewsRetriever(_client=client)

    docs = retriever.get_relevant_docs("Detroit-MI", "Detroit")

    assert isinstance(docs, list)
    for doc in docs:
        assert isinstance(doc, RetrievedSource)


def test_date_filter_excludes_old_docs(tmp_path: Path) -> None:
    """Articles published 60 days ago must not appear in a 7-day window."""
    client = _seed_collection(tmp_path, published_at_ts=_old_ts())
    retriever = NewsRetriever(_client=client)

    docs = retriever.get_relevant_docs("Detroit-MI", "Detroit", days_back=7)

    assert docs == [], (
        f"Expected empty list but got {len(docs)} docs "
        "— old articles slipped through date filter"
    )


def test_relevance_score_bounded(tmp_path: Path) -> None:
    """All ``relevance_score`` values must be in [0.0, 1.0]."""
    client = _seed_collection(tmp_path, published_at_ts=_recent_ts())
    retriever = NewsRetriever(_client=client)

    docs = retriever.get_relevant_docs("Detroit-MI", "Detroit")

    assert docs, "Expected at least one doc"
    for doc in docs:
        assert 0.0 <= doc.relevance_score <= 1.0, (
            f"relevance_score={doc.relevance_score} is out of [0, 1]"
        )


def test_empty_collection_returns_empty_list(tmp_path: Path) -> None:
    """An empty collection must return an empty list without raising."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    # Don't seed — collection is created empty by NewsRetriever constructor
    retriever = NewsRetriever(_client=client)

    docs = retriever.get_relevant_docs("Detroit-MI", "Detroit")

    assert docs == []


def test_recent_docs_appear_in_results(tmp_path: Path) -> None:
    """Recent articles must be retrievable within a matching days_back window."""
    client = _seed_collection(tmp_path, published_at_ts=_recent_ts(), n_docs=2)
    retriever = NewsRetriever(_client=client)

    docs = retriever.get_relevant_docs("Detroit-MI", "Detroit", days_back=7)

    assert len(docs) > 0


def test_published_at_is_datetime_or_none(tmp_path: Path) -> None:
    """``published_at`` on each result must be a ``datetime`` or ``None``."""
    client = _seed_collection(tmp_path, published_at_ts=_recent_ts(), n_docs=2)
    retriever = NewsRetriever(_client=client)

    docs = retriever.get_relevant_docs("Detroit-MI", "Detroit")

    for doc in docs:
        assert doc.published_at is None or isinstance(doc.published_at, datetime)
