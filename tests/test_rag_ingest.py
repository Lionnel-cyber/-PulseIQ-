"""Tests for src/rag/ingest.py.

Uses ``chromadb.PersistentClient`` with a ``tmp_path``-based directory so
each test gets an isolated database. This mirrors the DuckDB pattern used
elsewhere in the test suite and avoids shared-state issues with
``EphemeralClient``.

The ``_mock_model`` autouse fixture prevents downloading ``BAAI/bge-large-en-v1.5``
during tests by patching ``_get_model`` with a deterministic fake that returns
unit vectors of the correct dimension (1024).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import chromadb
import pytest

from src.rag.ingest import COLLECTION_NAME, _get_collection, _url_to_id, ingest_news

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_A1 = {
    "title": "Detroit unemployment rises",
    "description": "Jobless claims up 12 percent in Wayne County.",
    "url": "https://example.com/article-1",
    "publishedAt": "2024-01-15T12:00:00Z",
    "source": "AP",
    "geo_id": "Detroit-MI",
    "geo_level": "city",
}

_A2 = {
    "title": "Michigan poverty rate hits decade high",
    "description": "New census data shows rising poverty in metro areas.",
    "url": "https://example.com/article-2",
    "publishedAt": "2024-01-16T08:30:00Z",
    "source": "Reuters",
    "geo_id": "Detroit-MI",
    "geo_level": "city",
}

_A3 = {
    "title": "Chicago credit delinquencies climb",
    "description": "FRED data shows credit card defaults rising in Illinois.",
    "url": "https://example.com/article-3",
    "publishedAt": "2024-01-17T10:00:00Z",
    "source": "Bloomberg",
    "geo_id": "Chicago-IL",
    "geo_level": "city",
}


def _client(tmp_path: Path) -> chromadb.ClientAPI:
    """Return a PersistentClient backed by a fresh tmp_path directory."""
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_model():
    """Prevent downloading BAAI/bge-large-en-v1.5 during tests.

    Patches ``_get_model`` to return a mock ``SentenceTransformer`` that
    produces unit vectors of dimension 1024 — consistent between document
    embedding and query embedding so ChromaDB's HNSW index stays valid.
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
# _url_to_id unit tests (no ChromaDB needed)
# ---------------------------------------------------------------------------


def test_url_to_id_is_deterministic() -> None:
    """Same URL must always produce the same ID."""
    assert _url_to_id("https://example.com/a") == _url_to_id("https://example.com/a")


def test_url_to_id_differs_for_different_urls() -> None:
    """Different URLs must produce different IDs."""
    assert _url_to_id("https://example.com/a") != _url_to_id("https://example.com/b")


def test_url_to_id_is_hex_string() -> None:
    """ID must be a valid MD5 hex digest (32 chars, hex chars only)."""
    doc_id = _url_to_id("https://example.com/test")
    assert len(doc_id) == 32
    assert all(c in "0123456789abcdef" for c in doc_id)


# ---------------------------------------------------------------------------
# ingest_news tests
# ---------------------------------------------------------------------------


def test_ingest_empty_list_returns_zero(tmp_path: Path) -> None:
    """Empty article list must return 0 without touching ChromaDB."""
    count = ingest_news([], _client=_client(tmp_path))
    assert count == 0


def test_ingest_missing_url_raises(tmp_path: Path) -> None:
    """Article without 'url' key must raise ValueError."""
    bad_article = {"title": "No URL", "description": "Missing url field"}
    with pytest.raises(ValueError, match="url"):
        ingest_news([bad_article], _client=_client(tmp_path))


def test_ingest_returns_count(tmp_path: Path) -> None:
    """Ingesting two new articles must return 2."""
    count = ingest_news([_A1, _A2], _client=_client(tmp_path))
    assert count == 2


def test_ingest_deduplication(tmp_path: Path) -> None:
    """Second call with same articles must return 0 — all duplicates."""
    c = _client(tmp_path)
    first = ingest_news([_A1, _A2], _client=c)
    assert first == 2
    second = ingest_news([_A1, _A2], _client=c)
    assert second == 0


def test_ingest_partial_dedup(tmp_path: Path) -> None:
    """Mix of new and duplicate articles must count only the new ones."""
    c = _client(tmp_path)
    ingest_news([_A1, _A2], _client=c)      # seed A1, A2
    count = ingest_news([_A1, _A3], _client=c)  # A1 is dup, A3 is new
    assert count == 1


def test_ingest_metadata_stored(tmp_path: Path) -> None:
    """Stored metadata must include the correct url and geo_id."""
    c = _client(tmp_path)
    ingest_news([_A1], _client=c)

    collection = _get_collection(c)
    doc_id = _url_to_id(_A1["url"])
    result = collection.get(ids=[doc_id], include=["metadatas"])

    assert len(result["ids"]) == 1
    meta = result["metadatas"][0]
    assert meta["url"] == _A1["url"]
    assert meta["geo_id"] == _A1["geo_id"]
    assert meta["source"] == _A1["source"]
    assert isinstance(meta["published_at_ts"], int)
    assert meta["published_at_ts"] > 0
