"""News article ingestion into ChromaDB for PulseIQ RAG.

Embeds news articles using ``BAAI/bge-large-en-v1.5`` and stores them in a
ChromaDB collection named ``"pulseiq_news"``. Articles are deduplicated by
URL — calling ``ingest_news`` twice with the same articles is idempotent.

BGE uses asymmetric retrieval: documents are embedded as-is, while queries
receive a prefix (see ``embed_query``). Use ``embed_document`` for ingestion
and ``embed_query`` for retrieval — never mix the two.

Typical usage::

    from src.rag.ingest import ingest_news

    articles = news_connector.fetch().to_dict("records")
    new_count = ingest_news(articles)
    print(f"Ingested {new_count} new articles")
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = "pulseiq_news"
"""ChromaDB collection name for all ingested news articles."""

_EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
"""BGE large English model — 1024-dim, 512-token context, strong for domain retrieval."""

_BGE_QUERY_PREFIX: str = (
    "Represent this sentence for searching relevant passages: "
)
"""Asymmetric query prefix required by BGE models.

Applied to search queries only. Documents are stored without any prefix.
See: https://huggingface.co/BAAI/bge-large-en-v1.5
"""

_model: SentenceTransformer | None = None
"""Module-level lazy singleton — loaded once on first call to ``_get_model()``."""


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------


def _get_model() -> SentenceTransformer:
    """Load and return the BGE model, downloading it on first call.

    Uses a module-level singleton so the model is only loaded once per
    process, regardless of how many times ``embed_document`` or
    ``embed_query`` are called.

    Returns:
        Loaded ``SentenceTransformer`` instance for ``BAAI/bge-large-en-v1.5``.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", _EMBED_MODEL)
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Public embedding functions
# ---------------------------------------------------------------------------


def embed_document(texts: list[str]) -> list[list[float]]:
    """Embed document texts for storage in ChromaDB (no prefix).

    BGE documents require no prefix — the model is trained to represent
    raw passage text on the document side of asymmetric retrieval.

    Args:
        texts: List of strings to embed (e.g. ``"{title}. {description}"``).

    Returns:
        List of 1024-dim L2-normalised embedding vectors.
    """
    vectors = _get_model().encode(texts, normalize_embeddings=True)
    if hasattr(vectors, "ndim") and vectors.ndim > 2:
        vectors = vectors.squeeze()
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a retrieval query using the BGE asymmetric prefix.

    BGE projects queries differently from documents so that
    ``cosine(embed_query(q), embed_document(d))`` measures retrieval
    relevance rather than surface similarity.

    Args:
        text: Raw query string. The BGE prefix is prepended internally —
            do not include it in ``text``.

    Returns:
        Single 1024-dim L2-normalised embedding vector.
    """
    vector = _get_model().encode(
        f"{_BGE_QUERY_PREFIX}{text}", normalize_embeddings=True
    )
    if hasattr(vector, "ndim") and vector.ndim > 1:
        vector = vector.squeeze()
    return vector.tolist()


# ---------------------------------------------------------------------------
# ChromaDB embedding function wrapper
# ---------------------------------------------------------------------------


class _DocumentEmbeddingFunction:
    """ChromaDB-compatible callable that wraps ``embed_document``.

    Registered with the ``pulseiq_news`` collection so that
    ``collection.add(documents=...)`` embeds text correctly (no prefix).
    Queries bypass this by passing ``query_embeddings`` directly via
    ``embed_query``, ensuring the asymmetric BGE prefix is applied on the
    query side only.
    """

    def name(self) -> str:
        return "pulseiq-bge-large-en-v1.5"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return embed_document(list(input))


def _embedding_function() -> _DocumentEmbeddingFunction:
    """Return a ChromaDB-compatible document embedding function instance.

    Returns:
        ``_DocumentEmbeddingFunction`` — embeds documents without the BGE
        query prefix. Pass this to ``get_or_create_collection``.
    """
    return _DocumentEmbeddingFunction()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Return (or create) the ``pulseiq_news`` collection.

    Args:
        client: An active ChromaDB client.

    Returns:
        The ``pulseiq_news`` collection, configured with the
        ``BAAI/bge-large-en-v1.5`` document embedding function.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _url_to_id(url: str) -> str:
    """Compute a stable document ID from a URL.

    Args:
        url: Article URL.

    Returns:
        MD5 hex digest of the URL — safe to use as a ChromaDB document ID.
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _parse_published_at_ts(published_at: str | None) -> int:
    """Parse an ISO 8601 timestamp string to a Unix timestamp integer.

    Args:
        published_at: ISO 8601 string (e.g. ``"2024-01-15T12:00:00Z"``).
            ``None`` or unparseable values return 0.

    Returns:
        Unix timestamp as int, or 0 if unparseable.
    """
    if not published_at:
        return 0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, TypeError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_news(
    articles: list[dict[str, Any]],
    chroma_path: str | None = None,
    _client: chromadb.ClientAPI | None = None,
) -> int:
    """Embed and store news articles in ChromaDB, skipping duplicates.

    Each article is deduplicated by URL — an article whose URL is already
    present in the collection is silently skipped. The function is safe to
    call repeatedly with overlapping article lists.

    Args:
        articles: List of article dicts. Each must contain at least
            ``"url"``, ``"title"``, ``"description"``. Optional keys:
            ``"publishedAt"`` (ISO 8601 string), ``"source"`` (str),
            ``"geo_id"`` (str), ``"geo_level"`` (str).
        chroma_path: Path to the ChromaDB persistence directory. Resolved
            in order:

            1. ``chroma_path`` argument
            2. ``CHROMADB_PATH`` environment variable
            3. ``"data/processed/chroma/"`` (default)

        _client: Pre-built ChromaDB client. When provided, ``chroma_path``
            is ignored. Pass a ``PersistentClient`` backed by ``tmp_path``
            in tests to avoid filesystem side-effects.

    Returns:
        Count of articles newly ingested. Returns 0 if ``articles`` is
        empty or all URLs were already present.

    Raises:
        ValueError: If an article dict is missing the required ``"url"`` key.
    """
    if not articles:
        return 0

    # Validate required keys early
    for i, art in enumerate(articles):
        if "url" not in art or not art["url"]:
            raise ValueError(f"Article at index {i} is missing required 'url' key")

    # Build client
    if _client is not None:
        client: chromadb.ClientAPI = _client
    else:
        resolved_path = (
            chroma_path
            or os.getenv("CHROMADB_PATH")
            or "data/processed/chroma/"
        )
        client = chromadb.PersistentClient(path=resolved_path)

    collection = _get_collection(client)

    # Deduplicate within the batch by URL hash before querying ChromaDB.
    # RSS feeds can contain the same article link more than once; ChromaDB's
    # collection.get() raises DuplicateIDError if the ids list has repeats.
    seen_ids: set[str] = set()
    deduped_articles: list[dict[str, Any]] = []
    deduped_ids: list[str] = []
    for art in articles:
        doc_id = _url_to_id(art["url"])
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduped_articles.append(art)
            deduped_ids.append(doc_id)

    # Check which IDs are already stored in ChromaDB
    existing_result = collection.get(ids=deduped_ids)
    already_stored: set[str] = set(existing_result["ids"])

    # Filter to new articles only
    new_articles = [
        art for art, doc_id in zip(deduped_articles, deduped_ids)
        if doc_id not in already_stored
    ]
    new_ids = [
        doc_id for art, doc_id in zip(deduped_articles, deduped_ids)
        if doc_id not in already_stored
    ]

    skipped = len(articles) - len(new_articles)

    if not new_articles:
        logger.info(
            "Ingested 0 new articles (%d skipped as duplicates) — collection=%s",
            skipped, COLLECTION_NAME,
        )
        return 0

    # Build ChromaDB add() arguments
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for art in new_articles:
        title = art.get("title") or ""
        description = art.get("description") or ""
        documents.append(f"{title}. {description}".strip())

        metadatas.append({
            "url": art["url"],
            "title": title,
            "published_at_ts": _parse_published_at_ts(art.get("publishedAt")),
            "source": str(art.get("source") or ""),
            "geo_id": str(art.get("geo_id") or ""),
            "geo_level": str(art.get("geo_level") or ""),
        })

    collection.add(
        ids=new_ids,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(
        "Ingested %d new articles (%d skipped as duplicates) — collection=%s",
        len(new_articles), skipped, COLLECTION_NAME,
    )
    return len(new_articles)
