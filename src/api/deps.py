"""Shared FastAPI dependencies for PulseIQ API routes.

Provides reusable ``Depends()`` callables so all routes get the same
DuckDB connection factory and ``StressExplainer`` singleton.

Usage::

    from src.api.deps import get_db, get_explainer

    @router.get("/{geo_id}")
    async def endpoint(
        geo_id: str,
        db: duckdb.DuckDBPyConnection = Depends(get_db),
        explainer: StressExplainer = Depends(get_explainer),
    ) -> ...:
"""

from __future__ import annotations

import logging
from typing import Generator

import duckdb
from fastapi import Request

from src.rag.explainer import StressExplainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DuckDB connection dependency
# ---------------------------------------------------------------------------


def get_db(request: Request) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield a read-only DuckDB connection and close it when the request ends.

    The database path is read from ``request.app.state.db_path``, which is
    set by the lifespan startup handler in ``main.py``.

    Args:
        request: FastAPI ``Request`` object providing access to ``app.state``.

    Yields:
        An open, read-only ``DuckDBPyConnection``.
    """
    db_path: str = request.app.state.db_path
    conn = duckdb.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# StressExplainer singleton dependency
# ---------------------------------------------------------------------------

_explainer: StressExplainer | None = None


def get_explainer() -> StressExplainer:
    """Return the module-level ``StressExplainer`` singleton.

    Initialised lazily on first call and reused for the lifetime of the
    process. ``StressExplainer`` holds a ``ChatOpenAI`` client and a
    ``NewsRetriever`` backed by ChromaDB — both are expensive to create.

    Returns:
        The shared ``StressExplainer`` instance.
    """
    global _explainer
    if _explainer is None:
        logger.info("Initialising StressExplainer singleton")
        _explainer = StressExplainer()
    return _explainer
