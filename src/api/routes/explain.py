"""Explanation endpoints for the PulseIQ API.

Answers the question "Why is it that score?" across two endpoints:

    GET /explain/{geo_id}           — full four-section structured explanation
    GET /explain/{geo_id}/evidence  — raw retrieved news with relevance scores

All responses use contracts from ``src/contracts.py`` — no inline schemas.
``StressExplainer`` generates explanations via SHAP + RAG + LLM.
``NewsRetriever`` returns evidence without LLM involvement.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.deps import get_db, get_explainer
from src.contracts import Explanation, Prediction
from src.rag.explainer import StressExplainer
from src.rag.retriever import NewsRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_json_list(raw: Any) -> list[str]:
    """Parse a JSON-string list field from ``ess_scores``.

    Args:
        raw: Raw column value — JSON string, list, or ``None``.

    Returns:
        Parsed list of strings, or ``[]`` on any parse error.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        result = json.loads(str(raw))
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _parse_json_dict(raw: Any) -> dict[str, float]:
    """Parse a JSON-string dict field from ``ess_scores``.

    Args:
        raw: Raw column value — JSON string, dict, or ``None``.

    Returns:
        Parsed dict of float values, or ``{}`` on any parse error.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {k: float(v) for k, v in raw.items()}
    try:
        result = json.loads(str(raw))
        return {k: float(v) for k, v in result.items()} if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def _format_sse_event(data: str) -> str:
    """Format a payload as a standards-compliant SSE event.

    SSE requires every payload line to be prefixed with ``data:``.
    This keeps multi-line section blocks intact for compliant clients.
    """
    lines = data.splitlines() or [""]
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def _row_to_prediction(row: dict[str, Any]) -> Prediction:
    """Convert a DuckDB result row to a ``Prediction`` contract.

    Args:
        row: Dict mapping column names to values from the ``ess_scores`` table.

    Returns:
        Validated ``Prediction`` instance.
    """
    return Prediction(
        geo_id=str(row["geo_id"]),
        geo_name=str(row["geo_name"]),
        geo_level=row["geo_level"],
        run_date=row["run_date"],
        ess_score=float(row["ess_score"]),
        score_band=row["score_band"],
        delta_7d=float(row["delta_7d"]) if row.get("delta_7d") is not None else None,
        delta_30d=float(row["delta_30d"]) if row.get("delta_30d") is not None else None,
        confidence=row["confidence"],
        early_warning=bool(row.get("early_warning", False)),
        missing_sources=_parse_json_list(row.get("missing_sources")),
        stale_sources=_parse_json_list(row.get("stale_sources")),
        anomaly_flags=_parse_json_list(row.get("anomaly_flags")),
        granularity_warning=bool(row.get("granularity_warning", False)),
        model_version=str(row.get("model_version") or ""),
        feature_version=str(row.get("feature_version") or ""),
        calibrated=bool(row.get("calibrated", False)),
        tier1_score=float(row.get("tier1_score") or 0.0),
        tier2_score=float(row.get("tier2_score") or 0.0),
        tier3_score=float(row.get("tier3_score") or 0.0),
        shap_values=_parse_json_dict(row.get("shap_values")),
    )


def _fetch_latest_prediction(
    conn: duckdb.DuckDBPyConnection,
    geo_id: str,
) -> Prediction | None:
    """Load the latest scored Prediction for a geography from ``ess_scores``.

    Args:
        conn: Open DuckDB read connection.
        geo_id: Geography identifier.

    Returns:
        ``Prediction`` for the most recent run, or ``None`` if not found.
    """
    df = conn.execute(
        """
        SELECT *
        FROM ess_scores
        WHERE geo_id = ?
          AND run_date = (SELECT MAX(run_date) FROM ess_scores WHERE geo_id = ?)
        """,
        [geo_id, geo_id],
    ).fetchdf()

    if df.empty:
        return None

    return _row_to_prediction(df.iloc[0].to_dict())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{geo_id}", response_model=Explanation)
async def get_explanation(
    geo_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    explainer: StressExplainer = Depends(get_explainer),
) -> Explanation:
    """Generate a structured four-section explanation for a geography's ESS score.

    The explanation contains:
    1. SUMMARY — one factual sentence describing the score movement
    2. TOP_DRIVERS — up to 3 SHAP contributors in plain English
    3. EVIDENCE — retrieved news articles with URLs
    4. CAVEATS — source gaps, staleness notes, confidence caveats (never omitted)

    When the configured LLM provider is unavailable or unconfigured, the
    explainer returns a structured fallback explanation instead of failing.

    Args:
        geo_id: Geography identifier (e.g. ``"Detroit-MI"``).
        db: Injected DuckDB read connection.
        explainer: Injected ``StressExplainer`` singleton.

    Returns:
        Validated ``Explanation`` with all four sections populated.

    Raises:
        HTTPException: 404 if the geography is not found in ``ess_scores``.
    """
    try:
        prediction = _fetch_latest_prediction(db, geo_id)
    except Exception as exc:
        logger.error("get_explanation: DB error for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(status_code=503, detail="Score data unavailable") from exc

    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Geography '{geo_id}' not found")

    try:
        return await asyncio.to_thread(explainer.explain, prediction)
    except Exception as exc:
        logger.error("get_explanation: StressExplainer failed for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Explanation generation failed — LLM service error",
        ) from exc


@router.get("/{geo_id}/stream")
async def explain_stream(
    geo_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    explainer: StressExplainer = Depends(get_explainer),
) -> StreamingResponse:
    """Stream raw explanation text for progressive UI rendering."""
    try:
        prediction = _fetch_latest_prediction(db, geo_id)
    except Exception as exc:
        logger.error("explain_stream: DB error for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(status_code=503, detail="Score data unavailable") from exc

    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Geography '{geo_id}' not found")

    async def generate() -> Any:
        try:
            async for chunk in explainer.explain_stream(prediction):
                yield _format_sse_event(chunk)
        except Exception as exc:
            logger.error("explain_stream: streaming failed for geo_id=%s: %s", geo_id, exc)
            yield _format_sse_event(f"[Error: {str(exc)}]")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{geo_id}/evidence", response_model=list[dict])
async def get_evidence(
    geo_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return raw retrieved news documents for a geography without LLM processing.

    Queries ChromaDB via ``NewsRetriever`` for documents relevant to the
    geography. Returns each document's metadata with its relevance score.
    This endpoint does not require ``OPENAI_API_KEY``.

    Args:
        geo_id: Geography identifier.
        db: Injected DuckDB read connection.

    Returns:
        List of dicts with keys: ``url``, ``title``, ``published_at``,
        ``relevance_score``.

    Raises:
        HTTPException: 404 if the geography is not found in ``ess_scores``.
        HTTPException: 503 if ChromaDB retrieval fails.
    """
    # Resolve geo_name for the retriever query
    try:
        df = db.execute(
            """
            SELECT geo_name
            FROM ess_scores
            WHERE geo_id = ?
            LIMIT 1
            """,
            [geo_id],
        ).fetchdf()
    except Exception as exc:
        logger.error("get_evidence: DB error for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(status_code=503, detail="Score data unavailable") from exc

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Geography '{geo_id}' not found")

    geo_name = str(df.iloc[0]["geo_name"])

    try:
        retriever = NewsRetriever()
        docs = retriever.get_relevant_docs(geo_id=geo_id, geo_name=geo_name)
    except Exception as exc:
        logger.error("get_evidence: NewsRetriever failed for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Evidence retrieval failed — ChromaDB service error",
        ) from exc

    return [doc.model_dump(mode="json") for doc in docs]
