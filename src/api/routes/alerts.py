"""Alert endpoints for the PulseIQ API.

    POST /alerts                        — register an alert payload in DuckDB
    GET  /alerts/history/{geo_id}       — history of alerts for a geography

All responses use contracts from ``src/contracts.py`` — no inline schemas.

The ``POST /alerts`` endpoint stores the full ``AlertPayload`` to the
``api_alerts`` table (distinct from the observability ``alert_history`` table
written by ``src/observability/alerts.py``).

``GET /alerts/history/{geo_id}`` merges both sources:
- ``api_alerts`` — alerts registered via this API
- ``alert_history`` — alerts fired by the observability MTTD layer
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import duckdb
from fastapi import APIRouter, HTTPException, Request

from src.contracts import AlertPayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])

_CREATE_API_ALERTS_SQL = """
CREATE TABLE IF NOT EXISTS api_alerts (
    alert_id     VARCHAR   NOT NULL PRIMARY KEY,
    geo_id       VARCHAR   NOT NULL,
    payload_json TEXT      NOT NULL,
    created_at   TIMESTAMP NOT NULL
)
"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_api_alerts_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``api_alerts`` table if it does not exist.

    Args:
        conn: Open DuckDB write connection.
    """
    conn.execute(_CREATE_API_ALERTS_SQL)


def _reconstruct_from_history_row(row: dict[str, Any]) -> AlertPayload:
    """Build a best-effort ``AlertPayload`` from a sparse ``alert_history`` row.

    The ``alert_history`` table only records ``alert_id``, ``geo_id``,
    ``alert_type``, ``score``, and ``fired_at``. Missing fields are filled
    with safe defaults so ``AlertPayload`` validation passes.

    Args:
        row: Dict of column values from the ``alert_history`` join query.

    Returns:
        Populated ``AlertPayload`` with defaults for unrecorded fields.
    """
    fired_at_raw = row.get("fired_at")
    if isinstance(fired_at_raw, datetime):
        fired_at = fired_at_raw
    else:
        try:
            fired_at = datetime.fromisoformat(str(fired_at_raw)).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            fired_at = datetime.now(tz=timezone.utc)

    current_score = float(row.get("score") or 0.0)
    geo_id = str(row.get("geo_id") or "")
    geo_name = str(row.get("geo_name") or geo_id)

    missing_sources: list[str] = []
    raw_missing = row.get("missing_sources")
    if raw_missing:
        try:
            parsed = json.loads(str(raw_missing))
            missing_sources = parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            missing_sources = []

    return AlertPayload(
        alert_id=str(row.get("alert_id") or str(uuid.uuid4())),
        region_id=geo_id,
        region_name=geo_name,
        triggered_at=fired_at,
        current_score=current_score,
        previous_score=current_score,
        score_delta=0.0,
        delta_window_days=7,
        alert_type=str(row.get("alert_type") or "threshold_breach"),
        top_drivers=[],
        explanation_summary=(
            f"ESS score of {current_score:.1f} recorded for {geo_name}."
        ),
        confidence=str(row.get("confidence") or "low"),
        missing_sources=missing_sources,
        model_version=str(row.get("model_version") or ""),
        explanation_url="",
        suppressed_until=None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=AlertPayload, status_code=201)
async def create_alert(body: AlertPayload, request: Request) -> AlertPayload:
    """Register an alert payload in DuckDB and return it with a generated alert_id.

    The caller supplies all ``AlertPayload`` fields. Any provided ``alert_id``
    is replaced with a server-generated UUID4 to ensure uniqueness.

    The payload is stored in the ``api_alerts`` table for retrieval via
    ``GET /alerts/history/{geo_id}``.

    Args:
        body: Full ``AlertPayload`` — validated by Pydantic on ingestion.
        request: FastAPI request object (provides ``app.state.db_path``).

    Returns:
        The stored ``AlertPayload`` with the generated ``alert_id``.

    Raises:
        HTTPException: 503 if the DuckDB write fails.
    """
    db_path: str = request.app.state.db_path
    new_id = str(uuid.uuid4())

    # Replace alert_id with server-generated UUID
    payload = AlertPayload(**{**body.model_dump(), "alert_id": new_id})
    geo_id = payload.region_id
    created_at = datetime.now(tz=timezone.utc)
    payload_json = payload.model_dump_json()

    try:
        with duckdb.connect(db_path) as conn:
            _ensure_api_alerts_table(conn)
            conn.execute(
                "INSERT INTO api_alerts (alert_id, geo_id, payload_json, created_at)"
                " VALUES (?, ?, ?, ?)",
                [new_id, geo_id, payload_json, created_at],
            )
    except Exception as exc:
        logger.error("create_alert: DB write failed: %s", exc)
        raise HTTPException(status_code=503, detail="Alert storage failed") from exc

    logger.info("Alert registered: id=%s geo=%s type=%s", new_id, geo_id, payload.alert_type)
    return payload


@router.get("/history/{geo_id}", response_model=list[AlertPayload])
async def get_alert_history(
    geo_id: str,
    request: Request,
) -> list[AlertPayload]:
    """Return all alerts on record for a geography.

    Merges two sources:
    - ``api_alerts`` — alerts submitted via ``POST /alerts``
    - ``alert_history`` — alerts fired by the observability MTTD layer

    Results are deduplicated by ``alert_id`` and ordered by triggered time
    descending (most recent first).

    Args:
        geo_id: Geography identifier.
        request: FastAPI request object (provides ``app.state.db_path``).

    Returns:
        List of ``AlertPayload`` objects for the geography.

    Raises:
        HTTPException: 503 if the database cannot be queried.
    """
    db_path: str = request.app.state.db_path
    results: dict[str, AlertPayload] = {}

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            # --- Source 1: api_alerts (full payload stored as JSON) ---
            try:
                df_api = conn.execute(
                    "SELECT payload_json FROM api_alerts"
                    " WHERE geo_id = ? ORDER BY created_at DESC",
                    [geo_id],
                ).fetchdf()
                for row in df_api.to_dict(orient="records"):
                    try:
                        p = AlertPayload.model_validate_json(str(row["payload_json"]))
                        results[p.alert_id] = p
                    except Exception as exc:
                        logger.warning("Skipping malformed api_alerts row: %s", exc)
            except Exception as exc:
                logger.debug("api_alerts table not yet created or empty: %s", exc)

            # --- Source 2: alert_history (sparse, needs reconstruction) ---
            try:
                df_hist = conn.execute(
                    """
                    SELECT ah.alert_id, ah.geo_id, ah.alert_type, ah.score, ah.fired_at,
                           es.geo_name, es.confidence, es.model_version, es.missing_sources
                    FROM alert_history ah
                    LEFT JOIN ess_scores es
                      ON es.geo_id = ah.geo_id
                      AND es.run_date = (
                          SELECT MAX(run_date) FROM ess_scores WHERE geo_id = ah.geo_id
                      )
                    WHERE ah.geo_id = ?
                    ORDER BY ah.fired_at DESC
                    """,
                    [geo_id],
                ).fetchdf()
                for row in df_hist.to_dict(orient="records"):
                    alert_id = str(row.get("alert_id") or "")
                    if alert_id and alert_id not in results:
                        try:
                            results[alert_id] = _reconstruct_from_history_row(row)
                        except Exception as exc:
                            logger.warning("Skipping malformed alert_history row: %s", exc)
            except Exception as exc:
                logger.debug("alert_history query failed: %s", exc)
    except Exception as exc:
        logger.error("get_alert_history: DB error for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(status_code=503, detail="Alert history unavailable") from exc

    # Sort by triggered_at descending
    return sorted(
        results.values(),
        key=lambda p: p.triggered_at,
        reverse=True,
    )
