"""Score endpoints for the PulseIQ API.

Answers the question "What is the score?" across four endpoints:

    GET /scores/top                     — highest-stress geographies today
    GET /scores/{geo_id}                — latest score for a single geography
    GET /scores/{geo_id}/history        — score history over a rolling window
    GET /scores/{geo_id}/drivers        — top SHAP contributors + tier breakdown

All responses use contracts from ``src/contracts.py`` — no inline schemas.
All data is read from the ``ess_scores`` DuckDB table written by
``src/models/predict.py``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from datetime import date, timedelta
from typing import Any, Literal

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_db
from src.contracts import (
    ConfidenceLevel,
    MapScoreResponse,
    ScoreResponse,
    TimeSeriesPoint,
    TimeSeriesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scores", tags=["scores"])

# ---------------------------------------------------------------------------
# Confidence filter mapping
# ---------------------------------------------------------------------------

_CONFIDENCE_LEVELS: dict[ConfidenceLevel, list[str]] = {
    "low": ["low", "medium", "high"],
    "medium": ["medium", "high"],
    "high": ["high"],
}

_CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

_FIPS_TO_STATE: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}

_STATE_CODE_TO_NAME: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

_STATE_NAME_TO_CODE: dict[str, str] = {
    name.upper(): code for code, name in _STATE_CODE_TO_NAME.items()
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_json_list(raw: Any) -> list[str]:
    """Parse a stored JSON-string list field from ``ess_scores``.

    ``predict.py`` serialises list fields via ``json.dumps()``. This helper
    handles both JSON strings and already-parsed lists, and falls back to an
    empty list rather than raising.

    Args:
        raw: Raw value from DuckDB — typically a JSON string or ``None``.

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
    """Parse a stored JSON-string dict field (e.g. ``shap_values``) from ``ess_scores``.

    Args:
        raw: Raw value from DuckDB — typically a JSON string or ``None``.

    Returns:
        Parsed dict mapping string keys to float values, or ``{}`` on error.
    """
    def _sanitize_mapping(value: dict[Any, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, item in value.items():
            parsed = _safe_optional_float(item)
            if parsed is not None:
                result[str(key)] = parsed
        return result

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return _sanitize_mapping(raw)
    try:
        result = json.loads(str(raw))
        return _sanitize_mapping(result) if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def _safe_optional_float(raw: Any) -> float | None:
    """Return ``None`` for missing / non-finite numeric values.

    DuckDB rows often pass through pandas before reaching these route helpers.
    That means SQL ``NULL`` values can arrive here as ``float('nan')`` rather
    than ``None``. FastAPI's JSON renderer rejects ``NaN`` and ``Infinity``,
    so API responses must normalize them back to ``None`` first.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_float(raw: Any, default: float = 0.0) -> float:
    """Return a finite float or ``default`` when the value is missing/invalid."""
    value = _safe_optional_float(raw)
    return default if value is None else value


def _row_to_score(row: dict[str, Any]) -> ScoreResponse:
    """Convert a DuckDB result row to a ``ScoreResponse``.

    Args:
        row: Dict mapping column names to values from the ``ess_scores`` table.

    Returns:
        Validated ``ScoreResponse`` contract instance.
    """
    return ScoreResponse(
        geo_id=str(row["geo_id"]),
        geo_name=str(row["geo_name"]),
        geo_level=row["geo_level"],
        run_date=row["run_date"],
        ess_score=_safe_float(row["ess_score"]),
        score_band=row["score_band"],
        delta_7d=_safe_optional_float(row.get("delta_7d")),
        delta_30d=_safe_optional_float(row.get("delta_30d")),
        confidence=row["confidence"],
        early_warning=bool(row["early_warning"]),
        missing_sources=_parse_json_list(row.get("missing_sources")),
        stale_sources=_parse_json_list(row.get("stale_sources")),
        anomaly_flags=_parse_json_list(row.get("anomaly_flags")),
        granularity_warning=bool(row.get("granularity_warning", False)),
        model_version=str(row.get("model_version") or ""),
        feature_version=str(row.get("feature_version") or ""),
        calibrated=bool(row.get("calibrated", False)),
        tier1_score=_safe_float(row.get("tier1_score")),
        tier2_score=_safe_float(row.get("tier2_score")),
        tier3_score=_safe_float(row.get("tier3_score")),
        shap_values=_parse_json_dict(row.get("shap_values")),
    )


def _coerce_run_date(value: Any) -> date:
    """Coerce a DuckDB/pandas date-like value to ``datetime.date``."""
    if isinstance(value, date) and not hasattr(value, "date"):
        return value
    if hasattr(value, "date"):
        return value.date()
    return date.fromisoformat(str(value))


def _resolve_snapshot_date(
    conn: duckdb.DuckDBPyConnection,
    requested_date: date | None,
) -> date | None:
    """Resolve the most recent score snapshot on or before ``requested_date``."""
    if requested_date is None:
        row = conn.execute("SELECT MAX(run_date) FROM ess_scores").fetchone()
    else:
        row = conn.execute(
            "SELECT MAX(run_date) FROM ess_scores WHERE run_date <= ?",
            [requested_date],
        ).fetchone()

    if not row or row[0] is None:
        return None
    return _coerce_run_date(row[0])


def _normalize_state_code(raw: str | None) -> str | None:
    """Return a USPS state code when ``raw`` looks like a state identifier."""
    if not raw:
        return None

    candidate = raw.strip().upper()
    if candidate in _STATE_CODE_TO_NAME:
        return candidate

    if candidate.isdigit():
        return _FIPS_TO_STATE.get(candidate.zfill(2))

    return _STATE_NAME_TO_CODE.get(candidate)


def _extract_state_code(row: dict[str, Any]) -> str | None:
    """Infer a USPS state code from a score row."""
    geo_id = str(row.get("geo_id") or "")
    geo_name = str(row.get("geo_name") or "")
    geo_level = str(row.get("geo_level") or "")

    if geo_level == "state":
        code = _normalize_state_code(geo_id) or _normalize_state_code(geo_name)
        if code:
            return code

    match = re.search(r"[-, ]([A-Z]{2})$", geo_id.upper())
    if match:
        code = _normalize_state_code(match.group(1))
        if code:
            return code

    match = re.search(r",\s*([A-Z]{2})$", geo_name.upper())
    if match:
        code = _normalize_state_code(match.group(1))
        if code:
            return code

    words = geo_name.upper().split()
    if words:
        return _normalize_state_code(" ".join(words))
    return None


def _fetch_snapshot_df(
    conn: duckdb.DuckDBPyConnection,
    snapshot_date: date,
    geo_level: str | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> Any:
    """Return a score snapshot DataFrame for one run date."""
    where = ["run_date = ?"]
    params: list[Any] = [snapshot_date]

    if geo_level:
        where.append("geo_level = ?")
        params.append(geo_level)

    if query:
        where.append("(LOWER(geo_id) LIKE ? OR LOWER(geo_name) LIKE ?)")
        like = f"%{query.lower()}%"
        params.extend([like, like])

    sql = (
        "SELECT * FROM ess_scores WHERE "
        + " AND ".join(where)
        + " ORDER BY ess_score DESC, geo_name ASC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return conn.execute(sql, params).fetchdf()


def _row_to_map_score(row: dict[str, Any], state_code: str) -> MapScoreResponse:
    """Convert one score row to a state-level map payload."""
    return MapScoreResponse(
        state_code=state_code,
        geo_id=str(row["geo_id"]),
        geo_name=str(row["geo_name"]),
        geo_level=row["geo_level"],
        run_date=_coerce_run_date(row["run_date"]),
        ess_score=_safe_float(row["ess_score"]),
        delta_7d=_safe_optional_float(row.get("delta_7d")),
        confidence=row["confidence"],
        missing_sources=_parse_json_list(row.get("missing_sources")),
        granularity_warning=bool(row.get("granularity_warning", False)),
        drilldown_geo_id=str(row["geo_id"]),
    )


def _aggregate_map_rows(
    state_code: str,
    rows: list[dict[str, Any]],
    snapshot_date: date,
) -> MapScoreResponse:
    """Aggregate multiple sub-state score rows into one state map payload."""
    ess_scores = [_safe_float(row["ess_score"]) for row in rows]
    deltas = [
        delta
        for row in rows
        if (delta := _safe_optional_float(row.get("delta_7d"))) is not None
    ]
    confidences: list[ConfidenceLevel] = [row["confidence"] for row in rows]
    missing_sources = sorted({
        item
        for row in rows
        for item in _parse_json_list(row.get("missing_sources"))
    })

    worst_confidence = min(confidences, key=lambda value: _CONFIDENCE_ORDER[value])
    representative_row = max(rows, key=lambda row: _safe_float(row["ess_score"]))

    return MapScoreResponse(
        state_code=state_code,
        geo_id=state_code,
        geo_name=_STATE_CODE_TO_NAME.get(state_code, state_code),
        geo_level="state",
        run_date=snapshot_date,
        ess_score=round(statistics.mean(ess_scores), 2),
        delta_7d=round(statistics.mean(deltas), 2) if deltas else None,
        confidence=worst_confidence,
        missing_sources=missing_sources,
        granularity_warning=any(bool(row.get("granularity_warning", False)) for row in rows),
        drilldown_geo_id=str(representative_row["geo_id"]),
    )


def _compute_trend(scores: list[float]) -> Literal["improving", "stable", "deteriorating", "volatile"]:
    """Compute the directional trend of an ESS score series.

    Algorithm:
    1. If coefficient of variation (stddev/mean) > 0.15: volatile
    2. Compare the mean of the first third vs the mean of the last third.
    3. If the delta < −3.0: improving (stress declining)
    4. If the delta > +3.0: deteriorating (stress rising)
    5. Otherwise: stable

    Args:
        scores: List of ESS scores in ascending date order.

    Returns:
        One of ``"improving"``, ``"stable"``, ``"deteriorating"``, ``"volatile"``.
    """
    if len(scores) < 3:
        return "stable"

    third = max(len(scores) // 3, 1)
    first_mean = statistics.mean(scores[:third])
    last_mean = statistics.mean(scores[-third:])
    delta = last_mean - first_mean

    mean_score = statistics.mean(scores)
    stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    cv = stddev / mean_score if mean_score > 0 else 0.0

    if cv > 0.15:
        return "volatile"
    if delta < -3.0:
        return "improving"
    if delta > 3.0:
        return "deteriorating"
    return "stable"


def _fetch_latest_row(
    conn: duckdb.DuckDBPyConnection,
    geo_id: str,
) -> dict[str, Any] | None:
    """Query the latest ``ess_scores`` row for a geography.

    Args:
        conn: Open DuckDB connection.
        geo_id: Geography identifier.

    Returns:
        Dict of column values, or ``None`` if no row exists.
    """
    row = conn.execute(
        """
        SELECT *
        FROM ess_scores
        WHERE geo_id = ?
          AND run_date = (SELECT MAX(run_date) FROM ess_scores WHERE geo_id = ?)
        """,
        [geo_id, geo_id],
    ).fetchdf()
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Endpoints — /scores/top must be declared BEFORE /scores/{geo_id}
# ---------------------------------------------------------------------------


@router.get("/top", response_model=list[ScoreResponse])
async def get_top_scores(
    limit: int = Query(default=20, ge=1, le=100),
    min_confidence: ConfidenceLevel = Query(default="low"),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[ScoreResponse]:
    """Return the highest-stress geographies for the most recent run date.

    Results are ordered by ``ess_score`` descending (most stressed first).
    The ``min_confidence`` filter excludes rows below the requested
    confidence tier: ``"high"`` returns only high-confidence scores;
    ``"medium"`` returns medium and high; ``"low"`` (default) returns all.

    Args:
        limit: Maximum number of results to return. Default 20, maximum 100.
        min_confidence: Minimum confidence tier to include.
        db: Injected DuckDB read connection.

    Returns:
        List of ``ScoreResponse`` objects, ordered by ESS score descending.

    Raises:
        HTTPException: 503 if the ``ess_scores`` table cannot be queried.
    """
    confidence_values = _CONFIDENCE_LEVELS[min_confidence]
    placeholders = ", ".join("?" * len(confidence_values))
    try:
        df = db.execute(
            f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY geo_id
                           ORDER BY run_date DESC
                       ) AS rn
                FROM ess_scores
                WHERE confidence IN ({placeholders})
            )
            SELECT * FROM ranked
            WHERE rn = 1
            ORDER BY ess_score DESC
            LIMIT ?
            """,
            confidence_values + [limit],
        ).fetchdf()
    except Exception as exc:
        logger.error("get_top_scores: DB error: %s", exc)
        raise HTTPException(status_code=503, detail="Score data unavailable") from exc

    return [_row_to_score(row) for row in df.to_dict(orient="records")]


@router.get("/snapshot", response_model=list[ScoreResponse])
async def get_score_snapshot(
    run_date: date | None = Query(default=None),
    geo_level: str | None = Query(default=None),
    query: str | None = Query(default=None, min_length=1),
    limit: int = Query(default=500, ge=1, le=5000),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[ScoreResponse]:
    """Return one score snapshot for the most recent run on or before ``run_date``.

    Args:
        run_date: Desired snapshot date. When omitted, the latest available run
            is returned.
        geo_level: Optional geography-level filter.
        query: Optional case-insensitive ``geo_id`` / ``geo_name`` search term.
        limit: Maximum number of rows to return.
        db: Injected DuckDB read connection.

    Returns:
        List of ``ScoreResponse`` rows for the resolved snapshot date.

    Raises:
        HTTPException: 503 if the database cannot be queried.
    """
    try:
        if run_date is None:
            # Return the latest row per geo so all scored geographies appear,
            # even when different geos were last scored on different dates.
            where: list[str] = []
            params: list[Any] = []
            if geo_level:
                where.append("geo_level = ?")
                params.append(geo_level)
            if query:
                where.append("(LOWER(geo_id) LIKE ? OR LOWER(geo_name) LIKE ?)")
                like = f"%{query.lower()}%"
                params.extend([like, like])

            inner_where = f"WHERE {' AND '.join(where)}" if where else ""
            sql = f"""
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY geo_id
                               ORDER BY run_date DESC
                           ) AS rn
                    FROM ess_scores
                    {inner_where}
                )
                SELECT * FROM ranked
                WHERE rn = 1
                ORDER BY ess_score DESC, geo_name ASC
                LIMIT ?
            """
            params.append(limit)
            df = db.execute(sql, params).fetchdf()
        else:
            snapshot_date = _resolve_snapshot_date(db, run_date)
            if snapshot_date is None:
                return []
            df = _fetch_snapshot_df(
                db,
                snapshot_date=snapshot_date,
                geo_level=geo_level,
                query=query,
                limit=limit,
            )
    except Exception as exc:
        logger.error("get_score_snapshot: DB error: %s", exc)
        raise HTTPException(status_code=503, detail="Score snapshot unavailable") from exc

    return [_row_to_score(row) for row in df.to_dict(orient="records")]


@router.get("/search", response_model=list[ScoreResponse])
async def search_geographies(
    q: str = Query(min_length=1),
    as_of: date | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[ScoreResponse]:
    """Return the latest scored geographies whose ID or name matches ``q``."""
    like = f"%{q.lower()}%"
    try:
        if as_of is None:
            df = db.execute(
                """
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY geo_id
                               ORDER BY run_date DESC
                           ) AS rn
                    FROM ess_scores
                )
                SELECT *
                FROM ranked
                WHERE rn = 1
                  AND (LOWER(geo_id) LIKE ? OR LOWER(geo_name) LIKE ?)
                ORDER BY ess_score DESC, geo_name ASC
                LIMIT ?
                """,
                [like, like, limit],
            ).fetchdf()
        else:
            df = db.execute(
                """
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY geo_id
                               ORDER BY run_date DESC
                           ) AS rn
                    FROM ess_scores
                    WHERE run_date <= ?
                )
                SELECT *
                FROM ranked
                WHERE rn = 1
                  AND (LOWER(geo_id) LIKE ? OR LOWER(geo_name) LIKE ?)
                ORDER BY ess_score DESC, geo_name ASC
                LIMIT ?
                """,
                [as_of, like, like, limit],
            ).fetchdf()
    except Exception as exc:
        logger.error("search_geographies: DB error: %s", exc)
        raise HTTPException(status_code=503, detail="Geography search unavailable") from exc

    return [_row_to_score(row) for row in df.to_dict(orient="records")]


@router.get("/map", response_model=list[MapScoreResponse])
async def get_map_scores(
    run_date: date | None = Query(default=None),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[MapScoreResponse]:
    """Return state-level score payloads for the dashboard choropleth.

    Prefers genuine state-level rows when present. If a state has no direct
    state score, the endpoint falls back to aggregating sub-state rows whose
    identifiers clearly map to that state.

    When ``run_date`` is omitted the endpoint fetches the latest row for
    **each geography individually** rather than a single global snapshot date.
    This ensures that all scored geographies appear on the map even when they
    were last scored on different dates.
    """
    try:
        if run_date is None:
            df = db.execute(
                """
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY geo_id
                               ORDER BY run_date DESC
                           ) AS rn
                    FROM ess_scores
                )
                SELECT * FROM ranked WHERE rn = 1
                ORDER BY ess_score DESC
                """,
            ).fetchdf()
            if df.empty:
                return []
            snapshot_date = _coerce_run_date(df["run_date"].max())
        else:
            snapshot_date = _resolve_snapshot_date(db, run_date)
            if snapshot_date is None:
                return []
            df = _fetch_snapshot_df(db, snapshot_date=snapshot_date)
    except Exception as exc:
        logger.error("get_map_scores: DB error: %s", exc)
        raise HTTPException(status_code=503, detail="Map data unavailable") from exc

    state_rows: dict[str, MapScoreResponse] = {}
    aggregate_rows: dict[str, list[dict[str, Any]]] = {}

    for row in df.to_dict(orient="records"):
        state_code = _extract_state_code(row)
        if not state_code:
            continue
        if str(row.get("geo_level") or "") == "state":
            state_rows[state_code] = _row_to_map_score(row, state_code)
        else:
            aggregate_rows.setdefault(state_code, []).append(row)

    for state_code, rows in aggregate_rows.items():
        if state_code not in state_rows:
            state_rows[state_code] = _aggregate_map_rows(state_code, rows, snapshot_date)

    return [state_rows[state] for state in sorted(state_rows)]


@router.get("/{geo_id}", response_model=ScoreResponse)
async def get_score(
    geo_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> ScoreResponse:
    """Return the latest ESS score for a single geography.

    Always returns the most recent ``run_date`` row in ``ess_scores`` for
    the requested ``geo_id``. Includes full Prediction fields plus the
    computed ``is_trustworthy`` field.

    Args:
        geo_id: Geography identifier (e.g. ``"Detroit-MI"``).
        db: Injected DuckDB read connection.

    Returns:
        ``ScoreResponse`` for the latest run.

    Raises:
        HTTPException: 404 if ``geo_id`` is not in the database.
        HTTPException: 503 if the database cannot be queried.
    """
    try:
        row = _fetch_latest_row(db, geo_id)
    except Exception as exc:
        logger.error("get_score: DB error for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(status_code=503, detail="Score data unavailable") from exc

    if row is None:
        raise HTTPException(status_code=404, detail=f"Geography '{geo_id}' not found")

    return _row_to_score(row)


@router.get("/{geo_id}/history", response_model=TimeSeriesResponse)
async def get_score_history(
    geo_id: str,
    window: Literal["7d", "30d", "90d"] = Query(default="30d"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    as_of: date | None = Query(default=None),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> TimeSeriesResponse:
    """Return the ESS score history for a geography over a rolling window.

    Points are ordered by date ascending, one per day with recorded scores.
    The ``trend`` field summarises the direction of the series over the period.

    Args:
        geo_id: Geography identifier.
        window: Time window — ``"7d"``, ``"30d"``, or ``"90d"``. Used when an
            explicit date range is not supplied.
        start_date: Optional inclusive start date override.
        end_date: Optional inclusive end date override.
        as_of: Optional end date used with ``window`` when ``start_date`` is not
            supplied.
        db: Injected DuckDB read connection.

    Returns:
        ``TimeSeriesResponse`` with points list and computed trend.

    Raises:
        HTTPException: 404 if the geography has no score history.
        HTTPException: 503 if the database cannot be queried.
    """
    window_days = {"7d": 7, "30d": 30, "90d": 90}[window]

    if start_date is not None or end_date is not None:
        resolved_end = end_date or date.today()
        resolved_start = start_date or (resolved_end - timedelta(days=window_days))
    else:
        resolved_end = as_of or date.today()
        resolved_start = resolved_end - timedelta(days=window_days)

    if resolved_end < resolved_start:
        raise HTTPException(
            status_code=400,
            detail="end_date must be on or after start_date",
        )

    try:
        df = db.execute(
            """
            SELECT run_date, ess_score, confidence, missing_sources, anomaly_flags, geo_name
            FROM ess_scores
            WHERE geo_id = ?
              AND run_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY run_date ASC
            """,
            [geo_id, resolved_start.isoformat(), resolved_end.isoformat()],
        ).fetchdf()
    except Exception as exc:
        logger.error("get_score_history: DB error for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(status_code=503, detail="Score history unavailable") from exc

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No score history found for geography '{geo_id}'",
        )

    geo_name = str(df.iloc[0]["geo_name"]) if "geo_name" in df.columns else geo_id

    points = [
        TimeSeriesPoint(
            date=row["run_date"],
            ess_score=_safe_float(row["ess_score"]),
            confidence=row["confidence"],
            missing_sources=_parse_json_list(row.get("missing_sources")),
            anomaly_flag=bool(_parse_json_list(row.get("anomaly_flags"))),
        )
        for row in df.to_dict(orient="records")
    ]

    scores = [p.ess_score for p in points]
    trend = _compute_trend(scores)

    return TimeSeriesResponse(
        geo_id=geo_id,
        geo_name=geo_name,
        period_days=max((resolved_end - resolved_start).days, 1),
        points=points,
        trend=trend,
    )


@router.get("/{geo_id}/drivers")
async def get_score_drivers(
    geo_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict[str, Any]:
    """Return the top SHAP drivers and tier score breakdown for a geography.

    Reads SHAP values from the most recent ``ess_scores`` row and returns
    the top-5 contributors by absolute SHAP value alongside the tier scores.

    Args:
        geo_id: Geography identifier.
        db: Injected DuckDB read connection.

    Returns:
        Dict with keys:
            ``top_shap`` — list of dicts with ``feature``, ``contribution``
            ``tier_breakdown`` — dict with ``tier1_score``, ``tier2_score``,
                ``tier3_score``
            ``run_date`` — date of the underlying row

    Raises:
        HTTPException: 404 if the geography is not found.
        HTTPException: 503 if the database cannot be queried.
    """
    try:
        row = db.execute(
            """
            SELECT shap_values, tier1_score, tier2_score, tier3_score, run_date
            FROM ess_scores
            WHERE geo_id = ?
              AND run_date = (SELECT MAX(run_date) FROM ess_scores WHERE geo_id = ?)
            """,
            [geo_id, geo_id],
        ).fetchdf()
    except Exception as exc:
        logger.error("get_score_drivers: DB error for geo_id=%s: %s", geo_id, exc)
        raise HTTPException(status_code=503, detail="Driver data unavailable") from exc

    if row.empty:
        raise HTTPException(status_code=404, detail=f"Geography '{geo_id}' not found")

    r = row.iloc[0].to_dict()
    shap_values = _parse_json_dict(r.get("shap_values"))

    top_shap = [
        {"feature": name, "contribution": round(val, 4)}
        for name, val in sorted(
            shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:5]
    ]

    return {
        "geo_id": geo_id,
        "run_date": r["run_date"],
        "top_shap": top_shap,
        "tier_breakdown": {
            "tier1_score": _safe_float(r.get("tier1_score")),
            "tier2_score": _safe_float(r.get("tier2_score")),
            "tier3_score": _safe_float(r.get("tier3_score")),
        },
    }
