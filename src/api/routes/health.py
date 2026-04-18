"""Health and observability endpoints for the PulseIQ API.

Answers the three trust questions about infrastructure state:

    GET /health/freshness   — How fresh is the data?
    GET /health/model       — What model produced the scores?
    GET /health/pipeline    — Can I trust this run?

All responses use contracts from ``src/contracts.py`` — no inline schemas.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
from fastapi import APIRouter, HTTPException, Request

from src.contracts import (
    BenchmarkSummaryResponse,
    HealthDashboardResponse,
    HealthResponse,
    ModelHealthSummary,
    ModelVersionResponse,
    PipelineHealthSummary,
    PipelineStatusResponse,
    SourceFreshnessPayload,
    SourceHealthRow,
)
from src.models.evaluate import PulseIQEvaluator
from src.observability.metrics import MetricsWriter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

_SLOW_LATENCY_MULTIPLIER: float = 1.5


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_latest_feature_version(models_dir: Path) -> dict[str, Any] | None:
    """Scan ``models/`` for the most recently modified ``feature_version.json``.

    ``train.py`` saves artifacts to ``models/{run_id}/feature_version.json``.
    This helper finds the newest such file across all run subdirectories.

    Args:
        models_dir: Root ``models/`` directory path.

    Returns:
        Parsed JSON dict from the most recent ``feature_version.json``,
        or ``None`` if no such file exists.
    """
    if not models_dir.exists():
        return None

    candidates = list(models_dir.rglob("feature_version.json"))
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read feature_version.json at %s: %s", latest, exc)
        return None


def _find_latest_feature_version_file(models_dir: Path) -> Path | None:
    """Return the most recently modified ``feature_version.json`` file."""
    if not models_dir.exists():
        return None

    candidates = list(models_dir.rglob("feature_version.json"))
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _coerce_datetime(value: Any) -> datetime | None:
    """Coerce DuckDB timestamp-like values to timezone-aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _finite_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite float, else ``None``."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _build_benchmark_summary(db_path: str) -> BenchmarkSummaryResponse:
    """Return a contract-safe benchmark summary for dashboard consumers."""
    try:
        benchmark = PulseIQEvaluator(db_path).benchmark()
    except Exception as exc:
        logger.warning("_build_benchmark_summary: benchmark failed: %s", exc)
        benchmark = {
            "model_rmse": None,
            "baseline_rmse": None,
            "improvement_pct": None,
            "verdict": "unknown",
            "warning": f"Benchmark unavailable: {exc}",
        }

    return BenchmarkSummaryResponse(
        model_rmse=_finite_or_none(benchmark.get("model_rmse")),
        baseline_rmse=_finite_or_none(benchmark.get("baseline_rmse")),
        improvement_pct=_finite_or_none(benchmark.get("improvement_pct")),
        verdict=str(benchmark.get("verdict") or "unknown"),
        warning=(
            str(benchmark["warning"])
            if benchmark.get("warning") is not None
            else None
        ),
    )


def _build_source_health_rows(db_path: str) -> list[SourceHealthRow]:
    """Assemble dashboard-ready source health rows from ingestion metrics."""
    writer = MetricsWriter(db_path=db_path)
    freshness_map = {
        payload.source: payload
        for payload in writer.get_all_source_health()
    }

    rows: list[SourceHealthRow] = []
    conn = writer._conn

    for source in freshness_map:
        try:
            latest = conn.execute(
                """
                SELECT completed_at, success, records_fetched, latency_seconds
                FROM ingestion_metrics
                WHERE source = ?
                ORDER BY completed_at DESC, run_date DESC
                LIMIT 1
                """,
                [source],
            ).fetchone()
            history = conn.execute(
                """
                SELECT AVG(records_fetched) AS avg_records,
                       AVG(latency_seconds) AS avg_latency
                FROM ingestion_metrics
                WHERE source = ?
                  AND success = TRUE
                  AND run_date >= current_date - INTERVAL 7 DAY
                """,
                [source],
            ).fetchone()
        except Exception as exc:
            logger.debug("_build_source_health_rows: source=%s query failed: %s", source, exc)
            latest = None
            history = None

        freshness = freshness_map[source]
        last_run = _coerce_datetime(latest[0]) if latest else None
        latest_success = bool(latest[1]) if latest else False
        latest_records = int(latest[2]) if latest and latest[2] is not None else 0
        latest_latency = (
            float(latest[3]) if latest and latest[3] is not None else None
        )
        avg_records = float(history[0]) if history and history[0] is not None else None
        avg_latency = float(history[1]) if history and history[1] is not None else None

        if avg_records is None or avg_records <= 0 or latest is None:
            trend_7d = "unknown"
        elif latest_records > avg_records * 1.1:
            trend_7d = "improving"
        elif latest_records < avg_records * 0.9:
            trend_7d = "deteriorating"
        else:
            trend_7d = "stable"

        status = "ok"
        if (
            latest is None
            or not latest_success
            or freshness.freshness_status in ("critical", "unknown")
        ):
            status = "down"
        elif (
            freshness.freshness_status == "stale"
            or (
                latest_latency is not None
                and avg_latency is not None
                and avg_latency > 0
                and latest_latency > avg_latency * _SLOW_LATENCY_MULTIPLIER
            )
        ):
            status = "slow"

        rows.append(
            SourceHealthRow(
                source=source,
                last_run=last_run,
                status=status,
                records=latest_records,
                latency_seconds=latest_latency,
                trend_7d=trend_7d,
            )
        )

    return rows


def _build_health_dashboard_response(db_path: str) -> HealthDashboardResponse:
    """Assemble the combined health payload used by the Streamlit dashboard."""
    checked_at = datetime.now(tz=timezone.utc)
    source_health = _build_source_health_rows(db_path)
    pipeline_status = _build_pipeline_status(db_path)
    benchmark = _build_benchmark_summary(db_path)

    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    feature_version_file = _find_latest_feature_version_file(models_dir)
    trained_at = None
    if feature_version_file is not None:
        trained_at = datetime.fromtimestamp(
            feature_version_file.stat().st_mtime,
            tz=timezone.utc,
        )

    model_meta = get_model_metadata(db_path)

    anomaly_flags_count = 0
    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM ess_scores
                WHERE run_date = (SELECT MAX(run_date) FROM ess_scores)
                  AND anomaly_flags IS NOT NULL
                  AND anomaly_flags <> '[]'
                """
            ).fetchone()
            anomaly_flags_count = int(row[0]) if row and row[0] is not None else 0
    except Exception as exc:
        logger.warning("_build_health_dashboard_response: anomaly count unavailable: %s", exc)

    return HealthDashboardResponse(
        checked_at=checked_at,
        source_health=source_health,
        model_info=ModelHealthSummary(
            version=model_meta.model_version,
            trained_at=trained_at or model_meta.trained_at,
            calibrated=model_meta.calibrated,
            benchmark=benchmark,
        ),
        pipeline_info=PipelineHealthSummary(
            status=pipeline_status.status,
            last_ingest_run=pipeline_status.last_ingest_run,
            last_transform_run=pipeline_status.last_transform_run,
            last_score_run=pipeline_status.last_score_run,
            anomaly_flags_count=anomaly_flags_count,
            failures=pipeline_status.failures,
        ),
    )


def _build_pipeline_status(
    db_path: str,
) -> PipelineStatusResponse:
    """Query DuckDB tables to derive the current pipeline status.

    Checks:
    - ``ingestion_metrics`` for last successful connector run per source
    - ``ess_scores`` for the most recent scoring run date

    Args:
        db_path: Path to the DuckDB file.

    Returns:
        Populated ``PipelineStatusResponse``.
    """
    now = datetime.now(tz=timezone.utc)
    failures: list[str] = []
    last_ingest_run: datetime | None = None
    last_transform_run: datetime | None = None
    last_score_run: datetime | None = None

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            # Last successful connector run (proxy for ingest DAG)
            row = conn.execute(
                """
                SELECT MAX(completed_at) AS last_run
                FROM ingestion_metrics
                WHERE success = TRUE
                """
            ).fetchone()
            if row and row[0] is not None:
                ts = row[0]
                last_ingest_run = (
                    ts.replace(tzinfo=timezone.utc)
                    if isinstance(ts, datetime) and ts.tzinfo is None
                    else ts
                    if isinstance(ts, datetime)
                    else datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
                )

            # Last dbt transform run (source="dbt" convention)
            row = conn.execute(
                """
                SELECT MAX(completed_at) AS last_run
                FROM ingestion_metrics
                WHERE success = TRUE AND source = 'dbt'
                """
            ).fetchone()
            if row and row[0] is not None:
                ts = row[0]
                last_transform_run = (
                    ts.replace(tzinfo=timezone.utc)
                    if isinstance(ts, datetime) and ts.tzinfo is None
                    else ts
                    if isinstance(ts, datetime)
                    else datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
                )

            # Last scoring run — max run_date in ess_scores
            row = conn.execute("SELECT MAX(run_date) FROM ess_scores").fetchone()
            if row and row[0] is not None:
                d = row[0]
                last_score_run = datetime(
                    d.year, d.month, d.day, tzinfo=timezone.utc
                )
    except Exception as exc:
        logger.warning("_build_pipeline_status: DB query error: %s", exc)
        failures.append(f"Pipeline status query failed: {exc}")

    # Determine status
    if last_score_run is None or (now - last_score_run) > timedelta(hours=24):
        failures.insert(0, "Scoring pipeline has not run in >24 hours")
        status = "down"
    elif last_ingest_run is not None and (now - last_ingest_run) > timedelta(hours=30):
        failures.append("Ingestion DAG delayed — last run >30 hours ago")
        status = "degraded"
    elif failures:
        status = "degraded"
    else:
        status = "ok"

    return PipelineStatusResponse(
        status=status,
        last_ingest_run=last_ingest_run,
        last_transform_run=last_transform_run,
        last_score_run=last_score_run,
        checked_at=now,
        failures=failures,
    )


def get_model_metadata(db_path: str) -> ModelVersionResponse:
    """Return deployed model metadata independent of FastAPI request objects."""
    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    fv_data = _find_latest_feature_version(models_dir)

    if fv_data is None:
        logger.warning("get_model_metadata: no feature_version.json found in %s", models_dir)
        return ModelVersionResponse(
            model_version="unknown",
            feature_version="unknown",
            trained_at=None,
            calibrated=False,
            calibration_samples=0,
            mlflow_run_id=None,
        )

    feature_version = str(fv_data.get("feature_version", "unknown"))
    mlflow_run_id: str | None = fv_data.get("mlflow_run_id")

    calibrated = False
    calibration_samples = 0
    trained_at: datetime | None = None
    model_version = "unknown"

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            row = conn.execute(
                """
                SELECT model_version, calibrated, MAX(run_date) as last_run
                FROM ess_scores
                GROUP BY model_version, calibrated
                ORDER BY last_run DESC
                LIMIT 1
                """
            ).fetchone()
            if row:
                model_version = str(row[0] or "unknown")
                calibrated = bool(row[1])
                if row[2] is not None:
                    trained_at = datetime(
                        row[2].year, row[2].month, row[2].day, tzinfo=timezone.utc
                    )
    except Exception as exc:
        logger.warning("get_model_metadata: could not read ess_scores: %s", exc)

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM ground_truth_events"
            ).fetchone()
            calibration_samples = int(row[0]) if row else 0
    except Exception:
        calibration_samples = 0

    return ModelVersionResponse(
        model_version=model_version,
        feature_version=feature_version,
        trained_at=trained_at,
        calibrated=calibrated,
        calibration_samples=calibration_samples,
        mlflow_run_id=mlflow_run_id,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/freshness", response_model=list[SourceFreshnessPayload])
async def get_freshness(request: Request) -> list[SourceFreshnessPayload]:
    """Return per-source data freshness for all known PulseIQ sources.

    Queries the ``ingestion_metrics`` DuckDB table (written by every connector
    after each run) to compute how stale each source currently is relative to
    its expected cadence.

    Sources tracked: bls, fred, census, news, trends, openweather.

    Args:
        request: FastAPI request object (provides ``app.state.db_path``).

    Returns:
        List of ``SourceFreshnessPayload``, one per known source.

    Raises:
        HTTPException: 503 if the metrics database cannot be queried.
    """
    db_path: str = request.app.state.db_path
    try:
        writer = MetricsWriter(db_path=db_path)
        return writer.get_all_source_health()
    except Exception as exc:
        logger.error("get_freshness: MetricsWriter error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Freshness data unavailable — metrics database error",
        ) from exc


@router.get("/model", response_model=ModelVersionResponse)
async def get_model_version(request: Request) -> ModelVersionResponse:
    """Return metadata about the currently deployed scoring model.

    Reads ``feature_version.json`` from the most recently modified run
    subdirectory under ``models/``, then attempts to resolve the MLflow
    run ID from the same artifact.

    Args:
        request: FastAPI request object (unused — present for consistency).

    Returns:
        ``ModelVersionResponse`` with version, calibration, and MLflow metadata.

    Raises:
        HTTPException: 503 if the model artifacts directory cannot be read.
    """
    db_path: str = request.app.state.db_path
    return get_model_metadata(db_path)


@router.get("/dashboard", response_model=HealthDashboardResponse)
async def get_dashboard_health(request: Request) -> HealthDashboardResponse:
    """Return the combined health payload used by the Streamlit dashboard."""
    db_path: str = request.app.state.db_path
    try:
        return _build_health_dashboard_response(db_path)
    except Exception as exc:
        logger.error("get_dashboard_health: unexpected error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Dashboard health unavailable",
        ) from exc


@router.get("/pipeline", response_model=PipelineStatusResponse)
async def get_pipeline_status(request: Request) -> PipelineStatusResponse:
    """Return the current status of the three PulseIQ pipeline DAGs.

    Derives status from the most recent ingestion and scoring timestamps
    in DuckDB. A score run more than 24 hours ago is reported as ``"down"``.

    Args:
        request: FastAPI request object (provides ``app.state.db_path``).

    Returns:
        ``PipelineStatusResponse`` with timestamps and any active failures.

    Raises:
        HTTPException: 503 if pipeline status cannot be determined.
    """
    db_path: str = request.app.state.db_path
    try:
        return _build_pipeline_status(db_path)
    except Exception as exc:
        logger.error("get_pipeline_status: unexpected error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Pipeline status unavailable",
        ) from exc
