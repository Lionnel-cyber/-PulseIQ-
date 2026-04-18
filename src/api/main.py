"""PulseIQ FastAPI application entry point.

Creates the FastAPI ``app`` instance, registers all routers under the
``/api/v1`` prefix, and performs startup verification of the DuckDB
database connection.

Usage::

    uvicorn src.api.main:app --reload

Five trust endpoints, one root shortcut:

    GET /api/v1/health            — data freshness (shortcut to /health/freshness)
    GET /api/v1/scores/...        — score data
    GET /api/v1/explain/...       — explanations
    GET /api/v1/alerts/...        — alert history
    GET /api/v1/health/...        — full health suite
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import duckdb
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import alerts, explain, health, scores
from src.contracts import HealthResponse

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Perform startup checks and store app-level state.

    Checks:
    1. The DuckDB file path resolves from ``DUCKDB_PATH`` env var.
    2. A test ``SELECT 1`` query succeeds.

    Warns (does not fail) if no ``feature_version.json`` exists in the
    ``models/`` directory — the model health endpoint will return degraded
    status in that case.

    Sets ``app.state.db_path`` so all route dependencies can read it.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application after successful startup.

    Raises:
        RuntimeError: If the DuckDB file is missing or the connection fails.
    """
    db_path = os.getenv("DUCKDB_PATH", "data/processed/pulseiq.db")

    if db_path != ":memory:" and not Path(db_path).exists():
        raise RuntimeError(
            f"DuckDB database not found at '{db_path}'. "
            "Set DUCKDB_PATH or run the ingestion + transform pipelines first."
        )

    try:
        with duckdb.connect(db_path) as conn:
            conn.execute("SELECT 1")
        logger.info("DuckDB connection verified: %s", db_path)
    except Exception as exc:
        raise RuntimeError(f"DuckDB connection failed at '{db_path}': {exc}") from exc

    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    fv_files = list(models_dir.rglob("feature_version.json")) if models_dir.exists() else []
    if not fv_files:
        logger.warning(
            "No feature_version.json found under '%s' — "
            "GET /health/model will return 'unknown' until training completes.",
            models_dir,
        )
    else:
        logger.info("Model artifacts found: %d feature_version.json file(s)", len(fv_files))

    app.state.db_path = db_path
    logger.info("PulseIQ API startup complete")

    yield

    logger.info("PulseIQ API shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application.

    Registers all four route modules under ``/api/v1``. Called once at
    module load so ``uvicorn src.api.main:app`` works.

    Returns:
        Configured ``FastAPI`` instance with lifespan and all routers attached.
    """
    application = FastAPI(
        title="PulseIQ API",
        version="1.0.0",
        description=(
            "Near real-time Economic Stress Intelligence — "
            "daily ESS scores for US geographies with confidence, "
            "freshness, and explanation context."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(scores.router, prefix="/api/v1")
    application.include_router(explain.router, prefix="/api/v1")
    application.include_router(alerts.router, prefix="/api/v1")
    application.include_router(health.router, prefix="/api/v1")

    return application


app = create_app()


# ---------------------------------------------------------------------------
# Root health shortcut
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def root_health(request: Request) -> HealthResponse:
    """Aggregate pipeline health — shortcut to ``GET /health/freshness``.

    Returns a ``HealthResponse`` summarising source freshness across all
    known PulseIQ data sources. Suitable for load-balancer health checks
    and monitoring dashboards.

    The status field reflects the worst-case source freshness:
    - ``"ok"``      — all sources within SLA
    - ``"degraded"`` — one or more sources stale
    - ``"down"``    — critical source missing or no successful runs on record

    Args:
        request: FastAPI request providing ``app.state.db_path``.

    Returns:
        ``HealthResponse`` with per-source freshness and aggregate status.

    Raises:
        HTTPException: 503 if the metrics database cannot be queried.
    """
    from datetime import datetime, timezone

    from src.observability.metrics import MetricsWriter

    db_path: str = request.app.state.db_path
    try:
        writer = MetricsWriter(db_path=db_path)
        source_freshness = writer.get_all_source_health()
    except Exception as exc:
        logger.error("root_health: MetricsWriter error: %s", exc)
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Health data unavailable — metrics database error",
        ) from exc

    stale = [s.source for s in source_freshness if s.freshness_status in ("stale", "critical")]
    critical = [s.source for s in source_freshness if s.freshness_status == "critical"]
    unknown = [s.source for s in source_freshness if s.freshness_status == "unknown"]

    if critical or unknown:
        status = "down"
    elif stale:
        status = "degraded"
    else:
        status = "ok"

    # Overall data quality: fraction of sources that are "ok"
    ok_count = sum(1 for s in source_freshness if s.freshness_status == "ok")
    overall_dq = ok_count / len(source_freshness) if source_freshness else 0.0

    return HealthResponse(
        status=status,
        checked_at=datetime.now(tz=timezone.utc),
        source_freshness=source_freshness,
        stale_sources=stale,
        overall_data_quality=round(overall_dq, 3),
    )
