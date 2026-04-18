"""Observability metrics for the PulseIQ ingestion pipeline.

Every connector must write an ``IngestionMetrics`` record after each run —
whether it succeeds or fails. ``MetricsWriter`` persists those records to
DuckDB and fires MTTD alerts to a configured webhook when thresholds are
breached.

``IngestionMetrics`` and ``SourceFreshnessPayload`` are the canonical Pydantic
models from ``src.contracts``; they are re-exported here for convenience so
connectors can import from a single observability module.

Typical usage::

    from datetime import date, datetime, timezone
    import uuid
    from src.observability.metrics import IngestionMetrics, MetricsWriter, log_ingestion_metrics

    writer = MetricsWriter()
    started = datetime.now(timezone.utc)
    metrics = IngestionMetrics(
        source="bls",
        run_date=date.today(),
        run_id=str(uuid.uuid4()),
        started_at=started,
        completed_at=datetime.now(timezone.utc),
        records_fetched=1_200,
        records_rejected=0,
        records_suspect=3,
        latency_seconds=4.2,
        freshness_status="ok",
        http_retries=0,
        success=True,
        error_message=None,
    )
    writer.write_ingestion_metrics(metrics)
    log_ingestion_metrics(metrics, logger)

    health = writer.get_source_health("bls")
    all_health = writer.get_all_source_health()
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Any

import duckdb
import requests

from src.contracts import IngestionMetrics, SourceFreshnessPayload  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Freshness cadence per source (days between expected fetches)
# ---------------------------------------------------------------------------

SOURCE_CADENCE_DAYS: dict[str, int] = {
    "bls": 7,          # weekly
    "fred": 30,        # monthly
    "census": 365,     # annual
    "news": 1,         # daily
    "trends": 1,       # daily
    "openweather": 1,  # daily
}

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def log_ingestion_metrics(metrics: IngestionMetrics, log: logging.Logger) -> None:
    """Emit an ``IngestionMetrics`` record as a structured JSON INFO log line.

    The message is prefixed with ``"ingestion_metrics "`` for easy filtering.

    Args:
        metrics: Populated ``IngestionMetrics`` to emit.
        log: Logger instance to write to (typically the connector's logger).
    """
    log.info("ingestion_metrics %s", metrics.model_dump_json())


# ---------------------------------------------------------------------------
# MetricsWriter
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_metrics (
    source           VARCHAR   NOT NULL,
    run_date         DATE      NOT NULL,
    run_id           VARCHAR   NOT NULL,
    started_at       TIMESTAMP NOT NULL,
    completed_at     TIMESTAMP NOT NULL,
    records_fetched  INTEGER   NOT NULL,
    records_rejected INTEGER   NOT NULL,
    records_suspect  INTEGER   NOT NULL,
    latency_seconds  DOUBLE    NOT NULL,
    freshness_status VARCHAR   NOT NULL,
    http_retries     INTEGER   NOT NULL DEFAULT 0,
    success          BOOLEAN   NOT NULL,
    error_message    VARCHAR,
    PRIMARY KEY (source, run_date, run_id)
)
"""

_HISTORY_SQL = """
SELECT
    AVG(records_fetched)  AS avg_records,
    AVG(latency_seconds)  AS avg_latency,
    COUNT(*)              AS run_count
FROM ingestion_metrics
WHERE source = ?
  AND run_date >= current_date - INTERVAL 7 DAY
  AND success = TRUE
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO ingestion_metrics
    (source, run_date, run_id, started_at, completed_at,
     records_fetched, records_rejected, records_suspect,
     latency_seconds, freshness_status, http_retries, success, error_message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_HEALTH_SQL = """
SELECT
    MAX(run_date)                                      AS last_run_date,
    MAX(CASE WHEN success THEN run_date END)            AS last_success_date,
    MAX(CASE WHEN success THEN records_fetched END)     AS records_fetched_last_run
FROM ingestion_metrics
WHERE source = ?
"""


class MetricsWriter:
    """Persists ``IngestionMetrics`` to DuckDB and fires MTTD alerts.

    The ``ingestion_metrics`` table is created automatically on first use.
    Writes are idempotent: re-running with the same ``run_id`` overwrites
    the existing row rather than creating a duplicate.

    Args:
        db_path: Absolute path to the DuckDB file. Resolved in order:
            1. ``db_path`` constructor argument
            2. ``DUCKDB_PATH`` environment variable
            3. ``"data/processed/pulseiq.db"`` (default)

            Pass ``":memory:"`` in tests to use an in-memory database.

    Example::

        writer = MetricsWriter()                     # production
        writer = MetricsWriter(db_path=":memory:")   # tests
    """

    def __init__(self, db_path: str | None = None) -> None:
        resolved = (
            db_path
            or os.getenv("DUCKDB_PATH")
            or "data/processed/pulseiq.db"
        )
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(resolved)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the ``ingestion_metrics`` table if it does not yet exist."""
        self._conn.execute(_CREATE_TABLE_SQL)

    def write_ingestion_metrics(self, metrics: IngestionMetrics) -> None:
        """Persist one ingestion run record and fire any triggered MTTD alerts.

        Operations in order:
        1. Fetch 7-day history for this source (before insert) to compute
           rolling averages used in MTTD checks.
        2. Insert or replace the record (idempotent on ``run_id``).
        3. Evaluate MTTD rules and call ``_fire_alert()`` for each breach.

        MTTD rules evaluated:
        - ``success == False``               → critical / ingestion_failure
        - ``records_fetched < avg * 0.5``    → warning  / record_count_drop
          (only when ≥3 historical runs exist)
        - ``latency_seconds > avg * 3``      → warning  / latency_spike
          (only when ≥3 historical runs exist)
        - ``freshness_status == "critical"`` → critical / source_stale

        Args:
            metrics: Populated ``IngestionMetrics`` record to persist.
        """
        # 1. Fetch history before insert
        row = self._conn.execute(_HISTORY_SQL, [metrics.source]).fetchone()
        avg_records: float = row[0] or 0.0
        avg_latency: float = row[1] or 0.0
        run_count: int = row[2] or 0

        # 2. Persist (idempotent)
        self._conn.execute(
            _INSERT_SQL,
            [
                metrics.source,
                metrics.run_date,
                metrics.run_id,
                metrics.started_at,
                metrics.completed_at,
                metrics.records_fetched,
                metrics.records_rejected,
                metrics.records_suspect,
                metrics.latency_seconds,
                metrics.freshness_status,
                metrics.http_retries,
                metrics.success,
                metrics.error_message,
            ],
        )
        logger.debug(
            "Wrote ingestion_metrics: source=%s run_id=%s success=%s",
            metrics.source, metrics.run_id, metrics.success,
        )

        # 3. MTTD checks
        if not metrics.success:
            self._fire_alert(
                severity="critical",
                alert_type="ingestion_failure",
                source=metrics.source,
                message=(
                    f"Ingestion failed for source '{metrics.source}' "
                    f"on {metrics.run_date}: {metrics.error_message}"
                ),
            )

        if run_count >= 3 and avg_records > 0:
            if metrics.records_fetched < avg_records * 0.5:
                self._fire_alert(
                    severity="warning",
                    alert_type="record_count_drop",
                    source=metrics.source,
                    message=(
                        f"Record count for '{metrics.source}' dropped to "
                        f"{metrics.records_fetched} (7-day avg: {avg_records:.0f})"
                    ),
                )

        if run_count >= 3 and avg_latency > 0:
            if metrics.latency_seconds > avg_latency * 3:
                self._fire_alert(
                    severity="warning",
                    alert_type="latency_spike",
                    source=metrics.source,
                    message=(
                        f"Latency for '{metrics.source}' was "
                        f"{metrics.latency_seconds:.1f}s "
                        f"(7-day avg: {avg_latency:.1f}s)"
                    ),
                )

        if metrics.freshness_status == "critical":
            self._fire_alert(
                severity="critical",
                alert_type="source_stale",
                source=metrics.source,
                message=(
                    f"Source '{metrics.source}' freshness is critical "
                    f"as of {metrics.run_date}"
                ),
            )

    def _fire_alert(
        self,
        severity: str,
        alert_type: str,
        source: str,
        message: str,
    ) -> None:
        """POST an alert payload to the configured webhook URL.

        Silently skips if ``ALERT_WEBHOOK_URL`` is not set (e.g. development).
        Swallows all request exceptions so that a webhook failure never
        blocks a metric write.

        Args:
            severity: ``"critical"`` or ``"warning"``.
            alert_type: Machine-readable type (e.g. ``"ingestion_failure"``).
            source: Source identifier the alert relates to.
            message: Human-readable description of the condition.
        """
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if not webhook_url:
            logger.debug(
                "ALERT_WEBHOOK_URL not set — skipping %s/%s alert for %s",
                severity, alert_type, source,
            )
            return

        payload: dict[str, Any] = {
            "severity": severity,
            "alert_type": alert_type,
            "source": source,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            requests.post(webhook_url, json=payload, timeout=10).raise_for_status()
            logger.info(
                "Alert fired: severity=%s type=%s source=%s",
                severity, alert_type, source,
            )
        except requests.exceptions.RequestException as exc:
            logger.error("Alert webhook POST failed: %s", exc)

    def get_source_health(
        self, source: str, days: int = 7
    ) -> SourceFreshnessPayload:
        """Return the freshness status for a single source.

        Queries the ``ingestion_metrics`` table for the most recent run dates
        and record counts, then computes ``freshness_status`` using the
        expected cadence from ``SOURCE_CADENCE_DAYS``.

        Freshness thresholds (multiples of expected cadence):

        - ``days_since ≤ cadence × 1.5`` → ``"ok"``
        - ``days_since ≤ cadence × 3``   → ``"stale"``
        - ``days_since > cadence × 3``   → ``"critical"``
        - no successful run on record    → ``"unknown"``

        Args:
            source: Source identifier to query (e.g. ``"bls"``).
            days: Unused — reserved for future window-scoped queries.
                Currently the query considers all historical runs.

        Returns:
            Populated ``SourceFreshnessPayload`` for the source.
            If the source has no rows, all date fields are ``None`` and
            ``freshness_status`` is ``"unknown"``.
        """
        cadence = SOURCE_CADENCE_DAYS.get(source, 0)
        row = self._conn.execute(_HEALTH_SQL, [source]).fetchone()

        last_success_date: date | None = row[1]
        records_last_run: int = int(row[2]) if row[2] is not None else 0

        if last_success_date is None:
            return SourceFreshnessPayload(
                source=source,
                last_successful_fetch=None,
                days_since_fetch=None,
                freshness_status="unknown",
                expected_cadence_days=cadence,
                records_last_run=0,
            )

        days_since = (date.today() - last_success_date).days

        if cadence == 0:
            freshness_status = "unknown"
        elif days_since <= cadence * 1.5:
            freshness_status = "ok"
        elif days_since <= cadence * 3:
            freshness_status = "stale"
        else:
            freshness_status = "critical"

        return SourceFreshnessPayload(
            source=source,
            last_successful_fetch=last_success_date,
            days_since_fetch=days_since,
            freshness_status=freshness_status,
            expected_cadence_days=cadence,
            records_last_run=records_last_run,
        )

    def get_all_source_health(self) -> list[SourceFreshnessPayload]:
        """Return freshness status for every known source.

        Iterates over all keys in ``SOURCE_CADENCE_DAYS`` and calls
        ``get_source_health()`` for each.

        Returns:
            List of ``SourceFreshnessPayload``, one per known source,
            in the iteration order of ``SOURCE_CADENCE_DAYS``.
        """
        return [self.get_source_health(source) for source in SOURCE_CADENCE_DAYS]
