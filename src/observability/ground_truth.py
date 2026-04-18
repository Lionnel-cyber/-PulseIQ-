"""Ground truth logging for the PulseIQ observability layer.

Logs are written at three points in the pipeline:

1. **After ingestion** — ``log_raw_signal()`` for every validated record,
   before it is written to the mart.
2. **After scoring** — ``log_prediction()`` with the complete feature vector.
   The ``signal_snapshot`` is mandatory: without it no prediction can be
   reproduced or audited later.
3. **When events are confirmed** — ``log_ground_truth_event()`` for real-world
   economic events. These rows are gold-labelled training data.

All three tables are append-or-replace: rows are never deleted. Re-running
with the same primary key values overwrites the existing row.

Typical usage::

    from src.observability.ground_truth import GroundTruthLogger

    logger = GroundTruthLogger()

    # 1. After ingestion
    logger.log_raw_signal(
        source="bls",
        geo_id="26",
        run_date=date.today(),
        raw_value=248_000.0,
        processed_value=248_000.0,
        validation_status="valid",
        anomaly_flag=False,
    )

    # 2. After scoring
    logger.log_prediction(
        geo_id="26",
        run_date=date.today(),
        ess_score=72.4,
        confidence="high",
        shap_values={"bls_jobless_claims_delta": 0.35, ...},
        signal_snapshot=feature_vector_dict,
    )

    # 3. When event confirmed
    logger.log_ground_truth_event(
        geo_id="26",
        event_date=date(2024, 3, 1),
        event_type="mass_layoff",
        event_source="BLS_WARN_ACT",
        severity="high",
        confirmed_date=date.today(),
    )
"""

import json
import logging
import os
from datetime import date
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_RAW_SIGNAL_SQL = """
CREATE TABLE IF NOT EXISTS raw_signal_log (
    source            VARCHAR  NOT NULL,
    geo_id            VARCHAR  NOT NULL,
    run_date          DATE     NOT NULL,
    raw_value         DOUBLE,
    processed_value   DOUBLE,
    validation_status VARCHAR  NOT NULL,
    anomaly_flag      BOOLEAN  NOT NULL,
    PRIMARY KEY (source, geo_id, run_date)
)
"""

_CREATE_PREDICTION_SQL = """
CREATE TABLE IF NOT EXISTS prediction_log (
    geo_id           VARCHAR  NOT NULL,
    run_date         DATE     NOT NULL,
    ess_score        DOUBLE   NOT NULL,
    confidence       VARCHAR  NOT NULL,
    shap_values      VARCHAR  NOT NULL,
    signal_snapshot  VARCHAR  NOT NULL,
    PRIMARY KEY (geo_id, run_date)
)
"""

_CREATE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS ground_truth_events (
    geo_id          VARCHAR  NOT NULL,
    event_date      DATE     NOT NULL,
    event_type      VARCHAR  NOT NULL,
    event_source    VARCHAR  NOT NULL,
    severity        VARCHAR  NOT NULL,
    confirmed_date  DATE     NOT NULL,
    PRIMARY KEY (geo_id, event_date, event_type, event_source)
)
"""

# ---------------------------------------------------------------------------
# GroundTruthLogger
# ---------------------------------------------------------------------------


class GroundTruthLogger:
    """Persists raw signals, predictions, and confirmed events to DuckDB.

    All three tables are created automatically on first use.  Writes are
    idempotent: re-running with the same primary key values overwrites the
    existing row rather than duplicating it.  Rows are never deleted.

    Args:
        db_path: Absolute path to the DuckDB file. Resolved in order:
            1. ``db_path`` constructor argument
            2. ``DUCKDB_PATH`` environment variable
            3. ``"data/processed/pulseiq.db"`` (default)

            Pass ``":memory:"`` in tests.

    Example::

        logger = GroundTruthLogger()              # production
        logger = GroundTruthLogger(":memory:")    # tests
    """

    def __init__(self, db_path: str | None = None) -> None:
        resolved = (
            db_path
            or os.getenv("DUCKDB_PATH")
            or "data/processed/pulseiq.db"
        )
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(resolved)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create all three ground-truth tables if they do not exist."""
        self._conn.execute(_CREATE_RAW_SIGNAL_SQL)
        self._conn.execute(_CREATE_PREDICTION_SQL)
        self._conn.execute(_CREATE_EVENTS_SQL)

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def log_raw_signal(
        self,
        source: str,
        geo_id: str,
        run_date: date,
        raw_value: float | None,
        processed_value: float | None,
        validation_status: str,
        anomaly_flag: bool,
    ) -> None:
        """Log a single validated signal record before it is written to the mart.

        Called once per validated record immediately after
        ``validation/rules.py`` returns a result.  The ``validation_status``
        parameter accepts either a plain string (``"valid"``, ``"suspect"``,
        ``"rejected"``) or the ``ValidationResult.value`` attribute.

        Idempotent: re-running with the same ``(source, geo_id, run_date)``
        overwrites the existing row.

        Args:
            source: Data source identifier (e.g. ``"bls"``, ``"fred"``).
            geo_id: Geography identifier (e.g. FIPS code, ``"US"``).
            run_date: Date the ingestion run was executed.
            raw_value: Original value as received from the source API.
                ``None`` if the source did not provide a numeric value.
            processed_value: Value after any normalisation or transformation.
                ``None`` if transformation was not applicable.
            validation_status: Outcome string from the validation layer:
                ``"valid"``, ``"suspect"``, or ``"rejected"``.
            anomaly_flag: ``True`` if the record was flagged as anomalous
                (corresponds to ``ValidationResult.SUSPECT``).

        Raises:
            duckdb.Error: On database write failure.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO raw_signal_log
                (source, geo_id, run_date, raw_value, processed_value,
                 validation_status, anomaly_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [source, geo_id, run_date, raw_value, processed_value,
             validation_status, anomaly_flag],
        )
        logger.debug(
            "Logged raw signal: source=%s geo_id=%s run_date=%s status=%s",
            source, geo_id, run_date, validation_status,
        )

    def log_prediction(
        self,
        geo_id: str,
        run_date: date,
        ess_score: float,
        confidence: str,
        shap_values: dict[str, float],
        signal_snapshot: dict[str, Any],
    ) -> None:
        """Log a model prediction with its complete feature vector.

        **The ``signal_snapshot`` is mandatory.** It must contain the full
        feature vector passed to the model for this prediction. Without it,
        predictions cannot be reproduced, audited, or used for retraining.

        Both ``shap_values`` and ``signal_snapshot`` are serialised to JSON
        strings before storage and can be recovered with ``json.loads()``.

        Idempotent: re-running with the same ``(geo_id, run_date)`` overwrites
        the existing row, reflecting the latest scoring run for that day.

        Args:
            geo_id: Geography identifier.
            run_date: Date the scoring run was executed.
            ess_score: Calibrated Economic Stress Score (0–100).
            confidence: Model confidence label: ``"high"``, ``"medium"``,
                or ``"low"``.
            shap_values: Dict mapping feature name → SHAP contribution value
                for this prediction.
            signal_snapshot: Complete feature vector dict as passed to the
                model. Must contain all 14 feature columns defined in
                ``contracts.py`` (once implemented).

        Raises:
            duckdb.Error: On database write failure.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO prediction_log
                (geo_id, run_date, ess_score, confidence,
                 shap_values, signal_snapshot)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                geo_id,
                run_date,
                ess_score,
                confidence,
                json.dumps(shap_values),
                json.dumps(signal_snapshot),
            ],
        )
        logger.info(
            "Logged prediction: geo_id=%s run_date=%s ess_score=%.1f confidence=%s",
            geo_id, run_date, ess_score, confidence,
        )

    def log_ground_truth_event(
        self,
        geo_id: str,
        event_date: date,
        event_type: str,
        event_source: str,
        severity: str,
        confirmed_date: date,
    ) -> None:
        """Log a confirmed real-world economic event as gold training data.

        These rows are the labelled ground truth used to evaluate and retrain
        the model.  They must never be silently deleted; corrections should be
        made by writing a replacement row with corrected field values.

        Idempotent: re-running with the same
        ``(geo_id, event_date, event_type, event_source)`` overwrites the
        existing row.

        Args:
            geo_id: Geography where the event occurred.
            event_date: Date the event took place (not the confirmed date).
            event_type: One of ``"mass_layoff"``, ``"plant_closure"``,
                ``"bankruptcy_spike"``.
            event_source: Origin of the confirmation: ``"BLS_WARN_ACT"``,
                ``"news_confirmed"``, or ``"manual"``.
            severity: Impact severity label (e.g. ``"low"``, ``"medium"``,
                ``"high"``). No fixed enum — caller defines the scale.
            confirmed_date: Date the event was confirmed and this record was
                created.

        Raises:
            duckdb.Error: On database write failure.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO ground_truth_events
                (geo_id, event_date, event_type, event_source,
                 severity, confirmed_date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [geo_id, event_date, event_type, event_source,
             severity, confirmed_date],
        )
        logger.info(
            "Logged ground truth event: geo_id=%s event_date=%s type=%s source=%s",
            geo_id, event_date, event_type, event_source,
        )
