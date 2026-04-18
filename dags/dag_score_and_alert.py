"""Daily Airflow DAG for scoring, calibration, alerting, and drift monitoring.

Runs every day at 10:00 UTC. Waits for dag_transform_daily to fully succeed
before any scoring work starts, then executes four tasks in sequence:

1. ``task_score``     — batch-scores all geographies via XGBoost
2. ``task_calibrate`` — re-applies isotonic calibration to today's scores
3. ``task_alerts``    — fires suppressed alerts to the configured webhook
4. ``task_monitor``   — runs PSI drift checks and logs a retraining recommendation

Pipeline position::

    dag_ingest_daily (06:00 UTC)
        → dag_transform_daily (08:00 UTC)
            → dag_score_and_alert (10:00 UTC)

No business logic lives in this file — every task imports and delegates
to ``src/`` modules.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.sensors.external_task import ExternalTaskSensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Airflow Variable keys
# ---------------------------------------------------------------------------

_VAR_DUCKDB_PATH = "DUCKDB_PATH"
_VAR_MODEL_RUN_ID = "model_run_id"
_VAR_ALERT_THRESHOLD = "ALERT_THRESHOLD"


# ---------------------------------------------------------------------------
# Shared failure callback (mirrors dag_ingest_daily / dag_transform_daily)
# ---------------------------------------------------------------------------


def on_failure_callback(context: dict[str, Any]) -> None:
    """Log task failures with the DAG id, task id, and failure timestamp.

    Args:
        context: Airflow task context dictionary passed by the scheduler.
    """
    dag_id = context["dag"].dag_id if context.get("dag") else "unknown_dag"
    task_id = (
        context["task_instance"].task_id
        if context.get("task_instance")
        else "unknown_task"
    )
    timestamp = context.get("ts") or str(context.get("logical_date"))
    logger.error(
        "Task failure detected in DAG %s for task %s at %s",
        dag_id,
        task_id,
        timestamp,
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------


@dag(
    dag_id="dag_score_and_alert",
    schedule="0 10 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    on_failure_callback=on_failure_callback,
)
def dag_score_and_alert() -> None:
    """Score all geographies, calibrate, alert, and monitor drift — daily at 10:00 UTC.

    Waits for ``dag_transform_daily`` to fully succeed (specifically its final
    ``dbt_test`` task) before running. ``execution_delta=timedelta(hours=2)``
    maps this DAG's 10:00 UTC logical date back to the transform DAG's
    08:00 UTC logical date.

    Tasks run in strict sequence:

    ``sensor_transform_complete``
        → ``task_score``
        → ``task_calibrate``
        → ``task_alerts``
        → ``task_monitor``

    All inputs are read from Airflow Variables:

    - ``DUCKDB_PATH`` — path to the DuckDB file
      (fallback: ``DUCKDB_PATH`` env var, then ``data/processed/pulseiq.db``)
    - ``model_run_id`` — MLflow run ID of the trained model to score with
    - ``ALERT_THRESHOLD`` — minimum ESS score to consider for alerting
      (default 75)
    """

    # ------------------------------------------------------------------
    # Sensor: wait for dag_transform_daily to finish its last task (dbt_test)
    # ------------------------------------------------------------------

    sensor_transform_complete = ExternalTaskSensor(
        task_id="sensor_transform_complete",
        external_dag_id="dag_transform_daily",
        external_task_id="dbt_test",  # last task in dag_transform_daily
        # transform logical_date = score logical_date − 2 h
        execution_delta=timedelta(hours=2),
        mode="reschedule",
        poke_interval=60,
        timeout=3600,
        on_failure_callback=on_failure_callback,
    )

    # ------------------------------------------------------------------
    # task_score — run XGBoost scoring for every geography
    # ------------------------------------------------------------------

    @task(task_id="task_score")
    def task_score() -> int:
        """Score all geographies and write results to ``ess_scores``.

        Reads ``DUCKDB_PATH`` and ``model_run_id`` from Airflow Variables,
        calls ``score_all_geos()``, and logs the count of geographies scored.

        Returns:
            Count of ``Prediction`` objects written to ``ess_scores``.
        """
        from src.models.predict import score_all_geos

        db_path = Variable.get(
            _VAR_DUCKDB_PATH,
            default_var=os.getenv("DUCKDB_PATH", "data/processed/pulseiq.db"),
        )
        model_run_id = Variable.get(_VAR_MODEL_RUN_ID)

        predictions = score_all_geos(db_path=db_path, model_run_id=model_run_id)
        n_geos = len(predictions)
        logger.info("task_score: scored %d geographies (model_run_id=%s)", n_geos, model_run_id)
        return n_geos

    # ------------------------------------------------------------------
    # task_calibrate — re-apply isotonic calibration to today's scores
    # ------------------------------------------------------------------

    @task(task_id="task_calibrate")
    def task_calibrate() -> int:
        """Re-apply calibration to today's ``ess_scores`` rows.

        Loads the latest ``PulseIQCalibrator`` from ``models/calibration.pkl``.
        If the file does not exist, all rows are marked ``calibrated=False``
        and ``confidence="low"`` — the task does not fail.

        Returns:
            Count of rows updated.
        """
        from src.models.calibration import apply_calibration_to_today

        db_path = Variable.get(
            _VAR_DUCKDB_PATH,
            default_var=os.getenv("DUCKDB_PATH", "data/processed/pulseiq.db"),
        )

        count = apply_calibration_to_today(db_path=db_path)
        logger.info("task_calibrate: updated calibration on %d rows", count)
        return count

    # ------------------------------------------------------------------
    # task_alerts — fire suppressed alerts to the webhook
    # ------------------------------------------------------------------

    @task(task_id="task_alerts")
    def task_alerts() -> None:
        """Check today's scores against threshold and deliver alerts.

        Reads ``ALERT_THRESHOLD`` from Airflow Variables (default 75).
        For each geography exceeding the threshold, checks suppression rules
        (cooldown windows, min-delta) before POSTing an ``AlertPayload`` to
        the configured webhook.

        Logs: geos evaluated, alerts fired, alerts suppressed.
        """
        from src.observability.alerts import fire_alerts_for_today

        db_path = Variable.get(
            _VAR_DUCKDB_PATH,
            default_var=os.getenv("DUCKDB_PATH", "data/processed/pulseiq.db"),
        )
        threshold = float(
            Variable.get(_VAR_ALERT_THRESHOLD, default_var="75")
        )
        webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")
        webhook_secret = os.getenv("ALERT_WEBHOOK_SECRET", "")

        if not webhook_url:
            logger.warning(
                "task_alerts: ALERT_WEBHOOK_URL is not set — alert delivery skipped."
            )
            return

        fired, suppressed = fire_alerts_for_today(
            db_path=db_path,
            threshold=threshold,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
        )
        logger.info(
            "task_alerts: threshold=%.0f fired=%d suppressed=%d",
            threshold, fired, suppressed,
        )

    # ------------------------------------------------------------------
    # task_monitor — drift detection and retraining recommendation
    # ------------------------------------------------------------------

    @task(task_id="task_monitor")
    def task_monitor() -> None:
        """Run all drift checks and persist the report to ``monitor_log``.

        Runs ``feature_drift``, ``score_distribution_drift``, and
        ``missing_source_drift``. Calls ``retraining_recommendation()``
        to synthesise the signals into a single verdict.

        If ``recommendation == "immediate"``, logs at ``CRITICAL`` level
        so that log-aggregation alerts can surface it to on-call engineers.

        The full report is written to the ``monitor_log`` DuckDB table.
        """
        from src.models.monitor import run_monitor_and_log

        db_path = Variable.get(
            _VAR_DUCKDB_PATH,
            default_var=os.getenv("DUCKDB_PATH", "data/processed/pulseiq.db"),
        )

        report = run_monitor_and_log(db_path=db_path)
        logger.info(
            "task_monitor: recommendation=%s triggered_by=%s",
            report["recommendation"],
            report["triggered_by"],
        )

    # ------------------------------------------------------------------
    # Task dependency chain
    # ------------------------------------------------------------------

    scored = task_score()
    calibrated = task_calibrate()
    alerted = task_alerts()
    monitored = task_monitor()

    sensor_transform_complete >> scored >> calibrated >> alerted >> monitored


dag = dag_score_and_alert()
