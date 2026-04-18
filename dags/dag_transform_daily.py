"""Daily Airflow DAG for dbt transformations in PulseIQ.

Runs every day at 08:00 UTC. Waits for dag_ingest_daily to fully succeed
before executing any dbt work, then runs the full model chain and its tests.
If dbt test fails the DAG fails — dag_score_and_alert will not proceed.

Pipeline position:
    dag_ingest_daily (06:00) → dag_transform_daily (08:00) → dag_score_and_alert (10:00)
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pendulum
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor

logger = logging.getLogger(__name__)

# Path to the dbt project root, relative to Airflow's working directory.
# Airflow is launched from the repo root so this matches the layout in CLAUDE.md.
_DBT_PROJECT_DIR = "src/transforms"


def on_failure_callback(context: dict[str, Any]) -> None:
    """Log task failures with the DAG id, task id, and failure timestamp.

    Args:
        context: Airflow task context dictionary passed to the callback.
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


@dag(
    dag_id="dag_transform_daily",
    schedule="0 8 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    on_failure_callback=on_failure_callback,
)
def dag_transform_daily() -> None:
    """Run the full dbt transformation pipeline daily at 08:00 UTC.

    Waits for dag_ingest_daily to fully succeed (all five connectors), then:

    1. ``dbt_run``  — materialises all models: staging → intermediate → marts.
    2. ``dbt_test`` — executes all schema and singular tests.  The DAG fails
       immediately if any test fails so that dag_score_and_alert never runs
       on bad data.

    The ExternalTaskSensor targets the entire dag_ingest_daily DAG
    (``external_task_id=None``) rather than a single task because the ingest
    connectors run in parallel inside a TaskGroup with no explicit terminal
    task.  ``execution_delta=timedelta(hours=2)`` maps this DAG's 08:00 UTC
    logical date back to dag_ingest_daily's 06:00 UTC logical date.
    """

    sensor_ingest_complete = ExternalTaskSensor(
        task_id="sensor_ingest_complete",
        external_dag_id="dag_ingest_daily",
        # None → wait for ALL tasks in dag_ingest_daily to succeed
        external_task_id=None,
        # ingest logical_date = transform logical_date − 2 h
        execution_delta=timedelta(hours=2),
        # reschedule mode frees the worker slot while waiting
        mode="reschedule",
        poke_interval=60,
        timeout=3600,
        on_failure_callback=on_failure_callback,
    )

    task_dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"dbt run --project-dir {_DBT_PROJECT_DIR}",
        on_failure_callback=on_failure_callback,
    )

    task_dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"dbt test --project-dir {_DBT_PROJECT_DIR}",
        on_failure_callback=on_failure_callback,
    )

    sensor_ingest_complete >> task_dbt_run >> task_dbt_test


dag = dag_transform_daily()
