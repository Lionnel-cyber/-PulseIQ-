"""Daily Airflow DAG for raw-source ingestion in PulseIQ.

This DAG runs every day at 06:00 UTC and triggers all five source connectors
in parallel to fetch the latest raw data and write it into the partitioned
raw-data lake. It performs ingestion only and intentionally contains no
transformation or scoring logic.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

_MISSING_VARIABLE_JSON = '"__MISSING__"'
_OPENWEATHER_ZIP_VARIABLE = "openweather_zip_codes"


def on_failure_callback(context: dict[str, Any]) -> None:
    """Log task failures with the DAG id, task id, and failure timestamp.

    Args:
        context: Airflow task context dictionary passed to the callback.
    """
    dag_id = context["dag"].dag_id if context.get("dag") else "unknown_dag"
    task_id = context["task_instance"].task_id if context.get("task_instance") else "unknown_task"
    timestamp = context.get("ts") or str(context.get("logical_date"))
    logger.error(
        "Task failure detected in DAG %s for task %s at %s",
        dag_id,
        task_id,
        timestamp,
    )


def _get_optional_json_list(variable_name: str) -> list[str] | None:
    """Read a JSON-array Airflow Variable, returning ``None`` if it is absent.

    Args:
        variable_name: Name of the Airflow Variable to read.

    Returns:
        A list of string values, or ``None`` if the Variable is missing.

    Raises:
        ValueError: If the Variable exists but is not a JSON array.
    """
    value = Variable.get(
        variable_name,
        default_var=_MISSING_VARIABLE_JSON,
        deserialize_json=True,
    )
    if value == "__MISSING__":
        return None
    if not isinstance(value, list):
        raise ValueError(
            f"Airflow Variable '{variable_name}' must be a JSON array."
        )
    return [str(item) for item in value]


def _get_required_json_list(variable_name: str) -> list[str]:
    """Read a required JSON-array Airflow Variable.

    Args:
        variable_name: Name of the Airflow Variable to read.

    Returns:
        A non-empty list of string values.

    Raises:
        ValueError: If the Variable is missing, not a JSON array, or empty.
    """
    value = _get_optional_json_list(variable_name)
    if value is None or not value:
        raise ValueError(
            f"Airflow Variable '{variable_name}' must be set to a non-empty JSON array."
        )
    return value


@dag(
    dag_id="dag_ingest_daily",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    on_failure_callback=on_failure_callback,
)
def dag_ingest_daily() -> None:
    """Run all ingestion connectors daily at 06:00 UTC.

    The DAG orchestrates the FRED, BLS, Reddit, Census, and OpenWeather
    connectors in parallel under a single TaskGroup. Each task reads its
    runtime inputs from Airflow Variables so ZIP codes, series ids, and other
    source targets can be updated without code changes. The DAG is limited to
    raw-data ingestion and does not perform transformations or scoring.
    """

    @task(task_id="task_fred")
    def task_fred() -> None:
        from src.connectors.fred_connector import FREDConnector

        series_ids = _get_optional_json_list("fred_series_ids")
        connector = FREDConnector()
        if series_ids is None:
            connector.fetch()
        else:
            connector.fetch(series_ids=series_ids)

    @task(task_id="task_bls")
    def task_bls() -> None:
        from src.connectors.bls_connector import BLSConnector

        series_ids = _get_optional_json_list("bls_series_ids")
        connector = BLSConnector()
        if series_ids is None:
            connector.fetch()
        else:
            connector.fetch(series_ids=series_ids)

    @task(task_id="task_reddit")
    def task_reddit() -> None:
        from src.connectors.reddit_connector import RedditConnector

        subreddits = _get_optional_json_list("reddit_subreddits")
        connector = RedditConnector()
        if subreddits is None:
            connector.fetch()
        else:
            connector.fetch(subreddits=subreddits)

    @task(task_id="task_census")
    def task_census() -> None:
        from src.connectors.census_connector import CensusConnector

        state_fips = _get_optional_json_list("census_state_fips")
        connector = CensusConnector()
        if state_fips is None:
            connector.fetch()
        else:
            connector.fetch(state_fips=state_fips)

    @task(task_id="task_openweather")
    def task_openweather() -> None:
        from src.connectors.openweather_connector import OpenWeatherConnector

        zip_codes = _get_required_json_list(_OPENWEATHER_ZIP_VARIABLE)
        connector = OpenWeatherConnector()
        connector.fetch(zip_codes=zip_codes)

    @task(task_id="task_rss_ingest")
    def task_rss_ingest() -> None:
        from src.rag.rss_ingest import ingest_rss_feeds

        n = ingest_rss_feeds()
        logger.info("RSS ingest complete — %d new articles", n)

    with TaskGroup(group_id="connectors"):
        task_fred()
        task_bls()
        task_reddit()
        task_census()
        task_openweather()
        task_rss_ingest()


dag = dag_ingest_daily()
