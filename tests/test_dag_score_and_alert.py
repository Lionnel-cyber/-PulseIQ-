"""Tests for dags/dag_score_and_alert.py and its src/ dependencies.

Structure:
  1. DAG structure — task ordering via direct import of the dag object
  2. task_calibrate — apply_calibration_to_today handles missing pkl gracefully
  3. task_alerts   — AlertSuppressor respects cooldown and min-delta rules
  4. task_monitor  — run_monitor_and_log logs CRITICAL for "immediate" recommendation

All src/ imports inside tests are patched or given isolated tmp_path databases
so no real model files, ChromaDB, or network calls are needed.
"""

from __future__ import annotations

import os
import tempfile

os.environ["AIRFLOW_HOME"] = tempfile.mkdtemp()
os.environ["AIRFLOW__LOGGING__BASE_LOG_FOLDER"] = tempfile.mkdtemp()

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ESS_SCORES_DDL = """
CREATE TABLE IF NOT EXISTS ess_scores (
    geo_id              VARCHAR  NOT NULL,
    geo_name            VARCHAR  NOT NULL,
    geo_level           VARCHAR  NOT NULL,
    run_date            DATE     NOT NULL,
    ess_score           DOUBLE   NOT NULL,
    score_band          VARCHAR  NOT NULL,
    delta_7d            DOUBLE,
    delta_30d           DOUBLE,
    confidence          VARCHAR  NOT NULL,
    early_warning       BOOLEAN  NOT NULL,
    missing_sources     VARCHAR  NOT NULL,
    stale_sources       VARCHAR  NOT NULL,
    anomaly_flags       VARCHAR  NOT NULL,
    granularity_warning BOOLEAN  NOT NULL,
    model_version       VARCHAR  NOT NULL,
    feature_version     VARCHAR  NOT NULL,
    calibrated          BOOLEAN  NOT NULL,
    tier1_score         DOUBLE   NOT NULL,
    tier2_score         DOUBLE   NOT NULL,
    tier3_score         DOUBLE   NOT NULL,
    shap_values         VARCHAR  NOT NULL,
    PRIMARY KEY (geo_id, run_date)
)
"""

_SCORE_ROW = {
    "geo_id": "Detroit-MI",
    "geo_name": "Detroit",
    "geo_level": "city",
    "ess_score": 78.0,
    "score_band": "high",
    "delta_7d": 8.0,
    "delta_30d": 12.0,
    "confidence": "medium",
    "early_warning": False,
    "missing_sources": "[]",
    "stale_sources": "[]",
    "anomaly_flags": "[]",
    "granularity_warning": False,
    "model_version": "run-abc",
    "feature_version": "v1",
    "calibrated": False,
    "tier1_score": 35.0,
    "tier2_score": 22.0,
    "tier3_score": 11.0,
    "shap_values": json.dumps({"bls_jobless_claims_delta": 2.5, "fred_delinquency_rate": 1.8}),
}


def _seed_ess_scores(db_path: str, run_date: date, row: dict | None = None) -> None:
    r = {**(row or _SCORE_ROW)}
    with duckdb.connect(db_path) as conn:
        conn.execute(_ESS_SCORES_DDL)
        conn.execute(
            "INSERT OR REPLACE INTO ess_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                r["geo_id"], r["geo_name"], r["geo_level"], run_date,
                r["ess_score"], r["score_band"],
                r.get("delta_7d"), r.get("delta_30d"),
                r["confidence"], r["early_warning"],
                r["missing_sources"], r["stale_sources"], r["anomaly_flags"],
                r["granularity_warning"], r["model_version"], r["feature_version"],
                r["calibrated"],
                r["tier1_score"], r["tier2_score"], r["tier3_score"],
                r["shap_values"],
            ],
        )


# ===========================================================================
# 1. DAG structure — task ordering
# ===========================================================================


def test_dag_task_ids_and_ordering() -> None:
    """DAG must contain all five task IDs in the correct dependency chain."""
    from dags.dag_score_and_alert import dag as score_dag

    task_ids = {t.task_id for t in score_dag.tasks}

    assert "sensor_transform_complete" in task_ids, "sensor missing"
    assert "task_score" in task_ids, "task_score missing"
    assert "task_calibrate" in task_ids, "task_calibrate missing"
    assert "task_alerts" in task_ids, "task_alerts missing"
    assert "task_monitor" in task_ids, "task_monitor missing"

    # sensor → task_score
    sensor = score_dag.get_task("sensor_transform_complete")
    assert "task_score" in sensor.downstream_task_ids, (
        "sensor_transform_complete must be upstream of task_score"
    )

    # task_score → task_calibrate
    t_score = score_dag.get_task("task_score")
    assert "task_calibrate" in t_score.downstream_task_ids, (
        "task_score must be upstream of task_calibrate"
    )

    # task_calibrate → task_alerts
    t_cal = score_dag.get_task("task_calibrate")
    assert "task_alerts" in t_cal.downstream_task_ids, (
        "task_calibrate must be upstream of task_alerts"
    )

    # task_alerts → task_monitor
    t_alerts = score_dag.get_task("task_alerts")
    assert "task_monitor" in t_alerts.downstream_task_ids, (
        "task_alerts must be upstream of task_monitor"
    )


def test_dag_schedule_and_defaults() -> None:
    """DAG must use 10:00 UTC schedule and correct retry defaults."""
    from dags.dag_score_and_alert import dag as score_dag

    assert score_dag.schedule_interval == "0 10 * * *"
    assert score_dag.default_args["retries"] == 2
    assert score_dag.default_args["retry_delay"].total_seconds() == 300  # 5 min


def test_sensor_targets_dbt_test() -> None:
    """Sensor must wait on dag_transform_daily / dbt_test with 2h delta."""
    from datetime import timedelta

    from airflow.sensors.external_task import ExternalTaskSensor
    from dags.dag_score_and_alert import dag as score_dag

    sensor = score_dag.get_task("sensor_transform_complete")
    assert isinstance(sensor, ExternalTaskSensor)
    assert sensor.external_dag_id == "dag_transform_daily"
    assert sensor.external_task_id == "dbt_test"
    assert sensor.execution_delta == timedelta(hours=2)


# ===========================================================================
# 2. task_calibrate — graceful on missing calibration.pkl
# ===========================================================================


def test_apply_calibration_to_today_missing_pkl(tmp_path: Path) -> None:
    """apply_calibration_to_today must succeed and mark rows uncalibrated."""
    from src.models.calibration import apply_calibration_to_today

    db_path = str(tmp_path / "pulseiq.db")
    today = date.today()
    _seed_ess_scores(db_path, run_date=today)

    fake_pkl_path = tmp_path / "no_such_calibration.pkl"
    with patch("src.models.calibration._CALIB_PATH", fake_pkl_path):
        count = apply_calibration_to_today(db_path=db_path)

    assert count == 1

    with duckdb.connect(db_path) as conn:
        row = conn.execute(
            "SELECT calibrated, confidence FROM ess_scores WHERE geo_id='Detroit-MI'"
        ).fetchone()

    assert row is not None
    assert row[0] is False, "calibrated must be False when pkl is missing"
    assert row[1] == "low", "confidence must be 'low' when pkl is missing"


def test_apply_calibration_to_today_empty_table(tmp_path: Path) -> None:
    """apply_calibration_to_today must return 0 gracefully on empty table."""
    from src.models.calibration import apply_calibration_to_today

    db_path = str(tmp_path / "pulseiq.db")
    with duckdb.connect(db_path) as conn:
        conn.execute(_ESS_SCORES_DDL)  # empty table, no rows

    fake_pkl_path = tmp_path / "no_such.pkl"
    with patch("src.models.calibration._CALIB_PATH", fake_pkl_path):
        count = apply_calibration_to_today(db_path=db_path)

    assert count == 0


def test_apply_calibration_to_today_with_fitted_calibrator(tmp_path: Path) -> None:
    """apply_calibration_to_today must set calibrated=True when pkl is fitted."""
    import numpy as np

    from src.models.calibration import PulseIQCalibrator, apply_calibration_to_today

    db_path = str(tmp_path / "pulseiq.db")
    today = date.today()
    _seed_ess_scores(db_path, run_date=today)

    # Fit and save a calibrator at a temp path
    pkl_path = tmp_path / "calibration.pkl"
    cal = PulseIQCalibrator()
    predicted = list(np.linspace(0, 100, 70))
    actual = [p * 0.95 for p in predicted]
    cal.fit(predicted, actual, save_path=pkl_path)

    with patch("src.models.calibration._CALIB_PATH", pkl_path):
        count = apply_calibration_to_today(db_path=db_path)

    assert count == 1

    with duckdb.connect(db_path) as conn:
        row = conn.execute(
            "SELECT calibrated FROM ess_scores WHERE geo_id='Detroit-MI'"
        ).fetchone()

    assert row[0] is True, "calibrated must be True after fitting"


# ===========================================================================
# 3. task_alerts — suppression rules
# ===========================================================================


def test_alert_suppressor_allows_first_alert(tmp_path: Path) -> None:
    """First alert with sufficient delta must not be suppressed."""
    from src.observability.alerts import AlertSuppressor

    db_path = str(tmp_path / "pulseiq.db")
    suppressor = AlertSuppressor(db_path=db_path)

    # No prior record, delta=15 >> MIN_SCORE_DELTA
    result = suppressor.is_suppressed("Detroit-MI", "threshold_breach", 80.0, 65.0)
    assert result is False


def test_alert_suppressor_blocks_within_cooldown(tmp_path: Path) -> None:
    """Second alert within cooldown window must be suppressed."""
    from src.observability.alerts import AlertSuppressor

    db_path = str(tmp_path / "pulseiq.db")
    suppressor = AlertSuppressor(db_path=db_path)

    # Record an alert now
    suppressor.record_alert(
        "Detroit-MI", "threshold_breach", 78.0, datetime.now(tz=timezone.utc)
    )

    # Same geo+type, large delta — but still within 3-day cooldown
    result = suppressor.is_suppressed("Detroit-MI", "threshold_breach", 82.0, 65.0)
    assert result is True, "Alert should be suppressed during cooldown window"


def test_alert_suppressor_blocks_small_delta(tmp_path: Path) -> None:
    """Score delta < 5 pts must always be suppressed regardless of cooldown."""
    from src.observability.alerts import AlertSuppressor

    db_path = str(tmp_path / "pulseiq.db")
    suppressor = AlertSuppressor(db_path=db_path)

    # Delta = 2 < MIN_SCORE_DELTA (5)
    result = suppressor.is_suppressed("Detroit-MI", "threshold_breach", 77.0, 75.0)
    assert result is True, "Alert with delta < 5 must be suppressed"


def test_alert_suppressor_different_type_not_suppressed(tmp_path: Path) -> None:
    """Prior alert of one type must not suppress a different type."""
    from src.observability.alerts import AlertSuppressor

    db_path = str(tmp_path / "pulseiq.db")
    suppressor = AlertSuppressor(db_path=db_path)

    suppressor.record_alert(
        "Detroit-MI", "threshold_breach", 78.0, datetime.now(tz=timezone.utc)
    )

    # Different alert_type with sufficient delta
    result = suppressor.is_suppressed("Detroit-MI", "rapid_rise", 90.0, 70.0)
    assert result is False, "Different alert_type must not be suppressed by prior record"


def test_fire_alerts_respects_suppression(tmp_path: Path) -> None:
    """fire_alerts_for_today must not POST when alert is suppressed."""
    from src.observability.alerts import AlertSuppressor, fire_alerts_for_today

    db_path = str(tmp_path / "pulseiq.db")
    today = date.today()
    _seed_ess_scores(db_path, run_date=today)  # score=78, delta_7d=8

    # Pre-record the alert so it's within cooldown
    suppressor = AlertSuppressor(db_path=db_path)
    suppressor.record_alert(
        "Detroit-MI", "threshold_breach", 75.0, datetime.now(tz=timezone.utc)
    )

    with patch("src.observability.alerts.requests.post") as mock_post:
        fired, suppressed = fire_alerts_for_today(
            db_path=db_path,
            threshold=75.0,
            webhook_url="https://mock.example.com/hook",
            webhook_secret="secret",
        )

    assert fired == 0
    assert suppressed == 1
    mock_post.assert_not_called()


# ===========================================================================
# 4. task_monitor — critical log for "immediate" recommendation
# ===========================================================================


def test_run_monitor_and_log_logs_critical_for_immediate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """run_monitor_and_log must emit CRITICAL when recommendation is 'immediate'."""
    from src.models.monitor import run_monitor_and_log

    db_path = str(tmp_path / "pulseiq.db")

    mock_monitor = MagicMock()
    mock_monitor.feature_drift.return_value = {
        "bls_jobless_claims_delta": {"psi": 0.25, "status": "drift_detected"},
        "overall_status": "drift_detected",
        "insufficient_data": False,
    }
    mock_monitor.score_distribution_drift.return_value = {
        "mean_shift": 3.0, "std_shift": 0.5,
        "drift_detected": False, "insufficient_data": False,
    }
    mock_monitor.missing_source_drift.return_value = {}
    mock_monitor.retraining_recommendation.return_value = {
        "recommendation": "immediate",
        "evidence": ["PSI for bls_jobless_claims_delta = 0.25 (drift_detected)"],
        "triggered_by": ["feature_drift"],
    }

    with patch("src.models.monitor.PulseIQMonitor", return_value=mock_monitor):
        with caplog.at_level(logging.CRITICAL, logger="src.models.monitor"):
            report = run_monitor_and_log(db_path=db_path)

    assert report["recommendation"] == "immediate"
    critical_msgs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical_msgs) >= 1, "Expected at least one CRITICAL log record"
    assert any("immediate" in r.message.lower() for r in critical_msgs)


def test_run_monitor_and_log_writes_to_monitor_log(tmp_path: Path) -> None:
    """run_monitor_and_log must persist the report to the monitor_log table."""
    from src.models.monitor import run_monitor_and_log

    db_path = str(tmp_path / "pulseiq.db")

    mock_monitor = MagicMock()
    mock_monitor.feature_drift.return_value = {"overall_status": "stable", "insufficient_data": True}
    mock_monitor.score_distribution_drift.return_value = {"mean_shift": 0.0, "insufficient_data": True}
    mock_monitor.missing_source_drift.return_value = {}
    mock_monitor.retraining_recommendation.return_value = {
        "recommendation": "no_action",
        "evidence": [],
        "triggered_by": [],
    }

    with patch("src.models.monitor.PulseIQMonitor", return_value=mock_monitor):
        run_monitor_and_log(db_path=db_path)

    with duckdb.connect(db_path) as conn:
        row = conn.execute(
            "SELECT recommendation FROM monitor_log"
        ).fetchone()

    assert row is not None, "monitor_log must have a row after run_monitor_and_log"
    assert row[0] == "no_action"


def test_run_monitor_and_log_no_action_no_critical(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """run_monitor_and_log must NOT log CRITICAL for 'no_action' recommendation."""
    from src.models.monitor import run_monitor_and_log

    db_path = str(tmp_path / "pulseiq.db")

    mock_monitor = MagicMock()
    mock_monitor.feature_drift.return_value = {"overall_status": "stable", "insufficient_data": True}
    mock_monitor.score_distribution_drift.return_value = {"mean_shift": 0.0, "insufficient_data": True}
    mock_monitor.missing_source_drift.return_value = {}
    mock_monitor.retraining_recommendation.return_value = {
        "recommendation": "no_action",
        "evidence": [],
        "triggered_by": [],
    }

    with patch("src.models.monitor.PulseIQMonitor", return_value=mock_monitor):
        with caplog.at_level(logging.DEBUG, logger="src.models.monitor"):
            run_monitor_and_log(db_path=db_path)

    critical_msgs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical_msgs) == 0, "no_action must not produce CRITICAL log records"
