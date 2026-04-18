"""Tests for src/observability/ground_truth.py.

All tests use an in-memory DuckDB via ``GroundTruthLogger(":memory:")``.
No HTTP calls are made — no mocking required beyond the database.
Each test creates its own isolated logger instance.
"""

import json
from datetime import date

import pytest

from src.observability.ground_truth import GroundTruthLogger

# ---------------------------------------------------------------------------
# Shared test fixtures / constants
# ---------------------------------------------------------------------------

TODAY = date.today()
EVENT_DATE = date(2024, 3, 1)
CONFIRMED_DATE = date(2024, 3, 10)

# A realistic 14-feature snapshot matching the PulseIQ feature contract
FULL_SIGNAL_SNAPSHOT: dict = {
    "bls_jobless_claims_delta": 0.12,
    "bls_unemployment_rate": 4.2,
    "fred_delinquency_rate": 2.8,
    "fred_cpi_delta": 0.3,
    "fred_mortgage_rate_delta": -0.05,
    "census_poverty_baseline": 14.1,
    "census_median_income": 52_000.0,
    "trends_search_score": 68.0,
    "trends_search_delta": 5.0,
    "news_sentiment_score": 0.25,
    "news_article_count": 12,
    "openweather_severity_index": 0.0,
    "data_quality_score": 0.93,
    "stale_source_count": 0,
}

SHAP_VALUES: dict = {
    "bls_jobless_claims_delta": 0.35,
    "fred_delinquency_rate": 0.20,
    "census_poverty_baseline": 0.10,
}


# ---------------------------------------------------------------------------
# raw_signal_log tests
# ---------------------------------------------------------------------------


def test_log_raw_signal_inserts_row() -> None:
    """log_raw_signal() writes one row with the correct field values."""
    gl = GroundTruthLogger(":memory:")
    gl.log_raw_signal(
        source="bls",
        geo_id="26",
        run_date=TODAY,
        raw_value=248_000.0,
        processed_value=248_000.0,
        validation_status="valid",
        anomaly_flag=False,
    )

    rows = gl._conn.execute(
        "SELECT source, geo_id, validation_status, anomaly_flag FROM raw_signal_log"
    ).fetchall()

    assert len(rows) == 1
    source, geo_id, validation_status, anomaly_flag = rows[0]
    assert source == "bls"
    assert geo_id == "26"
    assert validation_status == "valid"
    assert anomaly_flag is False


def test_log_raw_signal_idempotent() -> None:
    """Writing the same (source, geo_id, run_date) twice produces one row."""
    gl = GroundTruthLogger(":memory:")
    for _ in range(2):
        gl.log_raw_signal("bls", "26", TODAY, 248_000.0, 248_000.0, "valid", False)

    count = gl._conn.execute("SELECT COUNT(*) FROM raw_signal_log").fetchone()[0]
    assert count == 1


def test_log_raw_signal_accepts_none_values() -> None:
    """raw_value and processed_value may be None (e.g. for rejected records)."""
    gl = GroundTruthLogger(":memory:")
    gl.log_raw_signal("bls", "US", TODAY, None, None, "rejected", False)

    row = gl._conn.execute(
        "SELECT raw_value, processed_value FROM raw_signal_log"
    ).fetchone()
    assert row[0] is None
    assert row[1] is None


# ---------------------------------------------------------------------------
# prediction_log tests
# ---------------------------------------------------------------------------


def test_log_prediction_stores_complete_signal_snapshot() -> None:
    """log_prediction() preserves all 14 feature keys in signal_snapshot via JSON round-trip."""
    gl = GroundTruthLogger(":memory:")
    gl.log_prediction(
        geo_id="26",
        run_date=TODAY,
        ess_score=72.4,
        confidence="high",
        shap_values=SHAP_VALUES,
        signal_snapshot=FULL_SIGNAL_SNAPSHOT,
    )

    row = gl._conn.execute(
        "SELECT signal_snapshot FROM prediction_log WHERE geo_id = '26'"
    ).fetchone()
    assert row is not None

    recovered: dict = json.loads(row[0])
    assert recovered == FULL_SIGNAL_SNAPSHOT
    assert len(recovered) == len(FULL_SIGNAL_SNAPSHOT)


def test_log_prediction_idempotent() -> None:
    """Writing the same (geo_id, run_date) twice produces one row."""
    gl = GroundTruthLogger(":memory:")
    for _ in range(2):
        gl.log_prediction("26", TODAY, 72.4, "high", SHAP_VALUES, FULL_SIGNAL_SNAPSHOT)

    count = gl._conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
    assert count == 1


def test_log_prediction_overwrite_updates_fields() -> None:
    """A second write for the same geo/run_date replaces the stored ess_score."""
    gl = GroundTruthLogger(":memory:")
    gl.log_prediction("26", TODAY, 72.4, "high", SHAP_VALUES, FULL_SIGNAL_SNAPSHOT)
    gl.log_prediction("26", TODAY, 85.0, "medium", SHAP_VALUES, FULL_SIGNAL_SNAPSHOT)

    row = gl._conn.execute(
        "SELECT ess_score, confidence FROM prediction_log"
    ).fetchone()
    assert row[0] == pytest.approx(85.0)
    assert row[1] == "medium"


def test_log_prediction_shap_values_round_trip() -> None:
    """shap_values dict is faithfully stored and recovered as JSON."""
    gl = GroundTruthLogger(":memory:")
    gl.log_prediction("26", TODAY, 72.4, "high", SHAP_VALUES, FULL_SIGNAL_SNAPSHOT)

    raw_json = gl._conn.execute(
        "SELECT shap_values FROM prediction_log"
    ).fetchone()[0]
    recovered = json.loads(raw_json)
    assert recovered == SHAP_VALUES


# ---------------------------------------------------------------------------
# ground_truth_events tests
# ---------------------------------------------------------------------------


def test_log_ground_truth_event_persists_correctly() -> None:
    """log_ground_truth_event() writes all six fields accurately."""
    gl = GroundTruthLogger(":memory:")
    gl.log_ground_truth_event(
        geo_id="26",
        event_date=EVENT_DATE,
        event_type="mass_layoff",
        event_source="BLS_WARN_ACT",
        severity="high",
        confirmed_date=CONFIRMED_DATE,
    )

    row = gl._conn.execute(
        """
        SELECT geo_id, event_date, event_type, event_source, severity, confirmed_date
        FROM ground_truth_events
        """
    ).fetchone()

    assert row is not None
    geo_id, event_date, event_type, event_source, severity, confirmed_date = row
    assert geo_id == "26"
    assert event_date == EVENT_DATE
    assert event_type == "mass_layoff"
    assert event_source == "BLS_WARN_ACT"
    assert severity == "high"
    assert confirmed_date == CONFIRMED_DATE


def test_log_ground_truth_event_idempotent() -> None:
    """Writing the same (geo_id, event_date, event_type, event_source) twice → 1 row."""
    gl = GroundTruthLogger(":memory:")
    for _ in range(2):
        gl.log_ground_truth_event(
            "26", EVENT_DATE, "mass_layoff", "BLS_WARN_ACT", "high", CONFIRMED_DATE
        )

    count = gl._conn.execute(
        "SELECT COUNT(*) FROM ground_truth_events"
    ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Structural test
# ---------------------------------------------------------------------------


def test_all_three_tables_exist_after_init() -> None:
    """GroundTruthLogger creates all three tables on initialisation."""
    gl = GroundTruthLogger(":memory:")

    table_names = {
        row[0]
        for row in gl._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }

    assert "raw_signal_log" in table_names
    assert "prediction_log" in table_names
    assert "ground_truth_events" in table_names
