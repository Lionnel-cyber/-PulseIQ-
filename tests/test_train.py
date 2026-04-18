"""Tests for src/models/train.py.

All tests use a temp DuckDB file (via pytest's ``tmp_path`` fixture) and a
mocked ``mlflow`` module so no running MLflow server is required.

``monkeypatch.chdir(tmp_path)`` redirects the ``models/{run_id}/`` artifact
writes to the temporary directory — the real project ``models/`` directory is
never touched.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from src.models.train import FEATURE_COLS, FEATURE_VERSION, train

# ---------------------------------------------------------------------------
# Synthetic data helpers — mirror the pattern from test_features.py
# ---------------------------------------------------------------------------

_BASE_DATE = date(2024, 1, 1)

_MART_ROW_DEFAULTS: dict = {
    "geo_level":                "city",
    "geo_name":                 "Test City",
    "tier1_score":              0.10,
    "tier2_score":              0.05,
    "tier3_score":              0.02,
    "claims_yoy_pct":           0.0,
    "county_unemployment_rate": 4.0,
    "cpi_monthly_delta":        0.3,
    "delinquency_rate":         2.5,
    "median_income_index":      52_000.0,
    "reddit_negativity_score":  0.2,
    "post_volume_delta":        0.1,
    "distress_keyword_freq":    5,
    "poverty_rate":             14.0,
    "income_quartile":          2,
    "housing_cost_burden":      0.30,
    "extreme_weather_events_7d": 0,
    "weather_stress_index":     0.10,
    "ess_score":                65.0,
    "data_quality_score":       0.80,
    "granularity_warning":      True,
    "data_granularity_note":    "test note",
    "stale_sources":            "",
    "anomaly_flags":            "",
}

_CREATE_TABLE_SQL = """
CREATE TABLE mart_economic_stress (
    geo_id                   VARCHAR,
    geo_level                VARCHAR,
    geo_name                 VARCHAR,
    run_date                 DATE,
    date                     DATE,
    tier1_score              DOUBLE,
    tier2_score              DOUBLE,
    tier3_score              DOUBLE,
    jobless_claims_delta     DOUBLE,
    claims_yoy_pct           DOUBLE,
    county_unemployment_rate DOUBLE,
    cpi_monthly_delta        DOUBLE,
    delinquency_rate         DOUBLE,
    median_income_index      DOUBLE,
    reddit_negativity_score  DOUBLE,
    post_volume_delta        DOUBLE,
    distress_keyword_freq    INTEGER,
    poverty_rate             DOUBLE,
    income_quartile          INTEGER,
    housing_cost_burden      DOUBLE,
    extreme_weather_events_7d INTEGER,
    weather_stress_index     DOUBLE,
    ess_score                DOUBLE,
    data_quality_score       DOUBLE,
    granularity_warning      BOOLEAN,
    data_granularity_note    VARCHAR,
    stale_sources            VARCHAR,
    anomaly_flags            VARCHAR
)
"""

_INSERT_SQL = """
INSERT INTO mart_economic_stress VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?
)
"""


def _make_rows(geo_id: str, n_rows: int, quartile_offset: int = 0) -> list[dict]:
    """Build mart row dicts for a single geo_id over n_rows consecutive dates.

    Args:
        geo_id: Geography identifier.
        n_rows: Number of rows to generate.
        quartile_offset: Added to the cycling quartile index so multiple geos
            can be interleaved with different starting quartiles.

    Returns:
        List of row dicts with all mart columns populated.
    """
    rows = []
    for i in range(n_rows):
        row = dict(_MART_ROW_DEFAULTS)
        row["geo_id"] = geo_id
        row["run_date"] = _BASE_DATE + timedelta(days=i)
        row["date"] = _BASE_DATE + timedelta(days=i)
        # Vary jobless_claims_delta so the label is not constant
        row["jobless_claims_delta"] = 0.1 + i * 0.01
        # Cycle income_quartile [1, 2, 3, 4] so stratified split has ≥2 samples each
        row["income_quartile"] = ((i + quartile_offset) % 4) + 1
        rows.append(row)
    return rows


def _insert_rows(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    for row in rows:
        conn.execute(_INSERT_SQL, [
            row["geo_id"], row["geo_level"], row["geo_name"],
            row["run_date"], row["date"],
            row["tier1_score"], row["tier2_score"], row["tier3_score"],
            row["jobless_claims_delta"], row["claims_yoy_pct"],
            row["county_unemployment_rate"], row["cpi_monthly_delta"],
            row["delinquency_rate"], row["median_income_index"],
            row["reddit_negativity_score"], row["post_volume_delta"],
            row["distress_keyword_freq"], row["poverty_rate"],
            row["income_quartile"], row["housing_cost_burden"],
            row["extreme_weather_events_7d"], row["weather_stress_index"],
            row["ess_score"], row["data_quality_score"],
            row["granularity_warning"], row["data_granularity_note"],
            row["stale_sources"], row["anomaly_flags"],
        ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_db(tmp_path) -> str:
    """Temp DuckDB file with 50 mart rows: 2 geo_ids × 25 rows each.

    ``income_quartile`` cycles [1, 2, 3, 4] with a per-geo offset so all
    four classes appear in both geos — enough for the stratified split.
    All rows have ``data_quality_score=0.80`` (above the 0.7 filter threshold).
    """
    db_path = str(tmp_path / "train_test.db")
    conn = duckdb.connect(db_path)
    conn.execute(_CREATE_TABLE_SQL)
    _insert_rows(conn, _make_rows("A-MI", 25, quartile_offset=0))
    _insert_rows(conn, _make_rows("B-OH", 25, quartile_offset=2))
    conn.close()
    return db_path


@pytest.fixture
def mock_mlflow():
    """Patch ``src.models.train.mlflow`` with a MagicMock.

    ``start_run()`` acts as a context manager returning a run with
    ``run_id = "mock-run-id-abc"``.
    """
    run_mock = MagicMock()
    run_mock.info.run_id = "mock-run-id-abc"

    cm_mock = MagicMock()
    cm_mock.__enter__.return_value = run_mock
    cm_mock.__exit__.return_value = False

    with patch("src.models.train.mlflow") as mlflow_mock:
        mlflow_mock.start_run.return_value = cm_mock
        yield mlflow_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_returns_string_run_id(
    train_db: str, mock_mlflow: MagicMock, monkeypatch, tmp_path
) -> None:
    """train() returns a non-empty string run_id."""
    monkeypatch.chdir(tmp_path)
    run_id = train(train_db, experiment_name="test-exp")
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_train_creates_model_file(
    train_db: str, mock_mlflow: MagicMock, monkeypatch, tmp_path
) -> None:
    """train() writes model.pkl to models/{run_id}/."""
    monkeypatch.chdir(tmp_path)
    run_id = train(train_db, experiment_name="test-exp")
    model_path = tmp_path / "models" / run_id / "model.pkl"
    assert model_path.exists(), f"Expected model file at {model_path}"


def test_train_creates_feature_version_file(
    train_db: str, mock_mlflow: MagicMock, monkeypatch, tmp_path
) -> None:
    """train() writes feature_version.json to models/{run_id}/."""
    monkeypatch.chdir(tmp_path)
    run_id = train(train_db, experiment_name="test-exp")
    version_path = tmp_path / "models" / run_id / "feature_version.json"
    assert version_path.exists(), f"Expected feature_version.json at {version_path}"


def test_train_feature_version_matches_constant(
    train_db: str, mock_mlflow: MagicMock, monkeypatch, tmp_path
) -> None:
    """The feature_version in the saved JSON matches the module constant."""
    monkeypatch.chdir(tmp_path)
    run_id = train(train_db, experiment_name="test-exp")
    version_path = tmp_path / "models" / run_id / "feature_version.json"
    with open(version_path) as f:
        saved = json.load(f)
    assert saved["feature_version"] == FEATURE_VERSION
    assert saved["feature_cols"] == FEATURE_COLS


def test_train_logs_mlflow_params(
    train_db: str, mock_mlflow: MagicMock, monkeypatch, tmp_path
) -> None:
    """train() calls mlflow.log_params() at least once."""
    monkeypatch.chdir(tmp_path)
    train(train_db, experiment_name="test-exp")
    assert mock_mlflow.log_params.called, "mlflow.log_params was not called"


def test_train_logs_mlflow_metrics(
    train_db: str, mock_mlflow: MagicMock, monkeypatch, tmp_path
) -> None:
    """train() calls mlflow.log_metrics() with rmse, mae, and r2 keys."""
    monkeypatch.chdir(tmp_path)
    train(train_db, experiment_name="test-exp")
    assert mock_mlflow.log_metrics.called, "mlflow.log_metrics was not called"
    metrics_call_kwargs = mock_mlflow.log_metrics.call_args[0][0]
    assert "rmse" in metrics_call_kwargs
    assert "mae" in metrics_call_kwargs
    assert "r2" in metrics_call_kwargs
