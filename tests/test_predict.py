"""Tests for src/models/predict.py.

All tests:
- Use ``monkeypatch.chdir(tmp_path)`` so model artifacts land under tmp_path.
- Patch ``src.models.predict.explain`` so SHAP does not run (tested separately
  in test_explainer.py).
- Use a tiny real XGBoost (5 estimators, max_depth=2) so ``model.predict()``
  produces genuine numeric output.
"""

from __future__ import annotations

import json
import pickle
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.contracts import Prediction
from src.models.predict import score_all_geos
from src.models.train import FEATURE_COLS, FEATURE_VERSION

# ---------------------------------------------------------------------------
# Synthetic mart data helpers (mirrors test_features.py pattern)
# ---------------------------------------------------------------------------

_BASE_DATE = date(2024, 1, 1)
_N_GEOS = 3

_MART_ROW_DEFAULTS: dict = {
    "geo_level":                "city",
    "geo_name":                 "Test City",
    "tier1_score":              15.0,
    "tier2_score":              10.0,
    "tier3_score":              5.0,
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
    "ess_score":                50.0,
    "data_quality_score":       0.80,
    "granularity_warning":      False,
    "data_granularity_note":    "test note",
    "stale_sources":            "",
    "anomaly_flags":            "",
}

_CREATE_MART_SQL = """
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

_INSERT_MART_SQL = """
INSERT INTO mart_economic_stress VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?
)
"""


def _insert_mart_row(conn: duckdb.DuckDBPyConnection, row: dict) -> None:
    conn.execute(_INSERT_MART_SQL, [
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


def _make_mart_row(geo_id: str, days_offset: int = 0) -> dict:
    row = dict(_MART_ROW_DEFAULTS)
    row["geo_id"] = geo_id
    row["run_date"] = _BASE_DATE + timedelta(days=days_offset)
    row["date"] = _BASE_DATE + timedelta(days=days_offset)
    row["jobless_claims_delta"] = 0.1 + days_offset * 0.01
    return row


# ---------------------------------------------------------------------------
# Fixture: model artifact + DuckDB mart
# ---------------------------------------------------------------------------

_MOCK_RUN_ID = "test-predict-run-001"


@pytest.fixture
def predict_env(tmp_path, monkeypatch):
    """Set up a scoring environment with a tiny XGBoost model and a DuckDB mart.

    Layout under ``tmp_path``:
        models/test-predict-run-001/model.pkl
        models/test-predict-run-001/feature_version.json
        test.db  (mart_economic_stress with 3 geo rows)

    Returns:
        (db_path: str, run_id: str, n_geos: int)
    """
    monkeypatch.chdir(tmp_path)

    # ── Tiny XGBoost model trained on FEATURE_COLS ────────────────────────────
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.random((30, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y_train = rng.random(30) * 100
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    model_dir = tmp_path / "models" / _MOCK_RUN_ID
    model_dir.mkdir(parents=True)
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / "feature_version.json", "w") as f:
        json.dump({"feature_version": FEATURE_VERSION, "feature_cols": FEATURE_COLS}, f)

    # ── DuckDB mart with 3 geos ───────────────────────────────────────────────
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    conn.execute(_CREATE_MART_SQL)
    geo_ids = ["A-MI", "B-OH", "C-IL"]
    for gid in geo_ids:
        _insert_mart_row(conn, _make_mart_row(gid, days_offset=0))
    conn.close()

    return db_path, _MOCK_RUN_ID, len(geo_ids)


def _mock_shap(n: int) -> list[dict[str, float]]:
    """Return n fixed SHAP dicts, one per geo."""
    return [{col: 0.1 for col in FEATURE_COLS}] * n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("src.models.predict.explain")
def test_score_all_geos_returns_list_of_predictions(
    mock_explain, predict_env
) -> None:
    """score_all_geos() returns a list of Prediction objects, one per geo."""
    db_path, run_id, n_geos = predict_env
    mock_explain.side_effect = lambda m, X: _mock_shap(len(X))

    results = score_all_geos(db_path, run_id)

    assert isinstance(results, list)
    assert len(results) == n_geos
    assert all(isinstance(p, Prediction) for p in results)


@patch("src.models.predict.explain")
def test_score_all_geos_ess_scores_in_range(mock_explain, predict_env) -> None:
    """All returned ESS scores are within [0, 100]."""
    db_path, run_id, n_geos = predict_env
    mock_explain.side_effect = lambda m, X: _mock_shap(len(X))

    results = score_all_geos(db_path, run_id)

    for pred in results:
        assert 0.0 <= pred.ess_score <= 100.0, (
            f"ess_score={pred.ess_score} is out of [0, 100] for geo {pred.geo_id}"
        )


@patch("src.models.predict.explain")
def test_score_all_geos_prediction_fields_present(
    mock_explain, predict_env
) -> None:
    """Core identity and provenance fields are populated on every Prediction."""
    db_path, run_id, n_geos = predict_env
    mock_explain.side_effect = lambda m, X: _mock_shap(len(X))

    results = score_all_geos(db_path, run_id)

    for pred in results:
        assert pred.geo_id, "geo_id must be non-empty"
        assert isinstance(pred.run_date, date), "run_date must be a date"
        assert pred.model_version == run_id, "model_version must equal model_run_id"
        assert isinstance(pred.shap_values, dict), "shap_values must be a dict"
        assert isinstance(pred.stale_sources, list)
        assert isinstance(pred.anomaly_flags, list)


@patch("src.models.predict.explain")
def test_score_all_geos_writes_ess_scores_table(
    mock_explain, predict_env
) -> None:
    """score_all_geos() creates the ess_scores table and writes one row per geo."""
    db_path, run_id, n_geos = predict_env
    mock_explain.side_effect = lambda m, X: _mock_shap(len(X))

    score_all_geos(db_path, run_id)

    conn = duckdb.connect(db_path, read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM ess_scores").fetchone()[0]
    conn.close()

    assert count == n_geos, (
        f"Expected {n_geos} rows in ess_scores, got {count}"
    )


@patch("src.models.predict.explain")
def test_score_all_geos_idempotent(mock_explain, predict_env) -> None:
    """Running score_all_geos() twice for the same day produces no duplicate rows."""
    db_path, run_id, n_geos = predict_env
    mock_explain.side_effect = lambda m, X: _mock_shap(len(X))

    score_all_geos(db_path, run_id)
    score_all_geos(db_path, run_id)   # second run — same (geo_id, run_date) keys

    conn = duckdb.connect(db_path, read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM ess_scores").fetchone()[0]
    conn.close()

    assert count == n_geos, (
        f"Expected {n_geos} rows after two runs (idempotent), got {count}"
    )
