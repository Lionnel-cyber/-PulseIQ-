"""Tests for src/models/evaluate.py.

Uses a temporary DuckDB with synthetic ``ess_scores`` and
``ground_truth_events`` tables. No model loading, no SHAP, no MLflow.

Fixture layout
--------------
- 3 geos × 5 consecutive run_dates starting 2024-01-01 → 15 rows total
  (produces 12 consecutive pairs — enough for the ≥10 benchmark minimum)
- Scores vary so consecutive pairs have non-zero errors
- 1 confirmed ground_truth_event for geo "A-MI" on 2024-01-03 (day 3)
- 1 row with ess_score=80 on geo "B-OH" on 2024-01-05 (alert, no event → FP)
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import duckdb
import pytest

from src.models.evaluate import PulseIQEvaluator

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_GEO_IDS = ["A-MI", "B-OH", "C-IL"]
_BASE_DATE = date(2024, 1, 1)
_N_DATES = 5

_CREATE_ESS_SCORES_SQL = """
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

_CREATE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS ground_truth_events (
    geo_id          VARCHAR NOT NULL,
    event_date      DATE    NOT NULL,
    event_type      VARCHAR NOT NULL,
    event_source    VARCHAR NOT NULL,
    severity        VARCHAR NOT NULL,
    confirmed_date  DATE    NOT NULL,
    PRIMARY KEY (geo_id, event_date, event_type, event_source)
)
"""

# SHAP dict with 3 features; feat_a has the largest |value|
_SHAP_JSON = json.dumps({"feat_a": 2.5, "feat_b": -0.8, "feat_c": 0.3})


def _make_ess_rows() -> list[dict]:
    """3 geos × 5 dates, varying ess_score, one alert row with ess_score=80."""
    rows = []
    for geo_id in _GEO_IDS:
        for i in range(_N_DATES):
            run_date = _BASE_DATE + timedelta(days=i)
            # Vary score so consecutive pairs have non-zero errors
            ess_score = 50.0 + i * 5.0 + (3.0 if geo_id == "B-OH" else 0.0)
            # Give B-OH day 4 (index 4) an ess_score of 80 → FP candidate
            if geo_id == "B-OH" and i == 4:
                ess_score = 80.0
            # Give A-MI day 2 (index 2) an early_warning
            early_warning = geo_id == "A-MI" and i == 2
            rows.append(
                {
                    "geo_id": geo_id,
                    "geo_name": f"{geo_id} City",
                    "geo_level": "city",
                    "run_date": run_date,
                    "ess_score": ess_score,
                    "score_band": "elevated",
                    "delta_7d": None,
                    "delta_30d": None,
                    "confidence": "medium",
                    "early_warning": early_warning,
                    "missing_sources": "[]",
                    "stale_sources": "[]",
                    "anomaly_flags": "[]",
                    "granularity_warning": False,
                    "model_version": "v1",
                    "feature_version": "abc",
                    "calibrated": True,
                    "tier1_score": 20.0,
                    "tier2_score": 15.0,
                    "tier3_score": 10.0,
                    "shap_values": _SHAP_JSON,
                }
            )
    return rows


@pytest.fixture
def eval_db(tmp_path):
    """Return a (db_path, PulseIQEvaluator) with pre-populated synthetic tables."""
    db_path = str(tmp_path / "eval_test.db")
    conn = duckdb.connect(db_path)

    conn.execute(_CREATE_ESS_SCORES_SQL)
    conn.execute(_CREATE_EVENTS_SQL)

    for row in _make_ess_rows():
        conn.execute(
            """
            INSERT INTO ess_scores VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                row["geo_id"], row["geo_name"], row["geo_level"],
                row["run_date"], row["ess_score"], row["score_band"],
                row["delta_7d"], row["delta_30d"], row["confidence"],
                row["early_warning"], row["missing_sources"],
                row["stale_sources"], row["anomaly_flags"],
                row["granularity_warning"], row["model_version"],
                row["feature_version"], row["calibrated"],
                row["tier1_score"], row["tier2_score"], row["tier3_score"],
                row["shap_values"],
            ],
        )

    # 1 confirmed event for A-MI on day 3 (2024-01-03) — within ±14 days of day 2
    conn.execute(
        "INSERT INTO ground_truth_events VALUES (?, ?, ?, ?, ?, ?)",
        ["A-MI", date(2024, 1, 3), "mass_layoff", "BLS_WARN_ACT", "high", date(2024, 1, 10)],
    )
    conn.close()

    ev = PulseIQEvaluator(db_path)
    yield ev
    ev._conn.close()


# ---------------------------------------------------------------------------
# benchmark tests
# ---------------------------------------------------------------------------


def test_benchmark_returns_required_keys(eval_db) -> None:
    """benchmark() returns all four required keys."""
    result = eval_db.benchmark()
    assert "model_rmse" in result
    assert "baseline_rmse" in result
    assert "improvement_pct" in result
    assert "verdict" in result


def test_benchmark_improvement_pct_is_float(eval_db) -> None:
    """improvement_pct is a plain float (may be negative)."""
    result = eval_db.benchmark()
    assert isinstance(result["improvement_pct"], float)


def test_benchmark_verdict_is_string(eval_db) -> None:
    """verdict is one of the two allowed string values."""
    result = eval_db.benchmark()
    assert result["verdict"] in {"PASS", "FAIL — do not ship"}


# ---------------------------------------------------------------------------
# threshold_evaluation tests
# ---------------------------------------------------------------------------


def test_threshold_evaluation_returns_all_5_thresholds(eval_db) -> None:
    """threshold_evaluation() returns exactly the 5 default thresholds."""
    result = eval_db.threshold_evaluation()
    assert set(result.keys()) == {60, 70, 75, 80, 85}
    for T in [60, 70, 75, 80, 85]:
        assert "precision" in result[T]
        assert "recall" in result[T]
        assert "alert_count" in result[T]


def test_threshold_evaluation_alert_count_decreases_with_threshold(eval_db) -> None:
    """Higher threshold produces fewer or equal alert_count (monotone)."""
    result = eval_db.threshold_evaluation()
    thresholds = sorted(result.keys())
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        assert result[lo]["alert_count"] >= result[hi]["alert_count"], (
            f"alert_count at T={lo} ({result[lo]['alert_count']}) < "
            f"alert_count at T={hi} ({result[hi]['alert_count']})"
        )


# ---------------------------------------------------------------------------
# false_positive_review tests
# ---------------------------------------------------------------------------


def test_false_positive_review_structure(eval_db) -> None:
    """false_positive_review() returns a list of dicts with required keys."""
    result = eval_db.false_positive_review(lookback_days=90)
    assert isinstance(result, list)
    for item in result:
        assert "geo_id" in item
        assert "alert_date" in item
        assert "ess_score" in item
        assert "top_signals" in item


def test_false_positive_review_top_signals_length(eval_db) -> None:
    """top_signals has at most 3 entries and they match the SHAP feature names."""
    result = eval_db.false_positive_review(lookback_days=90)
    assert isinstance(result, list)
    for item in result:
        assert len(item["top_signals"]) <= 3
        for feat in item["top_signals"]:
            assert feat in {"feat_a", "feat_b", "feat_c"}, (
                f"Unexpected feature name {feat!r} in top_signals"
            )


# ---------------------------------------------------------------------------
# performance_by_geography tests
# ---------------------------------------------------------------------------


def test_performance_by_geography_has_geo_level_key(eval_db) -> None:
    """performance_by_geography() contains 'by_geo_level' with rmse values."""
    result = eval_db.performance_by_geography()
    assert "by_geo_level" in result
    assert "overall_rmse" in result
    # All synthetic rows use geo_level="city"
    assert "city" in result["by_geo_level"]
    assert "rmse" in result["by_geo_level"]["city"]
    assert isinstance(result["by_geo_level"]["city"]["rmse"], float)


# ---------------------------------------------------------------------------
# performance_by_time_period tests
# ---------------------------------------------------------------------------


def test_performance_by_time_period_flags_high_months(eval_db) -> None:
    """performance_by_time_period() has flagged_months key and consistent structure."""
    result = eval_db.performance_by_time_period()
    assert "flagged_months" in result
    assert "overall_rmse" in result
    assert "by_month" in result
    assert isinstance(result["flagged_months"], list)
    for month, stats in result["by_month"].items():
        assert "rmse" in stats
        assert "flagged" in stats
        assert "n" in stats
        # flagged should be consistent: a month in flagged_months must have flagged=True
        if month in result["flagged_months"]:
            assert stats["flagged"] is True
