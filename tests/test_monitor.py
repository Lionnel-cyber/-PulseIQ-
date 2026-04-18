"""Tests for src/models/monitor.py.

Uses temporary DuckDB files seeded with synthetic ``ess_scores`` rows.
No external connectors, no MLflow, no model loading.

Fixture layout
--------------
``stable_monitor``   — baseline and current windows with identical SHAP
                       distributions; all PSI values should be near 0.
``drifted_monitor``  — current window SHAP values shifted by 2 orders of
                       magnitude; at least one feature PSI must exceed 0.2.
``stale_monitor``    — current window rows carry ``stale_sources=["bls"]``;
                       baseline rows have no stale sources.
``empty_monitor``    — table exists but contains no rows.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import duckdb
import pytest

from src.models.monitor import PulseIQMonitor, _compute_psi

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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

_INSERT_SQL = """
INSERT INTO ess_scores VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""

# Five distinct geo_ids so we can create enough rows per date
_GEO_IDS = ["A-MI", "B-OH", "C-IL", "D-IN", "E-WI"]


def _row(
    geo_id: str,
    run_date: date,
    ess_score: float = 55.0,
    shap_values: dict | None = None,
    stale_sources: list[str] | None = None,
) -> list:
    """Build a single ess_scores row as a parameter list for INSERT."""
    return [
        geo_id,
        f"{geo_id} city",
        "city",
        run_date,
        ess_score,
        "low",
        None,           # delta_7d
        None,           # delta_30d
        "medium",
        False,
        json.dumps([]),
        json.dumps(stale_sources or []),
        json.dumps([]),
        False,
        "v1.0",
        "v1.0",
        True,
        10.0,
        8.0,
        5.0,
        json.dumps(shap_values or {"feat_a": 1.0, "feat_b": 0.5}),
    ]


def _seed(
    conn: duckdb.DuckDBPyConnection,
    baseline_shap: dict,
    current_shap: dict,
    baseline_score: float = 55.0,
    current_score: float = 55.0,
    baseline_stale: list[str] | None = None,
    current_stale: list[str] | None = None,
) -> None:
    """Seed ess_scores with 10 baseline rows and 5 current rows.

    Baseline window: 5 geos × date(today - 20)  and  5 geos × date(today - 15)
    Current window:  5 geos × date(today - 3)
    """
    today = date.today()
    baseline_dates = [today - timedelta(days=20), today - timedelta(days=15)]
    current_date = today - timedelta(days=3)

    conn.execute(_CREATE_ESS_SCORES_SQL)

    for d in baseline_dates:
        for geo in _GEO_IDS:
            conn.execute(
                _INSERT_SQL,
                _row(geo, d, baseline_score, baseline_shap, baseline_stale),
            )

    for geo in _GEO_IDS:
        conn.execute(
            _INSERT_SQL,
            _row(geo, current_date, current_score, current_shap, current_stale),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stable_monitor(tmp_path):
    """Identical SHAP distributions in both windows — PSI should be ≈ 0."""
    db_path = str(tmp_path / "stable.db")
    conn = duckdb.connect(db_path)
    _seed(conn, {"feat_a": 1.0, "feat_b": 0.5}, {"feat_a": 1.0, "feat_b": 0.5})
    conn.close()
    monitor = PulseIQMonitor(db_path)
    yield monitor
    monitor._conn.close()


@pytest.fixture()
def drifted_monitor(tmp_path):
    """Current window SHAP feat_a shifted from 1.0 → 100.0 (PSI >> 0.2)."""
    db_path = str(tmp_path / "drifted.db")
    conn = duckdb.connect(db_path)
    # Use a larger spread of values so binning produces distinct bucket distributions
    baseline_shap = {"feat_a": 1.0, "feat_b": 0.5}
    current_shap = {"feat_a": 100.0, "feat_b": 0.5}
    _seed(conn, baseline_shap, current_shap)
    conn.close()
    monitor = PulseIQMonitor(db_path)
    yield monitor
    monitor._conn.close()


@pytest.fixture()
def stale_monitor(tmp_path):
    """Current rows have stale_sources=['bls']; baseline rows are clean."""
    db_path = str(tmp_path / "stale.db")
    conn = duckdb.connect(db_path)
    _seed(
        conn,
        {"feat_a": 1.0},
        {"feat_a": 1.0},
        baseline_stale=[],
        current_stale=["bls"],
    )
    conn.close()
    monitor = PulseIQMonitor(db_path)
    yield monitor
    monitor._conn.close()


@pytest.fixture()
def empty_monitor(tmp_path):
    """Table exists but has zero rows."""
    db_path = str(tmp_path / "empty.db")
    conn = duckdb.connect(db_path)
    conn.execute(_CREATE_ESS_SCORES_SQL)
    conn.close()
    monitor = PulseIQMonitor(db_path)
    yield monitor
    monitor._conn.close()


@pytest.fixture()
def high_score_monitor(tmp_path):
    """Current rows have ess_score=85 (above alert threshold); baseline rows are 55."""
    db_path = str(tmp_path / "high_score.db")
    conn = duckdb.connect(db_path)
    _seed(conn, {"feat_a": 1.0}, {"feat_a": 1.0},
          baseline_score=55.0, current_score=85.0)
    conn.close()
    monitor = PulseIQMonitor(db_path)
    yield monitor
    monitor._conn.close()


# ---------------------------------------------------------------------------
# _compute_psi unit tests (no DB needed)
# ---------------------------------------------------------------------------


def test_psi_identical_distributions_is_zero() -> None:
    """Identical distributions must produce PSI = 0.0."""
    psi = _compute_psi([1.0] * 20, [1.0] * 20)
    assert psi == 0.0


def test_psi_non_overlapping_exceeds_threshold() -> None:
    """Completely non-overlapping distributions must produce PSI > 0.2."""
    psi = _compute_psi([0.0] * 20, [100.0] * 20)
    assert psi > 0.2


def test_psi_empty_raises() -> None:
    """Empty input lists must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        _compute_psi([], [1.0, 2.0])
    with pytest.raises(ValueError, match="empty"):
        _compute_psi([1.0, 2.0], [])


def test_psi_symmetric_approximately() -> None:
    """PSI(A, B) and PSI(B, A) should be similar in magnitude."""
    a = [1.0, 1.0, 1.0, 2.0, 2.0] * 4
    b = [3.0, 3.0, 4.0, 4.0, 4.0] * 4
    psi_ab = _compute_psi(a, b)
    psi_ba = _compute_psi(b, a)
    # PSI is not perfectly symmetric but should be in the same ballpark
    assert abs(psi_ab - psi_ba) < max(psi_ab, psi_ba) * 0.5


# ---------------------------------------------------------------------------
# feature_drift tests
# ---------------------------------------------------------------------------


def test_feature_drift_stable(stable_monitor: PulseIQMonitor) -> None:
    """Identical SHAP distributions must yield stable status and low PSI."""
    result = stable_monitor.feature_drift()

    assert result["insufficient_data"] is False
    assert result["overall_status"] == "stable"

    for feat in ("feat_a", "feat_b"):
        assert feat in result
        assert result[feat]["psi"] is not None
        assert result[feat]["psi"] < 0.1, (
            f"Expected low PSI for {feat} but got {result[feat]['psi']}"
        )
        assert result[feat]["status"] == "stable"


def test_feature_drift_detected(drifted_monitor: PulseIQMonitor) -> None:
    """A large SHAP shift must be detected as drift_detected for feat_a."""
    result = drifted_monitor.feature_drift()

    assert result["insufficient_data"] is False
    assert result["overall_status"] == "drift_detected"
    assert "feat_a" in result
    assert result["feat_a"]["psi"] > 0.2
    assert result["feat_a"]["status"] == "drift_detected"


def test_feature_drift_insufficient_data(empty_monitor: PulseIQMonitor) -> None:
    """Empty table must return insufficient_data=True without raising."""
    result = empty_monitor.feature_drift()

    assert result["insufficient_data"] is True
    assert result["overall_status"] == "stable"


def test_feature_drift_returns_required_keys(stable_monitor: PulseIQMonitor) -> None:
    """Return dict must always contain overall_status and insufficient_data."""
    result = stable_monitor.feature_drift()
    assert "overall_status" in result
    assert "insufficient_data" in result


# ---------------------------------------------------------------------------
# score_distribution_drift tests
# ---------------------------------------------------------------------------


def test_score_distribution_drift_structure(stable_monitor: PulseIQMonitor) -> None:
    """Return dict must have the four required keys with correct types."""
    result = stable_monitor.score_distribution_drift()

    assert "mean_shift" in result
    assert "std_shift" in result
    assert "drift_detected" in result
    assert "insufficient_data" in result
    assert isinstance(result["mean_shift"], float)
    assert isinstance(result["drift_detected"], bool)


def test_score_distribution_drift_stable(stable_monitor: PulseIQMonitor) -> None:
    """Identical scores in both windows must report no drift."""
    result = stable_monitor.score_distribution_drift()

    assert result["insufficient_data"] is False
    assert result["drift_detected"] is False
    assert abs(result["mean_shift"]) < 1.0


def test_score_distribution_drift_large_shift(high_score_monitor: PulseIQMonitor) -> None:
    """Current scores 30 pts above baseline must be flagged as drift_detected."""
    result = high_score_monitor.score_distribution_drift()

    assert result["insufficient_data"] is False
    assert result["drift_detected"] is True
    assert result["mean_shift"] > 5.0


def test_score_distribution_drift_empty(empty_monitor: PulseIQMonitor) -> None:
    """Empty table must return insufficient_data=True."""
    result = empty_monitor.score_distribution_drift()
    assert result["insufficient_data"] is True


# ---------------------------------------------------------------------------
# missing_source_drift tests
# ---------------------------------------------------------------------------


def test_missing_source_drift_structure(stale_monitor: PulseIQMonitor) -> None:
    """Return dict must have an entry for every known source."""
    from src.observability.metrics import SOURCE_CADENCE_DAYS

    result = stale_monitor.missing_source_drift()

    for source in SOURCE_CADENCE_DAYS:
        assert source in result
        assert "current_missing_rate" in result[source]
        assert "baseline_rate" in result[source]
        assert "flagged" in result[source]


def test_missing_source_drift_bls_flagged(stale_monitor: PulseIQMonitor) -> None:
    """bls should be flagged when current rate is 100 % and baseline is 0 %."""
    result = stale_monitor.missing_source_drift()

    assert result["bls"]["flagged"] is True
    assert result["bls"]["current_missing_rate"] == pytest.approx(1.0)
    assert result["bls"]["baseline_rate"] == pytest.approx(0.0)


def test_missing_source_drift_stable_not_flagged(stable_monitor: PulseIQMonitor) -> None:
    """No sources should be flagged when stale_sources is empty in all rows."""
    result = stable_monitor.missing_source_drift()

    for source, info in result.items():
        assert info["flagged"] is False, f"Expected {source} not flagged"


# ---------------------------------------------------------------------------
# alert_volume_drift tests
# ---------------------------------------------------------------------------


def test_alert_volume_drift_structure(stable_monitor: PulseIQMonitor) -> None:
    """Return dict must have the four required keys."""
    result = stable_monitor.alert_volume_drift()

    assert "current_volume" in result
    assert "baseline_volume" in result
    assert "pct_change" in result
    assert "flagged" in result
    assert isinstance(result["current_volume"], int)
    assert isinstance(result["flagged"], bool)


def test_alert_volume_drift_no_alerts_not_flagged(stable_monitor: PulseIQMonitor) -> None:
    """Scores of 55 are below the 75 threshold — both volumes must be 0."""
    result = stable_monitor.alert_volume_drift()

    assert result["current_volume"] == 0
    assert result["baseline_volume"] == 0
    assert result["flagged"] is False


def test_alert_volume_drift_high_scores_flagged(high_score_monitor: PulseIQMonitor) -> None:
    """Current scores of 85 must appear in current_volume."""
    result = high_score_monitor.alert_volume_drift()

    assert result["current_volume"] > 0


# ---------------------------------------------------------------------------
# retraining_recommendation tests
# ---------------------------------------------------------------------------


def test_retraining_recommendation_immediate_on_high_psi(
    drifted_monitor: PulseIQMonitor,
) -> None:
    """Extreme SHAP shift must trigger an immediate recommendation."""
    result = drifted_monitor.retraining_recommendation()

    assert result["recommendation"] == "immediate"
    assert "feature_drift" in result["triggered_by"]
    assert len(result["evidence"]) > 0
    # Evidence must name the drifted feature
    evidence_text = " ".join(result["evidence"])
    assert "feat_a" in evidence_text


def test_retraining_recommendation_no_action_all_stable(
    stable_monitor: PulseIQMonitor,
) -> None:
    """All-stable signals must return no_action with empty evidence."""
    result = stable_monitor.retraining_recommendation()

    assert result["recommendation"] == "no_action"
    assert result["evidence"] == []
    assert result["triggered_by"] == []


def test_retraining_recommendation_returns_required_keys(
    stable_monitor: PulseIQMonitor,
) -> None:
    """Return dict must always contain recommendation, evidence, triggered_by."""
    result = stable_monitor.retraining_recommendation()

    assert "recommendation" in result
    assert "evidence" in result
    assert "triggered_by" in result
    assert result["recommendation"] in {"no_action", "schedule", "immediate"}
    assert isinstance(result["evidence"], list)
    assert isinstance(result["triggered_by"], list)


def test_retraining_recommendation_empty_table(empty_monitor: PulseIQMonitor) -> None:
    """Empty table must return no_action without raising."""
    result = empty_monitor.retraining_recommendation()

    assert result["recommendation"] == "no_action"
    assert isinstance(result["evidence"], list)
