"""Tests for src/models/features.py.

All tests that need DuckDB use a temp file (via pytest's ``tmp_path`` fixture)
rather than ``:memory:`` because ``load_features`` opens its own connection and
cannot share an in-memory database with the test setup code.

Tests that exercise ``engineer_features`` and ``to_feature_vectors`` build
DataFrames directly from helper functions, avoiding the DuckDB layer entirely.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest
from pydantic import ValidationError

from src.models.features import (
    _MART_FEATURE_COLS,
    engineer_features,
    load_features,
    to_feature_vectors,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_BASE_DATE = date(2024, 1, 1)

_MART_ROW_DEFAULTS: dict = {
    "geo_level":              "city",
    "geo_name":               "Test City",
    "tier1_score":            0.10,
    "tier2_score":            0.05,
    "tier3_score":            0.02,
    "claims_yoy_pct":         0.0,
    "county_unemployment_rate": 4.0,
    "cpi_monthly_delta":      0.3,
    "delinquency_rate":       2.5,
    "median_income_index":    52_000.0,
    "reddit_negativity_score": 0.2,
    "post_volume_delta":      0.1,
    "distress_keyword_freq":  5,
    "poverty_rate":           14.0,
    "income_quartile":        2,
    "housing_cost_burden":    0.30,
    "extreme_weather_events_7d": 0,
    "weather_stress_index":   0.10,
    "ess_score":              65.0,
    "data_quality_score":     0.80,
    "granularity_warning":    True,
    "data_granularity_note":  "test note",
    "stale_sources":          "",
    "anomaly_flags":          "",
}


def _make_rows(
    geo_id: str,
    n_rows: int,
    *,
    jobless_claims_delta_values: list[float] | None = None,
    data_quality_score: float = 0.80,
) -> list[dict]:
    """Build a list of mart row dicts for a single geo_id over n_rows dates."""
    values = jobless_claims_delta_values or [0.1] * n_rows
    assert len(values) == n_rows, "jobless_claims_delta_values length must equal n_rows"
    rows = []
    for i in range(n_rows):
        row = {**_MART_ROW_DEFAULTS}
        row["geo_id"] = geo_id
        row["run_date"] = _BASE_DATE + timedelta(days=i)
        row["date"] = _BASE_DATE + timedelta(days=i)
        row["jobless_claims_delta"] = values[i]
        row["data_quality_score"] = data_quality_score
        rows.append(row)
    return rows


def _make_df(
    geo_id: str = "A-MI",
    n_rows: int = 5,
    *,
    jobless_claims_delta_values: list[float] | None = None,
    data_quality_score: float = 0.80,
) -> pd.DataFrame:
    """Return a DataFrame of mart rows for use with engineer_features."""
    rows = _make_rows(
        geo_id,
        n_rows,
        jobless_claims_delta_values=jobless_claims_delta_values,
        data_quality_score=data_quality_score,
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DuckDB fixture
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE mart_economic_stress (
    geo_id              VARCHAR,
    geo_level           VARCHAR,
    geo_name            VARCHAR,
    run_date            DATE,
    date                DATE,
    tier1_score         DOUBLE,
    tier2_score         DOUBLE,
    tier3_score         DOUBLE,
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


@pytest.fixture
def mart_db(tmp_path) -> str:
    """Temp DuckDB file with mart_economic_stress populated.

    Contains:
    - geo "A-MI": data_quality_score=0.80 (5 rows, passes filter)
    - geo "B-OH": data_quality_score=0.50 (1 row, filtered out)
    """
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    conn.execute(_CREATE_TABLE_SQL)
    _insert_rows(conn, _make_rows("A-MI", 5, data_quality_score=0.80))
    _insert_rows(conn, _make_rows("B-OH", 1, data_quality_score=0.50))
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Test 1 — load_features filters low data_quality rows
# ---------------------------------------------------------------------------


def test_load_features_excludes_low_data_quality(mart_db: str) -> None:
    """Rows with data_quality_score < 0.7 are excluded from the loaded DataFrame."""
    df = load_features(mart_db)
    assert len(df) > 0, "Expected at least one row"
    assert (df["data_quality_score"] >= 0.7).all(), (
        "All returned rows must have data_quality_score >= 0.7"
    )
    assert "B-OH" not in df["geo_id"].values, (
        "geo 'B-OH' (quality=0.50) must be excluded by the quality filter"
    )


def test_load_features_includes_high_quality_rows(mart_db: str) -> None:
    """Rows with data_quality_score >= 0.7 are included."""
    df = load_features(mart_db)
    assert "A-MI" in df["geo_id"].values


def test_load_features_ordered_by_geo_and_date(mart_db: str) -> None:
    """Returned DataFrame is sorted by (geo_id, date) ascending."""
    df = load_features(mart_db)
    expected = df.sort_values(["geo_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_load_features_run_date_filter(tmp_path) -> None:
    """run_date filter returns only rows up to and including the cutoff."""
    db_path = str(tmp_path / "filter_test.db")
    conn = duckdb.connect(db_path)
    conn.execute(_CREATE_TABLE_SQL)
    # 5 rows: 2024-01-01 through 2024-01-05
    _insert_rows(conn, _make_rows("A-MI", 5, data_quality_score=0.80))
    conn.close()

    cutoff = date(2024, 1, 3)
    df = load_features(db_path, run_date=cutoff)
    assert len(df) == 3
    assert (df["run_date"].dt.date <= cutoff).all()


def test_load_features_raises_on_missing_table(tmp_path) -> None:
    """RuntimeError is raised when the mart table does not exist."""
    db_path = str(tmp_path / "empty.db")
    duckdb.connect(db_path).close()  # create empty DB with no tables
    with pytest.raises(RuntimeError):
        load_features(db_path)


# ---------------------------------------------------------------------------
# Test 2 — engineer_features: correct jobless_claims_delta demeaning
# ---------------------------------------------------------------------------


def test_engineer_features_jobless_claims_delta_first_row() -> None:
    """Row 0 has no prior rows — prior mean is 0.0 so delta = original value."""
    df = _make_df("A-MI", n_rows=5, jobless_claims_delta_values=[1.0, 2.0, 3.0, 4.0, 5.0])
    result = engineer_features(df)
    # Row 0: no prior rows → prior_mean filled with 0.0 → 1.0 - 0.0 = 1.0
    assert result.iloc[0]["jobless_claims_delta"] == pytest.approx(1.0)


def test_engineer_features_jobless_claims_delta_second_row() -> None:
    """Row 1 subtracts the mean of [row 0] = 1.0 from its value 2.0."""
    df = _make_df("A-MI", n_rows=5, jobless_claims_delta_values=[1.0, 2.0, 3.0, 4.0, 5.0])
    result = engineer_features(df)
    # Row 1: shift(1).rolling(4).mean() of [1.0] = 1.0 → 2.0 - 1.0 = 1.0
    assert result.iloc[1]["jobless_claims_delta"] == pytest.approx(1.0)


def test_engineer_features_jobless_claims_delta_fifth_row() -> None:
    """Row 4 subtracts the mean of the prior 4 rows [1, 2, 3, 4] = 2.5."""
    df = _make_df("A-MI", n_rows=5, jobless_claims_delta_values=[1.0, 2.0, 3.0, 4.0, 5.0])
    result = engineer_features(df)
    # Row 4: prior 4 rows = [1, 2, 3, 4], mean = 2.5 → 5.0 - 2.5 = 2.5
    assert result.iloc[4]["jobless_claims_delta"] == pytest.approx(2.5)


def test_engineer_features_median_income_normalised() -> None:
    """median_income_index is divided by 56_000 (national median)."""
    df = _make_df("A-MI", n_rows=1)
    # Default median_income_index = 52_000.0
    result = engineer_features(df)
    expected = 52_000.0 / 56_000.0
    assert result.iloc[0]["median_income_index"] == pytest.approx(expected)


def test_engineer_features_does_not_mutate_input() -> None:
    """engineer_features returns a new DataFrame; the input is unchanged."""
    df = _make_df("A-MI", n_rows=5)
    original_value = df.iloc[0]["jobless_claims_delta"]
    engineer_features(df)
    assert df.iloc[0]["jobless_claims_delta"] == pytest.approx(original_value)


# ---------------------------------------------------------------------------
# Test 3 — engineer_features: no NaN in output
# ---------------------------------------------------------------------------


def test_engineer_features_no_nan_in_output_small() -> None:
    """All mart feature columns are NaN-free after engineering (5 rows)."""
    df = _make_df("A-MI", n_rows=5)
    result = engineer_features(df)
    nan_mask = result[_MART_FEATURE_COLS].isna()
    assert not nan_mask.any().any(), (
        f"NaN found in: {nan_mask.any()[nan_mask.any()].index.tolist()}"
    )


def test_engineer_features_no_nan_in_output_large() -> None:
    """All mart feature columns are NaN-free for 60 rows (beyond YoY window)."""
    df = _make_df("A-MI", n_rows=60)
    result = engineer_features(df)
    nan_mask = result[_MART_FEATURE_COLS].isna()
    assert not nan_mask.any().any(), (
        f"NaN found in: {nan_mask.any()[nan_mask.any()].index.tolist()}"
    )


def test_engineer_features_no_nan_multiple_geos() -> None:
    """NaN-free guarantee holds when multiple geo_ids are present."""
    df_a = _make_df("A-MI", n_rows=10)
    df_b = _make_df("B-OH", n_rows=10)
    combined = pd.concat([df_a, df_b], ignore_index=True)
    result = engineer_features(combined)
    nan_mask = result[_MART_FEATURE_COLS].isna()
    assert not nan_mask.any().any()


# ---------------------------------------------------------------------------
# Test 4 — to_feature_vectors raises on null field
# ---------------------------------------------------------------------------


def test_to_feature_vectors_raises_on_null_delinquency_rate() -> None:
    """None in delinquency_rate → ValidationError naming fred_delinquency_rate."""
    df = _make_df("A-MI", n_rows=1)
    df = engineer_features(df)
    # Inject None after engineering to simulate an upstream data gap
    df = df.copy()
    df.loc[df.index[0], "delinquency_rate"] = None
    with pytest.raises(ValidationError, match="fred_delinquency_rate"):
        to_feature_vectors(df)


def test_to_feature_vectors_raises_on_null_unemployment_rate() -> None:
    """None in county_unemployment_rate → ValidationError naming bls_unemployment_rate."""
    df = _make_df("A-MI", n_rows=1)
    df = engineer_features(df)
    df = df.copy()
    df.loc[df.index[0], "county_unemployment_rate"] = None
    with pytest.raises(ValidationError, match="bls_unemployment_rate"):
        to_feature_vectors(df)


def test_to_feature_vectors_valid_construction() -> None:
    """to_feature_vectors succeeds on a clean engineered DataFrame."""
    df = _make_df("A-MI", n_rows=3)
    result = engineer_features(df)
    vectors = to_feature_vectors(result)
    assert len(vectors) == 3
    assert all(v.geo_id == "A-MI" for v in vectors)


def test_to_feature_vectors_missing_features_are_zero() -> None:
    """fred_mortgage_rate_delta and trends_* are 0.0 (not in mart)."""
    df = _make_df("A-MI", n_rows=1)
    result = engineer_features(df)
    vectors = to_feature_vectors(result)
    fv = vectors[0]
    assert fv.fred_mortgage_rate_delta == pytest.approx(0.0)
    assert fv.trends_search_score == pytest.approx(0.0)
    assert fv.trends_search_delta == pytest.approx(0.0)


def test_to_feature_vectors_stale_source_count_from_string() -> None:
    """stale_source_count is derived from the comma-separated stale_sources column."""
    df = _make_df("A-MI", n_rows=1)
    df = df.copy()
    df.loc[df.index[0], "stale_sources"] = "bls,fred"
    result = engineer_features(df)
    vectors = to_feature_vectors(result)
    assert vectors[0].stale_source_count == 2


def test_to_feature_vectors_stale_source_count_empty_string() -> None:
    """Empty stale_sources string → stale_source_count = 0."""
    df = _make_df("A-MI", n_rows=1)
    result = engineer_features(df)
    vectors = to_feature_vectors(result)
    assert vectors[0].stale_source_count == 0
