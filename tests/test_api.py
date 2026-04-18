"""Tests for the PulseIQ FastAPI layer.

Covers all 11 endpoints across four route files using FastAPI's ``TestClient``.
DuckDB queries are intercepted via ``unittest.mock.patch`` at the route-module
level so no real database is required.  ``StressExplainer`` and
``NewsRetriever`` are also mocked to avoid LLM and ChromaDB dependencies.

Test structure mirrors the five trust questions:
  - What is the score?        scores.py tests
  - Why is it that score?     explain.py tests
  - How fresh is the data?    health.py tests
  - What model produced it?   health.py tests
  - Can I trust this run?     health.py tests
  - Alert registration        alerts.py tests
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App bootstrap — must patch DUCKDB_PATH before importing main to bypass
# the lifespan startup DuckDB file-existence check.
# ---------------------------------------------------------------------------

import os
os.environ.setdefault("DUCKDB_PATH", ":memory:")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

from src.api.main import app  # noqa: E402 — must come after env setup


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a ``TestClient`` with ``app.state.db_path`` set to ``:memory:``.

    The ``TestClient`` context manager triggers the lifespan startup, which
    verifies the DuckDB connection.  We use ``:memory:`` so no real file
    is needed.
    """
    app.state.db_path = ":memory:"
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_row() -> dict[str, Any]:
    """Return a representative ``ess_scores`` row dict."""
    return {
        "geo_id": "Detroit-MI",
        "geo_name": "Detroit, MI",
        "geo_level": "city",
        "run_date": date.today(),
        "ess_score": 72.5,
        "score_band": "elevated",
        "delta_7d": 3.2,
        "delta_30d": 5.1,
        "confidence": "medium",
        "early_warning": False,
        "missing_sources": "[]",
        "stale_sources": "[]",
        "anomaly_flags": "[]",
        "granularity_warning": False,
        "model_version": "run-abc123",
        "feature_version": "abc123",
        "calibrated": True,
        "tier1_score": 35.0,
        "tier2_score": 20.0,
        "tier3_score": 17.5,
        "shap_values": json.dumps({
            "bls_jobless_claims_delta": 0.32,
            "fred_delinquency_rate": 0.21,
            "news_sentiment_score": -0.10,
        }),
    }


@pytest.fixture()
def sample_df(sample_row: dict[str, Any]) -> pd.DataFrame:
    """Single-row DataFrame matching ``ess_scores`` schema."""
    return pd.DataFrame([sample_row])


@pytest.fixture()
def empty_df() -> pd.DataFrame:
    """Empty DataFrame to simulate no matching rows."""
    return pd.DataFrame()


@pytest.fixture()
def sample_explanation():
    """Return a minimal valid ``Explanation`` object."""
    from src.contracts import Explanation, RetrievedSource

    return Explanation(
        geo_id="Detroit-MI",
        geo_name="Detroit, MI",
        run_date=date.today(),
        summary="Score rose 3.2 points over 7 days.",
        top_drivers=["Rising jobless claims", "Higher delinquency rate"],
        shap_breakdown={"bls_jobless_claims_delta": 0.32},
        retrieved_sources=[
            RetrievedSource(
                url="https://example.com/article",
                title="Detroit unemployment rises",
                published_at=datetime.now(tz=timezone.utc),
                relevance_score=0.88,
            )
        ],
        evidence_strength="moderate",
        confidence="medium",
        missing_sources=[],
        caveats=["None identified"],
        generated_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture()
def sample_alert_payload() -> dict[str, Any]:
    """Return a valid ``AlertPayload`` dict for POST /alerts tests."""
    return {
        "alert_id": str(uuid.uuid4()),
        "region_id": "Detroit-MI",
        "region_name": "Detroit, MI",
        "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
        "current_score": 78.0,
        "previous_score": 71.0,
        "score_delta": 7.0,
        "delta_window_days": 7,
        "alert_type": "threshold_breach",
        "top_drivers": ["Rising jobless claims"],
        "explanation_summary": "ESS crossed threshold of 75 for Detroit, MI.",
        "confidence": "medium",
        "missing_sources": [],
        "model_version": "run-abc123",
        "explanation_url": "http://localhost/api/v1/explain/Detroit-MI",
        "suppressed_until": None,
    }


# ---------------------------------------------------------------------------
# Helper: build mock connection that returns a DataFrame
# ---------------------------------------------------------------------------


def _mock_conn(df: pd.DataFrame) -> MagicMock:
    """Return a mock DuckDB connection whose ``execute().fetchdf()`` yields ``df``."""
    conn = MagicMock()
    conn.execute.return_value.fetchdf.return_value = df
    conn.execute.return_value.fetchone.return_value = None
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    return conn


def _mock_conn_with_snapshot(df: pd.DataFrame, snapshot_date: date) -> MagicMock:
    """Return a mock connection that can answer both fetchone() and fetchdf()."""
    conn = MagicMock()
    conn.execute.return_value.fetchdf.return_value = df
    conn.execute.return_value.fetchone.return_value = (snapshot_date,)
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    return conn


# ===========================================================================
# scores.py tests
# ===========================================================================


class TestGetScore:
    """Tests for GET /api/v1/scores/{geo_id}."""

    def test_200_schema_valid(self, client: TestClient, sample_df: pd.DataFrame) -> None:
        """200 response has all required fields including ``is_trustworthy``."""
        with patch("src.api.routes.scores.duckdb.connect") as mock_conn_cls, \
             patch("src.api.deps.duckdb.connect") as mock_dep_conn:
            mock_dep_conn.return_value = _mock_conn(sample_df)
            mock_conn_cls.return_value = _mock_conn(sample_df)

            # Override the Depends(get_db) dependency
            from src.api.deps import get_db
            app.dependency_overrides[get_db] = lambda: _mock_conn(sample_df)

            resp = client.get("/api/v1/scores/Detroit-MI")
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["geo_id"] == "Detroit-MI"
        assert "is_trustworthy" in data
        assert "confidence" in data
        assert "missing_sources" in data
        assert "model_version" in data

    def test_200_nan_optional_float_is_serialized_as_null(
        self, client: TestClient, sample_row: dict[str, Any]
    ) -> None:
        """Optional numeric fields normalize pandas NaN to JSON null."""
        row = dict(sample_row)
        row["delta_30d"] = float("nan")
        df = pd.DataFrame([row])

        from src.api.deps import get_db

        app.dependency_overrides[get_db] = lambda: _mock_conn(df)
        resp = client.get("/api/v1/scores/Detroit-MI")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["delta_30d"] is None

    def test_404_unknown_geo(self, client: TestClient, empty_df: pd.DataFrame) -> None:
        """404 when ``geo_id`` is not in ``ess_scores``."""
        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(empty_df)

        resp = client.get("/api/v1/scores/UNKNOWN-GEO")
        app.dependency_overrides.clear()

        assert resp.status_code == 404
        assert "detail" in resp.json()


class TestGetTopScores:
    """Tests for GET /api/v1/scores/top."""

    def test_200_limit_respected(self, client: TestClient) -> None:
        """Returns at most ``limit`` results.

        The mock DB returns exactly 5 rows (simulating what DuckDB would return
        after applying LIMIT 5).  The endpoint must not add more rows.
        """
        rows = [
            {
                "geo_id": f"City-{i}-ST", "geo_name": f"City {i}",
                "geo_level": "city", "run_date": date.today(),
                "ess_score": 90.0 - i, "score_band": "high",
                "delta_7d": 1.0, "delta_30d": 2.0,
                "confidence": "high", "early_warning": False,
                "missing_sources": "[]", "stale_sources": "[]",
                "anomaly_flags": "[]", "granularity_warning": False,
                "model_version": "v1", "feature_version": "abc",
                "calibrated": True,
                "tier1_score": 40.0, "tier2_score": 30.0, "tier3_score": 20.0,
                "shap_values": "{}",
            }
            for i in range(5)  # mock returns exactly 5 rows (DB enforces LIMIT)
        ]
        df = pd.DataFrame(rows)

        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(df)
        resp = client.get("/api/v1/scores/top?limit=5")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert len(resp.json()) <= 5

    def test_200_confidence_filter_high(self, client: TestClient) -> None:
        """When min_confidence=high, all results have confidence=='high'."""
        rows = [
            {
                "geo_id": "A-ST", "geo_name": "A",
                "geo_level": "city", "run_date": date.today(),
                "ess_score": 80.0, "score_band": "high",
                "delta_7d": None, "delta_30d": None,
                "confidence": "high", "early_warning": False,
                "missing_sources": "[]", "stale_sources": "[]",
                "anomaly_flags": "[]", "granularity_warning": False,
                "model_version": "v1", "feature_version": "abc",
                "calibrated": True,
                "tier1_score": 40.0, "tier2_score": 25.0, "tier3_score": 15.0,
                "shap_values": "{}",
            }
        ]
        df = pd.DataFrame(rows)

        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(df)
        resp = client.get("/api/v1/scores/top?min_confidence=high")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        for item in resp.json():
            assert item["confidence"] == "high"


class TestGetScoreSnapshot:
    """Tests for GET /api/v1/scores/snapshot."""

    def test_200_returns_snapshot_rows(
        self, client: TestClient, sample_df: pd.DataFrame
    ) -> None:
        """Snapshot endpoint returns score rows for a resolved run date."""
        from src.api.deps import get_db

        app.dependency_overrides[get_db] = lambda: _mock_conn_with_snapshot(
            sample_df, date.today()
        )
        resp = client.get("/api/v1/scores/snapshot")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["geo_id"] == "Detroit-MI"


class TestSearchGeographies:
    """Tests for GET /api/v1/scores/search."""

    def test_200_returns_matching_rows(
        self, client: TestClient, sample_df: pd.DataFrame
    ) -> None:
        """Search endpoint returns the latest matching score rows."""
        from src.api.deps import get_db

        app.dependency_overrides[get_db] = lambda: _mock_conn(sample_df)
        resp = client.get("/api/v1/scores/search?q=Detroit")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["geo_name"] == "Detroit, MI"


class TestGetMapScores:
    """Tests for GET /api/v1/scores/map."""

    def test_200_returns_state_level_map_rows(
        self, client: TestClient, sample_df: pd.DataFrame
    ) -> None:
        """Map endpoint returns state codes and drill-down geo IDs."""
        from src.api.deps import get_db

        app.dependency_overrides[get_db] = lambda: _mock_conn_with_snapshot(
            sample_df, date.today()
        )
        resp = client.get("/api/v1/scores/map")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["state_code"] == "MI"
        assert data[0]["drilldown_geo_id"] == "Detroit-MI"


class TestGetScoreHistory:
    """Tests for GET /api/v1/scores/{geo_id}/history."""

    def _make_history_df(self, n: int = 10) -> pd.DataFrame:
        from datetime import timedelta
        today = date.today()
        rows = []
        for i in range(n):
            rows.append({
                "run_date": today - timedelta(days=n - i - 1),
                "ess_score": 60.0 + i,
                "confidence": "medium",
                "missing_sources": "[]",
                "anomaly_flags": "[]",
                "geo_name": "Detroit, MI",
            })
        return pd.DataFrame(rows)

    def test_200_trend_present(self, client: TestClient) -> None:
        """200 response includes a valid trend value."""
        df = self._make_history_df(10)
        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(df)
        resp = client.get("/api/v1/scores/Detroit-MI/history?window=7d")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["trend"] in ("improving", "stable", "deteriorating", "volatile")
        assert "points" in data
        assert data["period_days"] == 7

    def test_trend_volatile_when_cv_above_threshold(self, client: TestClient) -> None:
        """Trend is 'volatile' when the CV of scores exceeds 0.15 (includes CV > 0.3).

        Uses oscillating scores [20, 80, 20, 80, 20, 80, 20, 80, 20, 80].
        Mean = 50, stddev ≈ 31.6 → CV ≈ 0.63 > 0.30 > 0.15 → volatile.
        """
        from datetime import timedelta
        from src.api.routes.scores import _compute_trend

        # Unit-test the function directly — no HTTP round-trip needed
        oscillating = [20.0, 80.0, 20.0, 80.0, 20.0, 80.0, 20.0, 80.0, 20.0, 80.0]
        import statistics
        mean = statistics.mean(oscillating)
        stddev = statistics.stdev(oscillating)
        cv = stddev / mean
        assert cv > 0.30, f"Test precondition: expected CV > 0.30, got {cv:.3f}"
        assert _compute_trend(oscillating) == "volatile"

    def test_404_unknown_geo(self, client: TestClient, empty_df: pd.DataFrame) -> None:
        """404 when no history exists for the geography."""
        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(empty_df)
        resp = client.get("/api/v1/scores/UNKNOWN-GEO/history")
        app.dependency_overrides.clear()

        assert resp.status_code == 404


class TestGetScoreDrivers:
    """Tests for GET /api/v1/scores/{geo_id}/drivers."""

    def test_200_has_top_shap_and_tier_breakdown(
        self, client: TestClient, sample_df: pd.DataFrame
    ) -> None:
        """200 response has ``top_shap`` and ``tier_breakdown`` keys."""
        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(sample_df)
        resp = client.get("/api/v1/scores/Detroit-MI/drivers")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert "top_shap" in data
        assert "tier_breakdown" in data
        assert "tier1_score" in data["tier_breakdown"]

    def test_404_unknown_geo(self, client: TestClient, empty_df: pd.DataFrame) -> None:
        """404 for unknown geography."""
        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(empty_df)
        resp = client.get("/api/v1/scores/UNKNOWN-GEO/drivers")
        app.dependency_overrides.clear()

        assert resp.status_code == 404


# ===========================================================================
# explain.py tests
# ===========================================================================


class TestGetExplanation:
    """Tests for GET /api/v1/explain/{geo_id}."""

    def test_200_explanation_schema(
        self,
        client: TestClient,
        sample_df: pd.DataFrame,
        sample_explanation,
    ) -> None:
        """200 response has Explanation schema with non-empty caveats."""
        mock_explainer = MagicMock()
        mock_explainer.explain.return_value = sample_explanation

        from src.api.deps import get_db, get_explainer
        app.dependency_overrides[get_db] = lambda: _mock_conn(sample_df)
        app.dependency_overrides[get_explainer] = lambda: mock_explainer

        resp = client.get("/api/v1/explain/Detroit-MI")
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["geo_id"] == "Detroit-MI"
        assert len(data["caveats"]) >= 1
        assert "generated_at" in data
        assert "summary" in data

    def test_404_unknown_geo(self, client: TestClient, empty_df: pd.DataFrame) -> None:
        """404 when geography is not in ess_scores."""
        from src.api.deps import get_db, get_explainer
        app.dependency_overrides[get_db] = lambda: _mock_conn(empty_df)
        app.dependency_overrides[get_explainer] = lambda: MagicMock()

        resp = client.get("/api/v1/explain/UNKNOWN-GEO")
        app.dependency_overrides.clear()

        assert resp.status_code == 404

    def test_200_without_llm_key_when_explainer_falls_back(
        self,
        client: TestClient,
        sample_df: pd.DataFrame,
        sample_explanation,
    ) -> None:
        """Route does not hard-fail when no LLM key is configured."""
        from src.api.deps import get_db, get_explainer

        mock_explainer = MagicMock()
        mock_explainer.explain.return_value = sample_explanation

        app.dependency_overrides[get_db] = lambda: _mock_conn(sample_df)
        app.dependency_overrides[get_explainer] = lambda: mock_explainer

        original = os.environ.pop("OPENAI_API_KEY", None)
        original_provider = os.environ.pop("LLM_PROVIDER", None)
        original_openrouter = os.environ.pop("OPENROUTER_API_KEY", None)
        original_nvidia = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            resp = client.get("/api/v1/explain/Detroit-MI")
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original
            if original_provider is not None:
                os.environ["LLM_PROVIDER"] = original_provider
            if original_openrouter is not None:
                os.environ["OPENROUTER_API_KEY"] = original_openrouter
            if original_nvidia is not None:
                os.environ["NVIDIA_API_KEY"] = original_nvidia
        app.dependency_overrides.clear()

        assert resp.status_code == 200


class TestGetEvidence:
    """Tests for GET /api/v1/explain/{geo_id}/evidence."""

    def test_200_returns_list_of_dicts(
        self, client: TestClient, sample_df: pd.DataFrame
    ) -> None:
        """200 response is a list of evidence dicts."""
        from src.contracts import RetrievedSource

        mock_retriever = MagicMock()
        mock_retriever.get_relevant_docs.return_value = [
            RetrievedSource(
                url="https://example.com/news",
                title="Test article",
                published_at=datetime.now(tz=timezone.utc),
                relevance_score=0.75,
            )
        ]

        with patch("src.api.routes.explain.NewsRetriever", return_value=mock_retriever):
            from src.api.deps import get_db
            app.dependency_overrides[get_db] = lambda: _mock_conn(sample_df)
            resp = client.get("/api/v1/explain/Detroit-MI/evidence")
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            assert "url" in data[0]
            assert "title" in data[0]
            assert "relevance_score" in data[0]

    def test_404_unknown_geo(self, client: TestClient, empty_df: pd.DataFrame) -> None:
        """404 when geography is not in ess_scores."""
        from src.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: _mock_conn(empty_df)
        resp = client.get("/api/v1/explain/UNKNOWN-GEO/evidence")
        app.dependency_overrides.clear()

        assert resp.status_code == 404


# ===========================================================================
# alerts.py tests
# ===========================================================================


class TestCreateAlert:
    """Tests for POST /api/v1/alerts."""

    def test_201_alert_id_generated(
        self, client: TestClient, sample_alert_payload: dict[str, Any]
    ) -> None:
        """201 response contains a UUID4 ``alert_id`` distinct from the input."""
        original_id = sample_alert_payload["alert_id"]

        with patch("src.api.routes.alerts.duckdb.connect") as mock_conn_cls:
            mock_conn_cls.return_value.__enter__.return_value.execute = MagicMock()
            mock_conn_cls.return_value.__exit__ = MagicMock(return_value=False)

            resp = client.post("/api/v1/alerts", json=sample_alert_payload)

        assert resp.status_code == 201
        data = resp.json()
        assert "alert_id" in data
        # Server replaces the caller-provided alert_id with a fresh UUID
        assert data["alert_id"] != original_id
        # Must be a valid UUID4
        uuid.UUID(data["alert_id"], version=4)

    def test_201_payload_fields_preserved(
        self, client: TestClient, sample_alert_payload: dict[str, Any]
    ) -> None:
        """Fields from the request body are echoed in the response."""
        with patch("src.api.routes.alerts.duckdb.connect") as mock_conn_cls:
            mock_conn_cls.return_value.__enter__.return_value.execute = MagicMock()
            mock_conn_cls.return_value.__exit__ = MagicMock(return_value=False)

            resp = client.post("/api/v1/alerts", json=sample_alert_payload)

        assert resp.status_code == 201
        data = resp.json()
        assert data["region_id"] == sample_alert_payload["region_id"]
        assert data["current_score"] == sample_alert_payload["current_score"]
        assert data["alert_type"] == sample_alert_payload["alert_type"]


class TestGetAlertHistory:
    """Tests for GET /api/v1/alerts/history/{geo_id}."""

    def test_200_returns_list(self, client: TestClient) -> None:
        """200 response is a (possibly empty) list."""
        with patch("src.api.routes.alerts.duckdb.connect") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame()
            mock_conn.__enter__ = lambda s: s
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn_cls.return_value = mock_conn

            resp = client.get("/api/v1/alerts/history/Detroit-MI")

        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ===========================================================================
# health.py tests
# ===========================================================================


class TestGetFreshness:
    """Tests for GET /api/v1/health/freshness."""

    def test_200_all_sources_present(self, client: TestClient) -> None:
        """Response includes one entry per known source."""
        from src.observability.metrics import SOURCE_CADENCE_DAYS
        from src.contracts import SourceFreshnessPayload

        mock_payloads = [
            SourceFreshnessPayload(
                source=src,
                last_successful_fetch=None,
                days_since_fetch=None,
                expected_cadence_days=cadence,
                freshness_status="unknown",
            )
            for src, cadence in SOURCE_CADENCE_DAYS.items()
        ]

        with patch(
            "src.api.routes.health.MetricsWriter"
        ) as mock_writer_cls:
            mock_writer_cls.return_value.get_all_source_health.return_value = mock_payloads
            resp = client.get("/api/v1/health/freshness")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        returned_sources = {item["source"] for item in data}
        expected_sources = set(SOURCE_CADENCE_DAYS.keys())
        assert expected_sources <= returned_sources


class TestGetModelVersion:
    """Tests for GET /api/v1/health/model."""

    def test_200_schema_valid(self, client: TestClient) -> None:
        """200 response satisfies ``ModelVersionResponse`` schema."""
        with patch("src.api.routes.health._find_latest_feature_version", return_value=None), \
             patch("src.api.routes.health.duckdb.connect") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            mock_conn.__enter__ = lambda s: s
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn_cls.return_value = mock_conn

            resp = client.get("/api/v1/health/model")

        assert resp.status_code == 200
        data = resp.json()
        # Required fields from ModelVersionResponse
        assert "model_version" in data
        assert "feature_version" in data
        assert "calibrated" in data
        assert "calibration_samples" in data


class TestGetPipelineStatus:
    """Tests for GET /api/v1/health/pipeline."""

    def test_200_schema_valid(self, client: TestClient) -> None:
        """200 response satisfies ``PipelineStatusResponse`` schema."""
        with patch("src.api.routes.health.duckdb.connect") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            mock_conn.__enter__ = lambda s: s
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn_cls.return_value = mock_conn

            resp = client.get("/api/v1/health/pipeline")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded", "down")
        assert "checked_at" in data
        assert "failures" in data


class TestGetDashboardHealth:
    """Tests for GET /api/v1/health/dashboard."""

    def test_200_dashboard_schema_valid(self, client: TestClient) -> None:
        """Dashboard endpoint returns combined source/model/pipeline payloads."""
        from src.contracts import (
            BenchmarkSummaryResponse,
            HealthDashboardResponse,
            ModelHealthSummary,
            PipelineHealthSummary,
            SourceHealthRow,
        )

        payload = HealthDashboardResponse(
            checked_at=datetime.now(tz=timezone.utc),
            source_health=[
                SourceHealthRow(
                    source="news",
                    last_run=datetime.now(tz=timezone.utc),
                    status="ok",
                    records=123,
                    latency_seconds=1.7,
                    trend_7d="stable",
                )
            ],
            model_info=ModelHealthSummary(
                version="run-abc123",
                trained_at=datetime.now(tz=timezone.utc),
                calibrated=True,
                benchmark=BenchmarkSummaryResponse(
                    model_rmse=1.2,
                    baseline_rmse=1.8,
                    improvement_pct=33.3,
                    verdict="PASS",
                    warning=None,
                ),
            ),
            pipeline_info=PipelineHealthSummary(
                status="ok",
                last_ingest_run=datetime.now(tz=timezone.utc),
                last_transform_run=datetime.now(tz=timezone.utc),
                last_score_run=datetime.now(tz=timezone.utc),
                anomaly_flags_count=2,
                failures=[],
            ),
        )

        with patch(
            "src.api.routes.health._build_health_dashboard_response",
            return_value=payload,
        ):
            resp = client.get("/api/v1/health/dashboard")

        assert resp.status_code == 200
        data = resp.json()
        assert "source_health" in data
        assert "model_info" in data
        assert "pipeline_info" in data


# ===========================================================================
# Root health shortcut
# ===========================================================================


class TestRootHealth:
    """Tests for GET /api/v1/health (root shortcut)."""

    def test_200_health_response_schema(self, client: TestClient) -> None:
        """Root /health returns a valid ``HealthResponse``."""
        from src.observability.metrics import SOURCE_CADENCE_DAYS
        from src.contracts import SourceFreshnessPayload

        mock_payloads = [
            SourceFreshnessPayload(
                source=src,
                last_successful_fetch=date.today(),
                days_since_fetch=0,
                expected_cadence_days=cadence,
                freshness_status="ok",
            )
            for src, cadence in SOURCE_CADENCE_DAYS.items()
        ]

        # main.py imports MetricsWriter inside the route function body;
        # patch it at its definition module so the local import picks it up.
        with patch("src.observability.metrics.MetricsWriter") as mock_writer_cls:
            mock_writer_cls.return_value.get_all_source_health.return_value = mock_payloads
            resp = client.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded", "down")
        assert "source_freshness" in data
        assert "stale_sources" in data
        assert "overall_data_quality" in data
        assert "checked_at" in data
