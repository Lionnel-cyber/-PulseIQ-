"""Tests for src/observability/metrics.py.

All tests use an in-memory DuckDB database via ``MetricsWriter(db_path=":memory:")``.
Webhook HTTP calls are intercepted with ``unittest.mock.patch("requests.post")``.
No fixtures are shared — each test creates its own isolated writer and data.
"""

import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.observability.metrics import (
    IngestionMetrics,
    MetricsWriter,
    SOURCE_CADENCE_DAYS,
    SourceFreshnessPayload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(
    source: str = "bls",
    run_date: date | None = None,
    records_fetched: int = 500,
    latency_seconds: float = 2.0,
    freshness_status: str = "ok",
    success: bool = True,
    error_message: str | None = None,
    run_id: str | None = None,
) -> IngestionMetrics:
    """Return a populated IngestionMetrics with sensible defaults."""
    now = datetime.now(timezone.utc)
    return IngestionMetrics(
        source=source,
        run_date=run_date or date.today(),
        run_id=run_id or str(uuid.uuid4()),
        started_at=now,
        completed_at=now,
        records_fetched=records_fetched,
        records_rejected=0,
        records_suspect=0,
        latency_seconds=latency_seconds,
        freshness_status=freshness_status,
        http_retries=0,
        success=success,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_write_inserts_row() -> None:
    """write_ingestion_metrics() inserts one row with the correct field values."""
    writer = MetricsWriter(db_path=":memory:")
    m = _make_metrics(source="bls", records_fetched=1_200, latency_seconds=3.5)
    writer.write_ingestion_metrics(m)

    rows = writer._conn.execute(
        "SELECT source, records_fetched, latency_seconds, success FROM ingestion_metrics"
    ).fetchall()

    assert len(rows) == 1
    source, records_fetched, latency_seconds, success = rows[0]
    assert source == "bls"
    assert records_fetched == 1_200
    assert latency_seconds == pytest.approx(3.5)
    assert success is True


def test_write_idempotent_same_run_id() -> None:
    """Writing the same run_id twice replaces the row — no duplicate is created."""
    writer = MetricsWriter(db_path=":memory:")
    run_id = str(uuid.uuid4())

    writer.write_ingestion_metrics(_make_metrics(records_fetched=100, run_id=run_id))
    writer.write_ingestion_metrics(_make_metrics(records_fetched=100, run_id=run_id))

    count = writer._conn.execute(
        "SELECT COUNT(*) FROM ingestion_metrics"
    ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Source health / freshness tests
# ---------------------------------------------------------------------------


def test_get_source_health_ok() -> None:
    """A source that ran today has freshness_status='ok'."""
    writer = MetricsWriter(db_path=":memory:")
    writer.write_ingestion_metrics(_make_metrics(source="bls", run_date=date.today()))

    health = writer.get_source_health("bls")

    assert health.freshness_status == "ok"
    assert health.days_since_fetch == 0
    assert health.expected_cadence_days == SOURCE_CADENCE_DAYS["bls"]


def test_get_source_health_stale() -> None:
    """BLS data 14 days old (> 7*1.5=10.5) is 'stale'."""
    writer = MetricsWriter(db_path=":memory:")
    stale_date = date.today() - timedelta(days=14)
    writer.write_ingestion_metrics(_make_metrics(source="bls", run_date=stale_date))

    health = writer.get_source_health("bls")

    assert health.freshness_status == "stale"
    assert health.days_since_fetch == 14


def test_get_source_health_critical() -> None:
    """BLS data 25 days old (> 7*3=21) is 'critical'."""
    writer = MetricsWriter(db_path=":memory:")
    critical_date = date.today() - timedelta(days=25)
    writer.write_ingestion_metrics(_make_metrics(source="bls", run_date=critical_date))

    health = writer.get_source_health("bls")

    assert health.freshness_status == "critical"
    assert health.days_since_fetch == 25


def test_get_source_health_unknown_no_data() -> None:
    """A source with no rows in the table returns freshness_status='unknown'."""
    writer = MetricsWriter(db_path=":memory:")

    health = writer.get_source_health("bls")

    assert health.freshness_status == "unknown"
    assert health.last_successful_fetch is None
    assert health.days_since_fetch is None


def test_get_source_health_uses_last_success_not_last_run() -> None:
    """Freshness is based on last SUCCESSFUL run date, ignoring failed runs."""
    writer = MetricsWriter(db_path=":memory:")
    # Last successful run was 5 days ago
    writer.write_ingestion_metrics(
        _make_metrics(source="bls", run_date=date.today() - timedelta(days=5))
    )
    # Failed run happened today — should not count toward freshness
    writer.write_ingestion_metrics(
        _make_metrics(source="bls", run_date=date.today(), success=False, error_message="API down")
    )

    health = writer.get_source_health("bls")

    assert health.days_since_fetch == 5
    assert health.freshness_status == "ok"  # 5 <= 7*1.5=10.5


def test_get_all_source_health_covers_all_sources() -> None:
    """get_all_source_health() returns exactly one entry per SOURCE_CADENCE_DAYS key."""
    writer = MetricsWriter(db_path=":memory:")

    results = writer.get_all_source_health()

    assert len(results) == len(SOURCE_CADENCE_DAYS)
    returned_sources = {r.source for r in results}
    assert returned_sources == set(SOURCE_CADENCE_DAYS.keys())
    # All entries should be SourceFreshnessPayload instances
    for r in results:
        assert isinstance(r, SourceFreshnessPayload)


# ---------------------------------------------------------------------------
# MTTD alert tests
# ---------------------------------------------------------------------------


def test_alert_fired_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed ingestion run fires a critical alert to the webhook."""
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/alert")
    writer = MetricsWriter(db_path=":memory:")

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        writer.write_ingestion_metrics(
            _make_metrics(success=False, error_message="Connection refused")
        )

    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["severity"] == "critical"
    assert payload["alert_type"] == "ingestion_failure"


def test_no_alert_when_webhook_url_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ALERT_WEBHOOK_URL is not set, requests.post is never called."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    writer = MetricsWriter(db_path=":memory:")

    with patch("requests.post") as mock_post:
        writer.write_ingestion_metrics(
            _make_metrics(success=False, error_message="Connection refused")
        )

    mock_post.assert_not_called()


def test_alert_fired_on_record_count_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A >50 % drop from 7-day average fires a record_count_drop warning."""
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/alert")
    writer = MetricsWriter(db_path=":memory:")

    # Write 5 historical successful runs with 200 records each
    for i in range(5):
        writer.write_ingestion_metrics(
            _make_metrics(
                run_date=date.today() - timedelta(days=i + 1),
                records_fetched=200,
            )
        )

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        # Today's run only fetched 50 records — well below avg * 0.5 = 100
        writer.write_ingestion_metrics(
            _make_metrics(run_date=date.today(), records_fetched=50)
        )

    # At least one call should be the record_count_drop alert
    alert_types = [
        call.kwargs["json"]["alert_type"] for call in mock_post.call_args_list
    ]
    assert "record_count_drop" in alert_types


def test_no_alert_on_success_within_normal_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean successful run with normal record count fires no alert."""
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/alert")
    writer = MetricsWriter(db_path=":memory:")

    with patch("requests.post") as mock_post:
        writer.write_ingestion_metrics(_make_metrics(success=True, records_fetched=500))

    mock_post.assert_not_called()
