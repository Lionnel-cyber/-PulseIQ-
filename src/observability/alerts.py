"""MTTD alert suppression and webhook delivery for PulseIQ.

Implements the suppression rules from CLAUDE.md:

    threshold_breach  → 3-day cooldown
    rapid_rise        → 1-day cooldown
    sustained_high    → 7-day cooldown
    min score delta   → 5 points (no re-alert for sticky scores)

Fired alerts are persisted in the DuckDB ``alert_history`` table so
suppression checks survive process restarts between daily runs.

Typical usage::

    from src.observability.alerts import fire_alerts_for_today

    fired, suppressed = fire_alerts_for_today(
        db_path="data/processed/pulseiq.db",
        threshold=75.0,
        webhook_url="https://hooks.example.com/pulseiq",
        webhook_secret="...",
    )
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date, datetime, timedelta, timezone

import duckdb
import requests

from src.contracts import AlertPayload, AlertType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — suppression rules (CLAUDE.md)
# ---------------------------------------------------------------------------

_COOLDOWN_DAYS: dict[str, int] = {
    "threshold_breach": 3,
    "rapid_rise": 1,
    "sustained_high": 7,
    # ingestion_failure, record_count_drop, latency_spike, source_stale
    # fire immediately — no cooldown configured
}
"""Per-alert-type cooldown windows in days."""

_MIN_SCORE_DELTA: float = 5.0
"""Minimum score change required to re-fire an alert for the same geo+type.

CLAUDE.md: "min score delta → 5 points (no re-alert for sticky scores)".
"""

_CREATE_ALERT_HISTORY_SQL: str = """
CREATE TABLE IF NOT EXISTS alert_history (
    alert_id   VARCHAR   NOT NULL,
    geo_id     VARCHAR   NOT NULL,
    alert_type VARCHAR   NOT NULL,
    score      DOUBLE    NOT NULL,
    fired_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (alert_id)
)
"""


# ---------------------------------------------------------------------------
# AlertSuppressor
# ---------------------------------------------------------------------------


class AlertSuppressor:
    """Manages per-geo cooldown windows to prevent alert flooding.

    Cooldown state is persisted in the DuckDB ``alert_history`` table.
    Two independent gates control every candidate alert:

    1. **Score-delta gate** — ``|current − previous| < 5 pts`` suppresses
       re-alerts on sticky scores even if the cooldown window has elapsed.
    2. **Cooldown gate** — checks whether the same ``geo_id`` + ``alert_type``
       pair fired within its type-specific window (3, 1, or 7 days).

    Args:
        db_path: Path to the DuckDB file that stores ``alert_history``.
            The table is created on first instantiation.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        with duckdb.connect(db_path) as conn:
            conn.execute(_CREATE_ALERT_HISTORY_SQL)

    def is_suppressed(
        self,
        geo_id: str,
        alert_type: str,
        current_score: float,
        previous_score: float,
    ) -> bool:
        """Return True if the alert should be suppressed.

        Applies the CLAUDE.md min-delta and cooldown rules in order.

        Args:
            geo_id: Geography identifier.
            alert_type: One of the ``AlertType`` literals.
            current_score: ESS score at the time of the candidate alert.
            previous_score: ESS score at the start of the delta window.

        Returns:
            True if the alert is suppressed, False if it should fire.
        """
        # Gate 1 — minimum score delta
        score_delta = abs(current_score - previous_score)
        if score_delta < _MIN_SCORE_DELTA:
            logger.debug(
                "Suppressed (min-delta %.1f < %.1f): geo=%s type=%s",
                score_delta, _MIN_SCORE_DELTA, geo_id, alert_type,
            )
            return True

        # Gate 2 — per-type cooldown window
        cooldown_days = _COOLDOWN_DAYS.get(alert_type, 0)
        if cooldown_days == 0:
            return False  # infrastructure alert types fire immediately

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=cooldown_days)
        try:
            with duckdb.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM alert_history"
                    " WHERE geo_id = ? AND alert_type = ? AND fired_at >= ?",
                    [geo_id, alert_type, cutoff],
                ).fetchone()
                recent_count = int(row[0]) if row else 0
        except Exception as exc:
            logger.warning("AlertSuppressor DB error during cooldown check: %s", exc)
            recent_count = 0

        if recent_count > 0:
            logger.debug(
                "Suppressed (cooldown %d d): geo=%s type=%s",
                cooldown_days, geo_id, alert_type,
            )
            return True

        return False

    def record_alert(
        self,
        geo_id: str,
        alert_type: str,
        score: float,
        fired_at: datetime,
    ) -> str:
        """Persist a fired alert so future suppression checks see it.

        Args:
            geo_id: Geography identifier.
            alert_type: One of the ``AlertType`` literals.
            score: ESS score at the time of firing.
            fired_at: Timestamp when the alert fired (timezone-aware).

        Returns:
            UUID4 ``alert_id`` string for the new record.
        """
        alert_id = str(uuid.uuid4())
        with duckdb.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO alert_history"
                " (alert_id, geo_id, alert_type, score, fired_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [alert_id, geo_id, alert_type, score, fired_at],
            )
        logger.debug(
            "Alert recorded: id=%s geo=%s type=%s score=%.1f",
            alert_id, geo_id, alert_type, score,
        )
        return alert_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fire_alerts_for_today(
    db_path: str,
    threshold: float,
    webhook_url: str,
    webhook_secret: str,
) -> tuple[int, int]:
    """Read today's ess_scores, apply suppression, POST alerts to webhook.

    For each geography whose ``ess_score >= threshold``:

    1. Derives ``alert_type`` from score and ``delta_7d``.
    2. Checks ``AlertSuppressor`` — skips if suppressed.
    3. Builds an ``AlertPayload`` and POSTs it to ``webhook_url``.
    4. Records the fired alert so future cooldown checks see it.

    Alert-type priority:

    - ``"rapid_rise"``     — ``delta_7d >= 10``
    - ``"sustained_high"`` — ``ess_score >= 85``
    - ``"threshold_breach"`` — ``ess_score >= threshold`` (default)

    Args:
        db_path: Path to the DuckDB file containing ``ess_scores``
            and ``alert_history``.
        threshold: Minimum ESS score to consider for alerting.
        webhook_url: Destination URL for POST requests.
        webhook_secret: Shared secret sent as ``X-Webhook-Secret`` header.

    Returns:
        Tuple ``(alerts_fired, alerts_suppressed)``.
    """
    today = date.today()
    suppressor = AlertSuppressor(db_path=db_path)

    try:
        with duckdb.connect(db_path) as conn:
            df = conn.execute(
                "SELECT * FROM ess_scores WHERE run_date = ? AND ess_score >= ?",
                [today, threshold],
            ).df()
    except Exception as exc:
        logger.warning("fire_alerts_for_today: cannot read ess_scores: %s", exc)
        return 0, 0

    fired = 0
    suppressed = 0

    for _, row in df.iterrows():
        geo_id = str(row["geo_id"])
        current_score = float(row["ess_score"])

        raw_delta = row.get("delta_7d")
        delta_7d: float | None = (
            float(raw_delta)
            if raw_delta is not None and str(raw_delta) not in ("nan", "None")
            else None
        )
        previous_score = (current_score - delta_7d) if delta_7d is not None else current_score

        # Determine alert type by priority
        if delta_7d is not None and delta_7d >= 10.0:
            alert_type: AlertType = "rapid_rise"
        elif current_score >= 85.0:
            alert_type = "sustained_high"
        else:
            alert_type = "threshold_breach"

        if suppressor.is_suppressed(geo_id, alert_type, current_score, previous_score):
            suppressed += 1
            continue

        # Extract top 3 SHAP drivers
        try:
            shap: dict[str, float] = json.loads(str(row.get("shap_values") or "{}"))
            top_drivers = [
                f"{k}: {v:+.3f}"
                for k, v in sorted(
                    shap.items(), key=lambda kv: abs(kv[1]), reverse=True
                )[:3]
            ]
        except (json.JSONDecodeError, TypeError, ValueError):
            top_drivers = []

        # Parse JSON list fields
        try:
            missing_sources: list[str] = json.loads(
                str(row.get("missing_sources") or "[]")
            )
        except (json.JSONDecodeError, TypeError):
            missing_sources = []

        fired_at = datetime.now(tz=timezone.utc)
        alert_id = suppressor.record_alert(geo_id, alert_type, current_score, fired_at)
        geo_name = str(row.get("geo_name") or geo_id)

        payload = AlertPayload(
            alert_id=alert_id,
            region_id=geo_id,
            region_name=geo_name,
            triggered_at=fired_at,
            current_score=current_score,
            previous_score=previous_score,
            score_delta=current_score - previous_score,
            delta_window_days=7,
            alert_type=alert_type,
            top_drivers=top_drivers,
            explanation_summary=(
                f"ESS score of {current_score:.1f} exceeded threshold of"
                f" {threshold:.0f} for {geo_name}."
            ),
            confidence=str(row.get("confidence") or "low"),
            missing_sources=missing_sources,
            model_version=str(row.get("model_version") or ""),
            explanation_url="",  # populated by the API layer in Phase 3
            suppressed_until=None,
        )

        try:
            resp = requests.post(
                webhook_url,
                json=payload.model_dump(mode="json"),
                headers={"X-Webhook-Secret": webhook_secret},
                timeout=10,
            )
            resp.raise_for_status()
            fired += 1
            logger.info(
                "Alert fired: geo=%s type=%s score=%.1f alert_id=%s",
                geo_id, alert_type, current_score, alert_id,
            )
        except Exception as exc:
            logger.error(
                "Alert delivery failed: geo=%s type=%s exc=%s",
                geo_id, alert_type, exc,
            )

    return fired, suppressed
