"""Validation rules for all PulseIQ data sources.

Every value ingested from every connector must pass through ``validate()``
before being written to the mart. Three outcomes are possible — never binary:

- ``ValidationResult.VALID``    — use normally
- ``ValidationResult.SUSPECT``  — use with ``anomaly_flag=True`` in output
- ``ValidationResult.REJECTED`` — do not write to mart; log to metrics

Rules are defined in ``VALIDATION_RULES`` keyed by ``source → field``.
Adding a new source or field means adding an entry here — nowhere else.
"""

import enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ValidationResult(enum.Enum):
    """Outcome of a single value validation check.

    Attributes:
        VALID:    Value is within expected range. Use normally.
        SUSPECT:  Value is anomalous but plausible. Write to mart with
                  ``anomaly_flag=True``.
        REJECTED: Value is impossible or a known data error. Do not write
                  to mart; log the rejection to ``IngestionMetrics``.
    """

    VALID = "valid"
    SUSPECT = "suspect"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

# Each entry has:
#   expected_range        (lo, hi)  — outside this → SUSPECT
#   hard_limits           (lo, hi)  — outside this → REJECTED
#   spike_threshold_pct   float     — pct change vs previous → SUSPECT
#   max_monthly_delta_pct float     — same check, CPI-specific label
#   drop_to_zero          "reject"  — value == 0 → REJECTED (data error, not reality)

VALIDATION_RULES: dict[str, dict[str, dict[str, Any]]] = {
    "bls": {
        "jobless_claims": {
            # Weekly initial jobless claims filed nationally.
            # Zero is impossible — BLS stops publishing rather than reporting 0.
            "expected_range": (1_000, 1_000_000),
            "hard_limits": (0, 10_000_000),
            "spike_threshold_pct": 300,
            "drop_to_zero": "reject",
        },
    },
    "fred": {
        "cpi": {
            # Consumer Price Index (CPI-U, 1982–84=100).
            # A monthly delta above 5 % is suspicious and warrants a flag,
            # but is not impossible — flag it, do not reject.
            "expected_range": (200, 400),
            "hard_limits": (0, 1_000),
            "max_monthly_delta_pct": 5,
        },
        "delinquency_rate": {
            # Delinquency rate on credit-card loans (percentage points).
            "expected_range": (0.5, 15.0),
            "hard_limits": (0.0, 100.0),
            "spike_threshold_pct": 200,
        },
    },
    "news": {
        "sentiment_score": {
            # Composite sentiment score, bounded [-1, 1].
            # A 10x spike relative to the previous value may indicate a
            # bot wave or a viral story rather than genuine economic signal.
            "expected_range": (-1.0, 1.0),
            "hard_limits": (-1.0, 1.0),
            "spike_threshold_pct": 1_000,
        },
    },
    "trends": {
        "search_score": {
            # Google Trends normalised search score, always 0–100.
            "expected_range": (0, 100),
            "hard_limits": (0, 100),
            "spike_threshold_pct": 500,
        },
    },
}


# ---------------------------------------------------------------------------
# Public validation function
# ---------------------------------------------------------------------------


def validate(
    source: str,
    field: str,
    value: float,
    previous_value: float | None = None,
) -> ValidationResult:
    """Validate a single ingested value against the rule for its source and field.

    Decision order — returns at the first matching condition:

    1. **Unknown rule** — ``source``/``field`` not in ``VALIDATION_RULES``.
       Returns ``VALID`` (no rule to enforce; unknown fields pass through).
    2. **Hard limits exceeded** — ``value`` is outside ``hard_limits``.
       Returns ``REJECTED``.
    3. **Drop-to-zero** — rule has ``drop_to_zero="reject"`` and ``value == 0``.
       Returns ``REJECTED`` (BLS claims can never legitimately be zero).
    4. **Spike or large delta** — ``previous_value`` is provided and non-zero,
       and the absolute percentage change exceeds ``spike_threshold_pct`` or
       ``max_monthly_delta_pct``. Returns ``SUSPECT``.
       Skipped entirely if ``previous_value == 0`` (avoids division by zero).
    5. **Outside expected range** — ``value`` is outside ``expected_range``.
       Returns ``SUSPECT``.
    6. **Default** — Returns ``VALID``.

    Args:
        source: Data source identifier (e.g. ``"bls"``, ``"fred"``,
            ``"news"``, ``"trends"``). Must match a top-level key in
            ``VALIDATION_RULES`` to apply any rules.
        field: Field name within the source (e.g. ``"jobless_claims"``,
            ``"cpi"``). Must match a second-level key in ``VALIDATION_RULES``.
        value: The numeric value to validate.
        previous_value: The most recent prior value for the same
            source/field/geography. Used to detect spikes and large deltas.
            Pass ``None`` (default) when no prior value is available — the
            spike check is skipped entirely in that case.

    Returns:
        ``ValidationResult.VALID``, ``ValidationResult.SUSPECT``, or
        ``ValidationResult.REJECTED``.

    Examples:
        >>> validate("bls", "jobless_claims", 0)
        <ValidationResult.REJECTED: 'rejected'>

        >>> validate("news", "sentiment_score", 0.6, previous_value=0.05)
        <ValidationResult.SUSPECT: 'suspect'>

        >>> validate("fred", "cpi", 310.0)
        <ValidationResult.VALID: 'valid'>

        >>> validate("openweather", "temperature", 72.0)
        <ValidationResult.VALID: 'valid'>
    """
    # 1. Unknown rule — no enforcement
    source_rules = VALIDATION_RULES.get(source)
    if source_rules is None:
        logger.debug("No validation rules for source %r — passing through", source)
        return ValidationResult.VALID

    rule = source_rules.get(field)
    if rule is None:
        logger.debug(
            "No validation rule for source=%r field=%r — passing through", source, field
        )
        return ValidationResult.VALID

    hard_lo, hard_hi = rule["hard_limits"]
    exp_lo, exp_hi = rule["expected_range"]

    # 2. Hard limits
    if value < hard_lo or value > hard_hi:
        logger.warning(
            "REJECTED: source=%r field=%r value=%s outside hard_limits=(%s, %s)",
            source, field, value, hard_lo, hard_hi,
        )
        return ValidationResult.REJECTED

    # 3. Drop-to-zero (data error, not economic reality)
    if rule.get("drop_to_zero") == "reject" and value == 0:
        logger.warning(
            "REJECTED: source=%r field=%r value=0 — drop_to_zero rule triggered",
            source, field,
        )
        return ValidationResult.REJECTED

    # 4. Spike / large monthly delta check
    if previous_value is not None and previous_value != 0:
        threshold: float | None = rule.get("spike_threshold_pct") or rule.get(
            "max_monthly_delta_pct"
        )
        if threshold is not None:
            pct_change = abs((value - previous_value) / previous_value * 100)
            if pct_change > threshold:
                logger.warning(
                    "SUSPECT: source=%r field=%r pct_change=%.1f%% > threshold=%.1f%%",
                    source, field, pct_change, threshold,
                )
                return ValidationResult.SUSPECT

    # 5. Expected range
    if value < exp_lo or value > exp_hi:
        logger.warning(
            "SUSPECT: source=%r field=%r value=%s outside expected_range=(%s, %s)",
            source, field, value, exp_lo, exp_hi,
        )
        return ValidationResult.SUSPECT

    return ValidationResult.VALID
