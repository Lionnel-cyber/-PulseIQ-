"""Tests for src/validation/rules.py.

Pure computation — no HTTP calls, no fixtures, no mocking required.
Each test exercises one branch of the validate() decision tree.
"""

import pytest

from src.validation.rules import ValidationResult, validate

# ---------------------------------------------------------------------------
# BLS jobless_claims
# ---------------------------------------------------------------------------


def test_bls_claims_zero_is_rejected() -> None:
    """Claims of exactly 0 are rejected — BLS never reports zero, only stops publishing."""
    assert validate("bls", "jobless_claims", 0) == ValidationResult.REJECTED


def test_bls_claims_normal_is_valid() -> None:
    """A typical weekly claims value in the expected range is valid."""
    assert validate("bls", "jobless_claims", 250_000) == ValidationResult.VALID


def test_bls_claims_outside_hard_limits_is_rejected() -> None:
    """A value above the hard limit (10 M) is rejected outright."""
    assert validate("bls", "jobless_claims", 15_000_000) == ValidationResult.REJECTED


def test_bls_claims_spike_is_suspect() -> None:
    """A week-on-week spike of >300 % is suspect.

    900_000 vs 200_000 → 350 % change, exceeds spike_threshold_pct=300.
    """
    assert validate("bls", "jobless_claims", 900_000, previous_value=200_000) == ValidationResult.SUSPECT


def test_bls_claims_spike_below_threshold_is_valid() -> None:
    """A spike just under the 300 % threshold is valid.

    600_000 vs 200_000 → 200 % change, below spike_threshold_pct=300.
    """
    assert validate("bls", "jobless_claims", 600_000, previous_value=200_000) == ValidationResult.VALID


# ---------------------------------------------------------------------------
# FRED CPI
# ---------------------------------------------------------------------------


def test_cpi_normal_value_is_valid() -> None:
    """A CPI value squarely in the expected range (200–400) is valid."""
    assert validate("fred", "cpi", 310.0) == ValidationResult.VALID


def test_cpi_outside_hard_limits_is_rejected() -> None:
    """A CPI above the hard limit (1 000) is rejected."""
    assert validate("fred", "cpi", 1_500.0) == ValidationResult.REJECTED


def test_cpi_monthly_delta_just_under_threshold_is_valid() -> None:
    """A monthly CPI delta just below 5 % is valid.

    325.0 vs 310.0 → ~4.84 % change, below max_monthly_delta_pct=5.
    """
    assert validate("fred", "cpi", 325.0, previous_value=310.0) == ValidationResult.VALID


def test_cpi_monthly_delta_over_threshold_is_suspect() -> None:
    """A monthly CPI delta above 5 % is suspect — flag but do not reject.

    327.0 vs 310.0 → ~5.48 % change, exceeds max_monthly_delta_pct=5.
    """
    assert validate("fred", "cpi", 327.0, previous_value=310.0) == ValidationResult.SUSPECT


# ---------------------------------------------------------------------------
# News sentiment_score
# ---------------------------------------------------------------------------


def test_news_normal_sentiment_is_valid() -> None:
    """Sentiment scores within the expected range with no significant delta are valid."""
    assert validate("news", "sentiment_score", 0.3, previous_value=0.25) == ValidationResult.VALID


def test_news_spike_10x_is_suspect() -> None:
    """A sentiment spike of >1 000 % is suspect (possible bot wave or viral story).

    0.6 vs 0.05 → 1 100 % change, exceeds spike_threshold_pct=1 000.
    """
    assert validate("news", "sentiment_score", 0.6, previous_value=0.05) == ValidationResult.SUSPECT


# ---------------------------------------------------------------------------
# Unknown source / field — passthrough
# ---------------------------------------------------------------------------


def test_unknown_source_is_valid() -> None:
    """Values from a source with no rules are passed through as valid."""
    assert validate("openweather", "temperature", 72.0) == ValidationResult.VALID


def test_unknown_field_is_valid() -> None:
    """Values for a field not in the rules dict are passed through as valid."""
    assert validate("bls", "nonexistent_field", 0.0) == ValidationResult.VALID


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_spike_check_skipped_when_previous_is_zero() -> None:
    """Spike check is skipped entirely when previous_value == 0 (avoids division by zero).

    Value 50_000 is within expected_range (1_000–1_000_000) → VALID.
    """
    assert validate("bls", "jobless_claims", 50_000, previous_value=0) == ValidationResult.VALID


def test_value_below_hard_limit_is_rejected() -> None:
    """A delinquency rate below its hard lower limit (0.0) is rejected."""
    assert validate("fred", "delinquency_rate", -1.0) == ValidationResult.REJECTED


def test_no_previous_value_skips_spike_check() -> None:
    """When previous_value is None the spike check is skipped; value judged on range alone."""
    # 1_500_000 is above expected_range (1M) but below hard_limit (10M) → SUSPECT
    assert validate("bls", "jobless_claims", 1_500_000) == ValidationResult.SUSPECT
