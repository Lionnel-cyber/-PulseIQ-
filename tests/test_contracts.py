"""Tests for src/contracts.py.

Covers the four behavioural guarantees called out in the task:
1. FeatureVector raises ValidationError on any None feature field.
2. Prediction rejects ess_score outside 0–100.
3. Explanation rejects top_drivers with more than 3 items.
4. ScoreResponse.is_trustworthy is True only when all four criteria are met.

All tests are pure Python — no network calls, no database, no fixtures.
"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.contracts import (
    Explanation,
    Prediction,
    RetrievedSource,
    ScoreResponse,
    FeatureVector,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TODAY = date.today()
NOW = datetime.utcnow()


def _make_feature_kwargs() -> dict:
    """Return a complete, valid set of FeatureVector constructor kwargs.

    Individual tests copy this dict and override one field to trigger
    the specific failure they are testing.
    """
    return {
        "geo_id": "Detroit-MI",
        "geo_level": "city",
        "run_date": TODAY,
        # Tier 1
        "bls_jobless_claims_delta": 0.12,
        "bls_unemployment_rate": 4.2,
        "fred_delinquency_rate": 2.8,
        "census_poverty_baseline": 14.1,
        "census_median_income": 52_000.0,
        # Tier 2
        "fred_cpi_delta": 0.3,
        "fred_mortgage_rate_delta": -0.05,
        # Tier 3
        "trends_search_score": 68.0,
        "trends_search_delta": 5.0,
        "news_sentiment_score": 0.25,
        "news_article_count": 12,
        "openweather_severity_index": 0.0,
        # Observability
        "data_quality_score": 0.93,
        "stale_source_count": 0,
    }


def _make_prediction_kwargs(**overrides) -> dict:
    """Return a complete, valid set of Prediction / ScoreResponse kwargs.

    Pass keyword arguments to override specific fields before construction.
    """
    base = {
        "geo_id": "Detroit-MI",
        "geo_name": "Detroit, MI",
        "geo_level": "city",
        "run_date": TODAY,
        "ess_score": 72.0,
        "score_band": "high",
        "delta_7d": 3.5,
        "delta_30d": -1.2,
        "confidence": "high",
        "early_warning": False,
        "missing_sources": [],
        "stale_sources": [],
        "anomaly_flags": [],
        "granularity_warning": False,
        "model_version": "v1.0.0",
        "feature_version": "v1.0.0",
        "calibrated": True,
        "tier1_score": 0.42,
        "tier2_score": 0.18,
        "tier3_score": 0.08,
        "shap_values": {"bls_jobless_claims_delta": 0.35, "fred_delinquency_rate": 0.20},
    }
    base.update(overrides)
    return base


def _make_explanation_kwargs(**overrides) -> dict:
    """Return a complete, valid set of Explanation constructor kwargs."""
    base = {
        "geo_id": "Detroit-MI",
        "geo_name": "Detroit, MI",
        "run_date": TODAY,
        "summary": "Score rose 8 points over 7 days driven by rising delinquency.",
        "top_drivers": ["Rising delinquency rate", "Increase in jobless claims"],
        "shap_breakdown": {"fred_delinquency_rate": 0.20, "bls_jobless_claims_delta": 0.15},
        "retrieved_sources": [
            RetrievedSource(
                url="https://example.com/article",
                title="Detroit layoffs spike",
                published_at=NOW,
                relevance_score=0.91,
            )
        ],
        "evidence_strength": "strong",
        "confidence": "high",
        "missing_sources": [],
        "caveats": ["BLS and FRED data are national-level proxies for this city."],
        "generated_at": NOW,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. FeatureVector — raises on None feature field
# ---------------------------------------------------------------------------


def test_feature_vector_valid_construction() -> None:
    """FeatureVector constructs without error given all valid fields."""
    fv = FeatureVector(**_make_feature_kwargs())
    assert fv.geo_id == "Detroit-MI"
    assert fv.bls_unemployment_rate == pytest.approx(4.2)


def test_feature_vector_raises_on_none_feature() -> None:
    """FeatureVector raises ValidationError naming the offending field."""
    kwargs = _make_feature_kwargs()
    kwargs["bls_unemployment_rate"] = None
    with pytest.raises(ValidationError, match="bls_unemployment_rate"):
        FeatureVector(**kwargs)


def test_feature_vector_raises_on_multiple_none_features() -> None:
    """ValidationError message lists all None fields, not just the first."""
    kwargs = _make_feature_kwargs()
    kwargs["bls_unemployment_rate"] = None
    kwargs["fred_delinquency_rate"] = None
    with pytest.raises(ValidationError) as exc_info:
        FeatureVector(**kwargs)
    error_str = str(exc_info.value)
    assert "bls_unemployment_rate" in error_str
    assert "fred_delinquency_rate" in error_str


def test_feature_vector_raises_on_missing_feature_field() -> None:
    """FeatureVector raises if a required feature field is absent entirely."""
    kwargs = _make_feature_kwargs()
    del kwargs["trends_search_score"]
    with pytest.raises(ValidationError):
        FeatureVector(**kwargs)


def test_feature_vector_data_quality_score_bounds() -> None:
    """data_quality_score outside 0–1 raises ValidationError."""
    kwargs = _make_feature_kwargs()
    kwargs["data_quality_score"] = 1.1
    with pytest.raises(ValidationError):
        FeatureVector(**kwargs)


# ---------------------------------------------------------------------------
# 2. Prediction — ess_score must be 0–100
# ---------------------------------------------------------------------------


def test_prediction_valid_construction() -> None:
    """Prediction constructs without error given all valid fields."""
    pred = Prediction(**_make_prediction_kwargs())
    assert pred.ess_score == pytest.approx(72.0)


def test_prediction_rejects_ess_score_above_100() -> None:
    """ess_score > 100 raises ValidationError."""
    with pytest.raises(ValidationError):
        Prediction(**_make_prediction_kwargs(ess_score=100.1))


def test_prediction_rejects_negative_ess_score() -> None:
    """ess_score < 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        Prediction(**_make_prediction_kwargs(ess_score=-0.1))


def test_prediction_accepts_boundary_ess_scores() -> None:
    """ess_score values of exactly 0 and 100 are both valid."""
    pred_zero = Prediction(**_make_prediction_kwargs(ess_score=0.0))
    pred_max = Prediction(**_make_prediction_kwargs(ess_score=100.0))
    assert pred_zero.ess_score == pytest.approx(0.0)
    assert pred_max.ess_score == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 3. Explanation — top_drivers max 3 items
# ---------------------------------------------------------------------------


def test_explanation_valid_construction() -> None:
    """Explanation constructs without error given all valid fields."""
    exp = Explanation(**_make_explanation_kwargs())
    assert len(exp.top_drivers) == 2


def test_explanation_rejects_four_top_drivers() -> None:
    """top_drivers with 4 items raises ValidationError."""
    with pytest.raises(ValidationError):
        Explanation(**_make_explanation_kwargs(
            top_drivers=["driver A", "driver B", "driver C", "driver D"]
        ))


def test_explanation_accepts_exactly_three_top_drivers() -> None:
    """top_drivers with exactly 3 items is valid."""
    exp = Explanation(**_make_explanation_kwargs(
        top_drivers=["driver A", "driver B", "driver C"]
    ))
    assert len(exp.top_drivers) == 3


def test_explanation_rejects_empty_caveats() -> None:
    """caveats=[] raises ValidationError — caveats section is never omitted."""
    with pytest.raises(ValidationError, match="caveats"):
        Explanation(**_make_explanation_kwargs(caveats=[]))


def test_explanation_accepts_none_identified_caveats() -> None:
    """Caller may pass ["None identified"] when no caveats apply."""
    exp = Explanation(**_make_explanation_kwargs(caveats=["None identified"]))
    assert exp.caveats == ["None identified"]


# ---------------------------------------------------------------------------
# 4. ScoreResponse.is_trustworthy logic
# ---------------------------------------------------------------------------


def test_score_response_is_trustworthy_true_when_all_criteria_met() -> None:
    """is_trustworthy is True when confidence=high, calibrated, no warnings."""
    sr = ScoreResponse(**_make_prediction_kwargs(
        confidence="high",
        calibrated=True,
        granularity_warning=False,
        anomaly_flags=[],
    ))
    assert sr.is_trustworthy is True


def test_score_response_not_trustworthy_when_medium_confidence() -> None:
    """is_trustworthy is False when confidence is not 'high'."""
    sr = ScoreResponse(**_make_prediction_kwargs(
        confidence="medium",
        calibrated=True,
        granularity_warning=False,
        anomaly_flags=[],
    ))
    assert sr.is_trustworthy is False


def test_score_response_not_trustworthy_when_low_confidence() -> None:
    """is_trustworthy is False when confidence is 'low'."""
    sr = ScoreResponse(**_make_prediction_kwargs(
        confidence="low",
        calibrated=True,
        granularity_warning=False,
        anomaly_flags=[],
    ))
    assert sr.is_trustworthy is False


def test_score_response_not_trustworthy_when_not_calibrated() -> None:
    """is_trustworthy is False when calibrated is False."""
    sr = ScoreResponse(**_make_prediction_kwargs(
        confidence="high",
        calibrated=False,
        granularity_warning=False,
        anomaly_flags=[],
    ))
    assert sr.is_trustworthy is False


def test_score_response_not_trustworthy_when_granularity_warning() -> None:
    """is_trustworthy is False when granularity_warning is True."""
    sr = ScoreResponse(**_make_prediction_kwargs(
        confidence="high",
        calibrated=True,
        granularity_warning=True,
        anomaly_flags=[],
    ))
    assert sr.is_trustworthy is False


def test_score_response_not_trustworthy_when_anomaly_flags_present() -> None:
    """is_trustworthy is False when anomaly_flags is non-empty."""
    sr = ScoreResponse(**_make_prediction_kwargs(
        confidence="high",
        calibrated=True,
        granularity_warning=False,
        anomaly_flags=["FRED_CPI_SPIKE"],
    ))
    assert sr.is_trustworthy is False


def test_score_response_is_trustworthy_serialised_in_model_dump() -> None:
    """is_trustworthy appears in model_dump() output (computed_field)."""
    sr = ScoreResponse(**_make_prediction_kwargs(
        confidence="high",
        calibrated=True,
        granularity_warning=False,
        anomaly_flags=[],
    ))
    dumped = sr.model_dump()
    assert "is_trustworthy" in dumped
    assert dumped["is_trustworthy"] is True
