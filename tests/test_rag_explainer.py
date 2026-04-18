"""Tests for src/rag/explainer.py.

Uses ``unittest.mock`` to patch the Ollama chat client so no real LLM calls are made.
``NewsRetriever.get_relevant_docs`` is replaced on the instance directly.
All tests verify the ``Explanation`` contract is correctly satisfied.

Fixture: ``_pred`` — a minimal ``Prediction`` with controlled SHAP values.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.contracts import Explanation, Prediction, RetrievedSource
from src.rag.explainer import StressExplainer, _parse_llm_response, _top_shap_features

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_SHAP = {
    "bls_jobless_claims_delta": 3.5,
    "fred_delinquency_rate": 2.1,
    "news_sentiment_score": -1.2,
    "census_poverty_baseline": 0.4,
}

_MOCK_LLM_TEXT = """SUMMARY: Score rose 8 points over 7 days in Detroit-MI.
TOP_DRIVERS:
- Jobless claims delta increased significantly
- Credit card delinquency reached multi-year high
- News sentiment shifted negative
CAVEATS:
- Census data is annual and may not reflect current conditions
"""

_MOCK_SOURCES = [
    RetrievedSource(
        url="https://example.com/a1",
        title="Detroit unemployment rises",
        published_at=None,
        relevance_score=0.85,
    ),
    RetrievedSource(
        url="https://example.com/a2",
        title="Michigan poverty rate at decade high",
        published_at=None,
        relevance_score=0.72,
    ),
    RetrievedSource(
        url="https://example.com/a3",
        title="Credit defaults climb in Wayne County",
        published_at=None,
        relevance_score=0.61,
    ),
]


def _make_prediction(
    missing_sources: list[str] | None = None,
    delta_7d: float | None = 8.0,
) -> Prediction:
    """Build a minimal ``Prediction`` for use in explainer tests."""
    return Prediction(
        geo_id="Detroit-MI",
        geo_name="Detroit",
        geo_level="city",
        run_date=date(2024, 1, 15),
        ess_score=68.0,
        score_band="elevated",
        delta_7d=delta_7d,
        delta_30d=12.0,
        confidence="medium",
        early_warning=False,
        missing_sources=missing_sources or [],
        stale_sources=[],
        anomaly_flags=[],
        granularity_warning=False,
        model_version="v1.0",
        feature_version="v1.0",
        calibrated=True,
        tier1_score=38.0,
        tier2_score=19.0,
        tier3_score=11.0,
        shap_values=_SHAP,
    )


def _make_explainer(
    llm_text: str = _MOCK_LLM_TEXT,
    sources: list[RetrievedSource] | None = None,
) -> StressExplainer:
    """Build a ``StressExplainer`` with mocked LLM and retriever.

    Args:
        llm_text: Text the mock LLM will return.
        sources: Docs the mock retriever will return (default: 3 docs).

    Returns:
        ``StressExplainer`` ready for testing — no real HTTP calls made.
    """
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_docs.return_value = (
        sources if sources is not None else _MOCK_SOURCES
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = llm_text

    with patch("src.rag.explainer._build_llm", return_value=mock_llm):
        explainer = StressExplainer(retriever=mock_retriever)

    explainer._llm = mock_llm
    return explainer


# ---------------------------------------------------------------------------
# _top_shap_features unit tests (no mocking needed)
# ---------------------------------------------------------------------------


def test_top_shap_features_sorted_by_abs() -> None:
    """Top features must be sorted by absolute value, descending."""
    top = _top_shap_features(_SHAP, n=3)
    abs_values = [abs(v) for _, v in top]
    assert abs_values == sorted(abs_values, reverse=True)


def test_top_shap_features_respects_n() -> None:
    """Result must not exceed ``n`` items."""
    top = _top_shap_features(_SHAP, n=2)
    assert len(top) <= 2


def test_top_shap_features_empty_shap() -> None:
    """Empty SHAP dict must return empty list."""
    assert _top_shap_features({}, n=3) == []


# ---------------------------------------------------------------------------
# _parse_llm_response unit tests
# ---------------------------------------------------------------------------


def test_parse_llm_response_extracts_summary() -> None:
    """Parsed summary must be the single factual sentence."""
    summary, _, _ = _parse_llm_response(_MOCK_LLM_TEXT)
    assert "Score rose 8 points" in summary


def test_parse_llm_response_extracts_top_drivers() -> None:
    """Parsed top_drivers must be a list of bullet strings."""
    _, top_drivers, _ = _parse_llm_response(_MOCK_LLM_TEXT)
    assert len(top_drivers) == 3
    assert all(isinstance(d, str) and d for d in top_drivers)


def test_parse_llm_response_extracts_caveats() -> None:
    """Parsed caveats must be non-empty list of strings."""
    _, _, caveats = _parse_llm_response(_MOCK_LLM_TEXT)
    assert len(caveats) >= 1
    assert all(isinstance(c, str) and c for c in caveats)


def test_parse_llm_response_missing_sections_returns_empty() -> None:
    """Malformed response must not raise — missing sections return empty."""
    summary, top_drivers, caveats = _parse_llm_response("Some random text")
    assert isinstance(summary, str)
    assert isinstance(top_drivers, list)
    assert isinstance(caveats, list)


# ---------------------------------------------------------------------------
# StressExplainer.explain tests
# ---------------------------------------------------------------------------


def test_explain_returns_explanation_contract() -> None:
    """``explain()`` must return a valid ``Explanation`` instance."""
    explainer = _make_explainer()
    result = explainer.explain(_make_prediction())
    assert isinstance(result, Explanation)


def test_explain_top_drivers_max_3() -> None:
    """``top_drivers`` must never exceed 3 items."""
    # Give LLM text with 5 drivers to check truncation
    long_text = (
        "SUMMARY: Score rose.\n"
        "TOP_DRIVERS:\n"
        "- Driver one\n- Driver two\n- Driver three\n- Driver four\n- Driver five\n"
        "CAVEATS:\n- None identified\n"
    )
    explainer = _make_explainer(llm_text=long_text)
    result = explainer.explain(_make_prediction())
    assert len(result.top_drivers) <= 3


def test_explain_caveats_not_empty() -> None:
    """``caveats`` must never be empty — contract enforces this."""
    explainer = _make_explainer()
    result = explainer.explain(_make_prediction())
    assert len(result.caveats) >= 1


def test_caveats_populated_when_missing_sources() -> None:
    """When ``missing_sources`` is non-empty, caveats must mention missing sources."""
    explainer = _make_explainer()
    pred = _make_prediction(missing_sources=["census"])
    result = explainer.explain(pred)

    caveats_text = " ".join(result.caveats).lower()
    assert "census" in caveats_text


def test_evidence_strength_strong_when_three_docs() -> None:
    """Three retrieved docs must yield ``evidence_strength == "strong"``."""
    explainer = _make_explainer(sources=_MOCK_SOURCES)  # 3 docs
    result = explainer.explain(_make_prediction())
    assert result.evidence_strength == "strong"


def test_evidence_strength_weak_when_no_docs() -> None:
    """Zero retrieved docs must yield ``evidence_strength == "weak"``."""
    explainer = _make_explainer(sources=[])
    result = explainer.explain(_make_prediction())
    assert result.evidence_strength == "weak"


def test_evidence_strength_moderate_when_two_docs() -> None:
    """One or two retrieved docs must yield ``evidence_strength == "moderate"``."""
    explainer = _make_explainer(sources=_MOCK_SOURCES[:2])
    result = explainer.explain(_make_prediction())
    assert result.evidence_strength == "moderate"


def test_explain_survives_llm_failure() -> None:
    """LLM exception must not propagate — fallback ``Explanation`` returned."""
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_docs.return_value = []

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

    with patch("src.rag.explainer._build_llm", return_value=mock_llm):
        explainer = StressExplainer(retriever=mock_retriever)
    explainer._llm = mock_llm

    result = explainer.explain(_make_prediction())

    assert isinstance(result, Explanation)
    caveats_text = " ".join(result.caveats).lower()
    assert "confidence" in caveats_text or "coverage" in caveats_text


def test_stress_explainer_uses_ollama_config_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explainer should build its client from Ollama env settings."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "lfm2.5-thinking")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    mock_retriever = MagicMock()
    with patch("src.rag.explainer.OllamaChatClient") as mock_client:
        explainer = StressExplainer(retriever=mock_retriever)

    assert explainer._llm == mock_client.return_value
    mock_client.assert_called_once_with(
        model="lfm2.5-thinking",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=180.0,
        max_tokens=900,
    )


def test_fallback_mentions_missing_supporting_articles_when_no_docs(
) -> None:
    """Fallback explanation should explain why supporting context is empty."""
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_docs.return_value = []

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Ollama unavailable")

    with patch("src.rag.explainer._build_llm", return_value=mock_llm):
        explainer = StressExplainer(retriever=mock_retriever)
    explainer._llm = mock_llm
    result = explainer.explain(_make_prediction())

    caveats_text = " ".join(result.caveats).lower()
    assert "coverage is limited" in caveats_text or "news coverage" in caveats_text


def test_explain_falls_back_when_ollama_returns_unstructured_content() -> None:
    """A non-templated Ollama response should fall back to a valid explanation."""
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_docs.return_value = _MOCK_SOURCES[:1]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "<think>reasoning only</think>"

    with patch("src.rag.explainer._build_llm", return_value=mock_llm):
        explainer = StressExplainer(retriever=mock_retriever)
    explainer._llm = mock_llm

    result = explainer.explain(_make_prediction())

    assert isinstance(result, Explanation)
    assert len(result.top_drivers) >= 1
    assert len(result.caveats) >= 1


def test_explain_shap_breakdown_matches_prediction() -> None:
    """``shap_breakdown`` must be the prediction's full SHAP dict."""
    explainer = _make_explainer()
    pred = _make_prediction()
    result = explainer.explain(pred)
    assert result.shap_breakdown == pred.shap_values


def test_explain_geo_fields_match_prediction() -> None:
    """``geo_id``, ``geo_name``, and ``run_date`` must pass through unchanged."""
    explainer = _make_explainer()
    pred = _make_prediction()
    result = explainer.explain(pred)
    assert result.geo_id == pred.geo_id
    assert result.geo_name == pred.geo_name
    assert result.run_date == pred.run_date
