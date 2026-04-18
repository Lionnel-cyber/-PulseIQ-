"""LLM-powered structured explanation generator for PulseIQ RAG.

Combines SHAP feature attributions, retrieved news context, and an
OpenRouter chat call to produce an ``Explanation`` that satisfies the rigid
four-section template required by CLAUDE.md:

    1. SUMMARY     - one factual sentence
    2. TOP_DRIVERS - max 3 SHAP contributors in plain English
    3. EVIDENCE    - retrieved news (populated from ChromaDB, not the LLM)
    4. CAVEATS     - source gaps, staleness, weak evidence (never omitted)

The configured LLM generates the rendered briefing, but evidence is grounded in
retrieved documents and passed in as explicit bullets to minimize hallucination.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Iterator
from urllib.parse import urlparse

from openai import OpenAI

from src.contracts import Explanation, Prediction, RetrievedSource
from src.rag.retriever import NewsRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_LLM_MODEL: str = "xiaomi/mimo-v2-flash"
"""Default OpenRouter model when ``LLM_MODEL`` is not set."""

_DEFAULT_LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
"""Default OpenRouter base URL."""

_SYSTEM_PROMPT: str = (
    "You are an economic stress analyst. "
    "Respond with EXACTLY this structure, "
    "with a blank line between each section:\n\n"
    "SUMMARY\n"
    "[One sentence about the score change]\n\n"
    "TOP DRIVERS\n"
    "- [driver 1]\n"
    "- [driver 2]\n"
    "- [driver 3]\n\n"
    "EVIDENCE\n"
    "- [article title] (source)\n"
    "- [article title] (source)\n\n"
    "CAVEATS\n"
    "[One sentence or: None identified]\n\n"
    "Use only the data provided. "
    "Do not add any text outside these sections. "
    "Do not use colons after section headers."
)

_NO_SUPPORTING_ARTICLES_PROMPT: str = (
    "No supporting articles available.\n"
    "Base analysis on score data only."
)

_SECTION_RE = re.compile(
    r"^(SUMMARY|TOP DRIVERS|EVIDENCE|CAVEATS)\s*:?\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# OpenRouter helpers
# ---------------------------------------------------------------------------


@dataclass
class OpenRouterChatClient:
    """Small wrapper around OpenRouter's OpenAI-compatible chat endpoint."""

    model: str
    base_url: str
    api_key: str
    temperature: float
    timeout: float = 30.0
    max_tokens: int = 250
    default_headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key.strip()
        if not self.api_key or self.api_key == "your_key_here":
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Set a real OpenRouter key in .env "
                "and restart FastAPI."
            )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            default_headers=self.default_headers,
        )

    @staticmethod
    def _extract_content(response: Any) -> str | None:
        """Extract the assistant text content from a chat completion response."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None

        message = choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)
            joined = "".join(text_parts).strip()
            if joined:
                return joined
            return None
        return None

    def invoke(self, messages: list[dict[str, str]]) -> str:
        """Submit a non-streaming chat request and return the response text."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"reasoning": {"enabled": False}},
        )
        content = self._extract_content(response)
        if content:
            return content

        finish_reason = None
        choices = getattr(response, "choices", None) or []
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
        raise RuntimeError(
            "OpenRouter returned no message content "
            f"(finish_reason={finish_reason!r})."
        )

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream assistant content chunks from OpenRouter."""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            extra_body={"reasoning": {"enabled": False}},
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        yield text


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _top_shap_features(shap_values: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
    """Return the top-N SHAP features by absolute contribution."""
    return sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]


def _truncate_prompt_text(text: str, limit: int = 100) -> str:
    """Trim prompt text aggressively to keep thinking-model prompts small."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _format_prompt_docs(docs: list[RetrievedSource]) -> str:
    """Format up to three retrieved article titles for the prompt."""
    if not docs:
        return _NO_SUPPORTING_ARTICLES_PROMPT

    lines = [
        f"  [{i + 1}] {_truncate_prompt_text(doc.title, 100)}"
        for i, doc in enumerate(docs[:3])
    ]
    return "\n".join(lines)


def _sanitize_docs(retrieved_docs: list[RetrievedSource]) -> list[RetrievedSource]:
    """Drop empty URLs and placeholder docs before prompting the LLM."""
    return [
        doc
        for doc in retrieved_docs
        if doc.url and "example.com" not in doc.url.lower()
    ]


def _source_label_from_url(url: str) -> str:
    """Derive a compact source label from a retrieved document URL."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown source"


def _format_evidence_lines(docs: list[RetrievedSource]) -> str:
    """Format up to three evidence bullets for the prompt."""
    if not docs:
        return _NO_SUPPORTING_ARTICLES_PROMPT

    lines = []
    for doc in docs[:3]:
        title = _truncate_prompt_text(doc.title, 100)
        source = _truncate_prompt_text(_source_label_from_url(doc.url), 60)
        lines.append(f"- {title} ({source})")
    return "\n".join(lines)


def _format_driver_hints(top_shap: list[tuple[str, float]]) -> str:
    """Format plain-English driver cues from top SHAP contributions."""
    if not top_shap:
        return "- No strong driver identified"

    lines = []
    for feature_name, value in top_shap[:3]:
        label = _humanize_feature_name(feature_name).capitalize()
        direction = "raised" if value >= 0 else "softened"
        lines.append(f"- {label} {direction} the score")
    return "\n".join(lines)


def _build_user_message(
    prediction: Prediction,
    top_shap: list[tuple[str, float]],
    docs: list[RetrievedSource],
) -> str:
    """Compose the user message sent to the LLM."""
    delta_str = (
        f"{prediction.delta_7d:+.1f} points over 7 days"
        if prediction.delta_7d is not None
        else "delta unknown (first run)"
    )

    doc_lines = _format_prompt_docs(docs[:3])
    driver_hints = _format_driver_hints(top_shap)
    evidence_lines = _format_evidence_lines(docs[:3])

    missing_str = (
        ", ".join(prediction.missing_sources)
        if prediction.missing_sources
        else "none"
    )

    return (
        f"Geography: {prediction.geo_name} ({prediction.geo_id})\n"
        f"ESS Score: {prediction.ess_score:.1f} / 100\n"
        f"Change: {delta_str}\n"
        f"Confidence: {prediction.confidence}\n"
        f"Missing sources: {missing_str}\n\n"
        f"TOP DRIVERS bullets to copy exactly:\n{driver_hints}\n\n"
        f"Retrieved news context:\n{doc_lines}\n\n"
        f"EVIDENCE bullets to copy exactly when relevant:\n"
        f"{evidence_lines}"
    )


def _parse_llm_response(text: str) -> tuple[str, list[str], list[str]]:
    """Parse the LLM response into ``(summary, top_drivers, caveats)``."""
    parts = _SECTION_RE.split(text)
    sections: dict[str, str] = {}
    i = 1
    while i + 1 < len(parts):
        marker = parts[i].strip().upper()
        body = parts[i + 1].strip()
        sections[marker] = body
        i += 2

    summary_body = sections.get("SUMMARY", "")
    summary = next(
        (line.strip() for line in summary_body.splitlines() if line.strip()),
        "Score change explanation unavailable.",
    )

    def _extract_bullets(body: str) -> list[str]:
        lines = [
            line.lstrip("-* \t").strip()
            for line in body.splitlines()
            if line.strip().startswith(("-", "*"))
        ]
        return [line for line in lines if line]

    top_drivers = _extract_bullets(sections.get("TOP DRIVERS", ""))[:3]

    caveat_body = sections.get("CAVEATS", "")
    caveat_bullets = _extract_bullets(caveat_body)
    if caveat_bullets:
        caveats = caveat_bullets
    else:
        caveats = [
            line.strip()
            for line in caveat_body.splitlines()
            if line.strip()
        ]

    return summary, top_drivers, caveats


def _missing_source_caveats(missing_sources: list[str]) -> list[str]:
    """Convert missing source names into human-readable caveats."""
    labels = {
        "bls": "BLS jobless claims data unavailable - Tier 1 signal missing",
        "fred": "FRED economic indicators unavailable - Tier 1/2 signals missing",
        "census": "Census poverty/income data unavailable - Tier 1 baseline missing",
        "news": "News sentiment data unavailable - Tier 3 signal zeroed",
        "trends": "Google Trends data unavailable - Tier 3 signal zeroed",
        "openweather": "Weather stress data unavailable - Tier 3 signal zeroed",
    }
    return [
        labels.get(src, f"Data source '{src}' unavailable")
        for src in missing_sources
    ]


def _humanize_feature_name(feature_name: str) -> str:
    """Convert a model feature name into a readable label."""
    labels = {
        "bls_jobless_claims_delta": "jobless claims",
        "bls_unemployment_rate": "unemployment rate",
        "fred_delinquency_rate": "credit delinquency",
        "census_poverty_baseline": "poverty baseline",
        "census_median_income": "median income",
        "fred_cpi_delta": "inflation",
        "fred_mortgage_rate_delta": "mortgage rates",
        "trends_search_score": "distress-related search activity",
        "trends_search_delta": "search momentum",
        "news_sentiment_score": "news sentiment",
        "news_article_count": "news volume",
        "openweather_severity_index": "weather stress",
        "data_quality_score": "data quality coverage",
        "stale_source_count": "stale-source count",
    }
    return labels.get(feature_name, feature_name.replace("_", " "))


def _fallback_top_drivers(top_shap: list[tuple[str, float]]) -> list[str]:
    """Build plain-English driver bullets from top SHAP features."""
    drivers: list[str] = []
    for feature_name, value in top_shap[:3]:
        label = _humanize_feature_name(feature_name)
        direction = "raised" if value >= 0 else "softened"
        drivers.append(f"{label.capitalize()} {direction} the score.")
    return drivers


def _build_llm(model_name: str | None = None) -> OpenRouterChatClient:
    """Return an OpenRouter-backed chat client."""
    resolved_model = model_name or os.getenv("LLM_MODEL") or _DEFAULT_LLM_MODEL
    temperature = float(os.getenv("LLM_TEMPERATURE") or "0.1")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS") or "30")
    base_url = os.getenv("LLM_BASE_URL") or _DEFAULT_LLM_BASE_URL
    api_key = os.getenv("OPENROUTER_API_KEY") or ""

    return OpenRouterChatClient(
        model=resolved_model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_tokens=250,
        default_headers={
            "HTTP-Referer": "https://pulseiq.app",
            "X-Title": "PulseIQ",
        },
    )


# ---------------------------------------------------------------------------
# StressExplainer
# ---------------------------------------------------------------------------


class StressExplainer:
    """Generate structured ``Explanation`` objects from scored predictions."""

    def __init__(
        self,
        retriever: NewsRetriever | None = None,
        model_name: str | None = None,
    ) -> None:
        self._retriever: NewsRetriever = retriever or NewsRetriever()
        self._llm = _build_llm(model_name)

    def _build_system_prompt(self) -> str:
        """Return the exact system prompt string sent to the LLM."""
        return _SYSTEM_PROMPT

    def _get_docs(
        self,
        prediction: Prediction,
        top_k: int = 3,
    ) -> list[RetrievedSource]:
        """Retrieve a small set of supporting news docs for prompt construction."""
        retrieved_docs = self._retriever.get_relevant_docs(
            geo_id=prediction.geo_id,
            geo_name=prediction.geo_name,
            top_k=top_k,
        )
        return _sanitize_docs(retrieved_docs)

    def _build_messages(
        self,
        prediction: Prediction,
        top_shap: list[tuple[str, float]],
        docs: list[RetrievedSource],
    ) -> list[dict[str, str]]:
        """Build the system/user message list sent to OpenRouter."""
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": _build_user_message(prediction, top_shap, docs)},
        ]

    @staticmethod
    def _render_stream_text(explanation: Explanation) -> str:
        """Render a canonical four-section text block for streaming."""
        evidence_lines = _format_evidence_lines(explanation.retrieved_sources)
        if evidence_lines == _NO_SUPPORTING_ARTICLES_PROMPT:
            evidence_block = "No supporting articles available."
        else:
            evidence_block = evidence_lines

        drivers = explanation.top_drivers[:3]
        while len(drivers) < 3:
            drivers.append("No additional driver identified")
        driver_block = "\n".join(
            f"- {driver.lstrip('- ').strip()}" for driver in drivers[:3]
        )

        caveat_text = next(
            (c.strip() for c in explanation.caveats if c.strip()),
            "None identified",
        )

        return (
            "SUMMARY\n"
            f"{explanation.summary.strip()}\n\n"
            "TOP DRIVERS\n"
            f"{driver_block}\n\n"
            "EVIDENCE\n"
            f"{evidence_block}\n\n"
            "CAVEATS\n"
            f"{caveat_text}"
        )

    def explain(self, prediction: Prediction) -> Explanation:
        """Generate a structured explanation for a scored prediction."""
        top_shap = _top_shap_features(prediction.shap_values)

        docs = self._get_docs(prediction, top_k=3)

        if len(docs) >= 3:
            evidence_strength = "strong"
        elif len(docs) >= 1:
            evidence_strength = "moderate"
        else:
            evidence_strength = "weak"

        try:
            raw_text = self._llm.invoke(self._build_messages(prediction, top_shap, docs))
            if not all(
                marker in raw_text
                for marker in ("SUMMARY", "TOP DRIVERS", "EVIDENCE", "CAVEATS")
            ):
                logger.warning(
                    "StressExplainer: OpenRouter returned no structured answer for geo_id=%s",
                    prediction.geo_id,
                )
                return self._fallback_explanation(
                    prediction,
                    docs,
                    evidence_strength,
                )
            summary, top_drivers, llm_caveats = _parse_llm_response(raw_text)
        except Exception as exc:
            logger.error(
                "StressExplainer: LLM call failed for geo_id=%s: %s",
                prediction.geo_id,
                exc,
            )
            return self._fallback_explanation(
                prediction,
                docs,
                evidence_strength,
            )

        source_caveats = _missing_source_caveats(prediction.missing_sources)
        all_caveats = source_caveats + [c for c in llm_caveats if c]
        if not all_caveats:
            all_caveats = ["None identified"]

        if not top_drivers and top_shap:
            top_drivers = [f"{name} contribution: {val:+.3f}" for name, val in top_shap]

        return Explanation(
            geo_id=prediction.geo_id,
            geo_name=prediction.geo_name,
            run_date=prediction.run_date,
            summary=summary,
            top_drivers=top_drivers[:3],
            shap_breakdown=prediction.shap_values,
            retrieved_sources=docs,
            evidence_strength=evidence_strength,
            confidence=prediction.confidence,
            missing_sources=prediction.missing_sources,
            caveats=all_caveats,
            generated_at=datetime.now(tz=timezone.utc),
        )

    async def explain_stream(self, prediction: Prediction) -> AsyncGenerator[str, None]:
        """Stream stable section blocks instead of raw token fragments."""
        explanation = self.explain(prediction)
        rendered = self._render_stream_text(explanation)

        for block in rendered.split("\n\n"):
            yield block
            await asyncio.sleep(0)

    def _fallback_explanation(
        self,
        prediction: Prediction,
        docs: list[RetrievedSource],
        evidence_strength: str,
    ) -> Explanation:
        """Return a minimal valid explanation when the LLM call fails."""
        delta_str = (
            f"{prediction.delta_7d:+.1f} pts over 7 days"
            if prediction.delta_7d is not None
            else "delta unknown"
        )
        top_drivers = _fallback_top_drivers(_top_shap_features(prediction.shap_values))
        caveats = _missing_source_caveats(prediction.missing_sources)
        if prediction.granularity_warning:
            caveats.append(
                "Some signals are inferred from broader regional or national proxies."
            )
        if prediction.confidence != "high":
            caveats.append(
                f"Confidence is {prediction.confidence}; treat small score moves cautiously."
            )
        if evidence_strength == "weak":
            caveats.append(
                "Recent supporting news coverage is limited for this geography."
            )
        elif evidence_strength == "moderate":
            caveats.append(
                "Supporting evidence is based on a small number of recent articles."
            )
        if not caveats:
            caveats = ["No major source gaps identified."]

        return Explanation(
            geo_id=prediction.geo_id,
            geo_name=prediction.geo_name,
            run_date=prediction.run_date,
            summary=(
                f"Score of {prediction.ess_score:.1f} recorded for"
                f" {prediction.geo_name} ({delta_str})."
            ),
            top_drivers=top_drivers,
            shap_breakdown=prediction.shap_values,
            retrieved_sources=docs,
            evidence_strength=evidence_strength,
            confidence=prediction.confidence,
            missing_sources=prediction.missing_sources,
            caveats=caveats,
            generated_at=datetime.now(tz=timezone.utc),
        )
