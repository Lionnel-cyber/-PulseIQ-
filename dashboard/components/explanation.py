"""Explanation component for the PulseIQ dashboard."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

_BADGE_STYLE = {
    "strong": ("#1a9850", "green"),
    "moderate": ("#f39c12", "orange"),
    "weak": ("#c0392b", "red"),
}


def render_explanation(
    geo_id: str,
    fetch_explanation: Callable[[str, int], dict[str, Any]],
    key_prefix: str = "pulseiq-explanation",
) -> None:
    """Render the structured explanation panel for the selected geography."""
    token_key = f"{key_prefix}-{geo_id}-refresh"
    button_key = f"{key_prefix}-{geo_id}-button"

    if token_key not in st.session_state:
        st.session_state[token_key] = 0

    if st.button("Generate explanation", key=button_key):
        st.session_state[token_key] += 1

    refresh_token = st.session_state[token_key]
    if refresh_token == 0:
        st.info("Generate explanation to load the structured four-section explanation.")
        return

    try:
        explanation = fetch_explanation(geo_id, refresh_token)
    except Exception as exc:
        st.error(str(exc))
        return
    summary = explanation.get("summary") or "Explanation unavailable"
    drivers = list(explanation.get("top_drivers") or [])[:3]
    sources = list(explanation.get("retrieved_sources") or [])
    caveats = list(explanation.get("caveats") or [])
    evidence_strength = str(explanation.get("evidence_strength") or "weak")
    generated_at = explanation.get("generated_at") or "unknown"

    badge_color, badge_label = _BADGE_STYLE.get(
        evidence_strength,
        ("#c0392b", "red"),
    )

    st.markdown(f"### {summary}")
    st.markdown(
        (
            f"<span style='display:inline-block;padding:0.2rem 0.55rem;"
            f"border-radius:999px;background:{badge_color};color:white;"
            f"font-size:0.85rem;font-weight:600;'>Evidence: {badge_label}</span>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("**Top model drivers**")
    if drivers:
        for driver in drivers:
            st.markdown(f"- {driver}")
    else:
        st.markdown("- No model drivers returned.")

    st.markdown("**Supporting context**")
    if sources:
        for source in sources:
            title = source.get("title") or source.get("url") or "Untitled source"
            url = source.get("url") or "#"
            st.markdown(f"- [{title}]({url})")
    else:
        st.markdown("- No supporting articles returned.")

    st.warning("\n".join(caveats) if caveats else "No caveats identified")
    st.caption(f"Generated at {generated_at}")
