"""Streamlit dashboard for PulseIQ."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from dashboard.components.explanation import render_explanation
from dashboard.components.health import render_health_page
from dashboard.components.map import render_map
from dashboard.components.trend import render_trend

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_DEFAULT_API_BASE = os.getenv("PULSEIQ_API_BASE_URL", "http://localhost:8000/api/v1")
_REQUEST_TIMEOUT = 15
_EXPLANATION_TIMEOUT = 90


def _api_get(
    api_base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Any:
    """Call one FastAPI endpoint and return the decoded JSON payload."""
    url = f"{api_base_url.rstrip('/')}{path}"
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.Timeout as exc:
        raise RuntimeError(
            "PulseIQ API request timed out. If this happened while generating an "
            "explanation, the local embedding model may still be loading or the "
            "provider call may be slow. Please try again."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach the PulseIQ API. Start it with "
            "`uvicorn src.api.main:app --reload`."
        ) from exc

    if response.status_code >= 400:
        detail = None
        try:
            payload = response.json()
            detail = payload.get("detail")
        except ValueError:
            detail = response.text
        raise RuntimeError(detail or f"API request failed with status {response.status_code}.")

    return response.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_map_scores(api_base_url: str, run_date: date) -> list[dict[str, Any]]:
    """Fetch state-level map scores for one snapshot date."""
    return _api_get(api_base_url, "/scores/map", {"run_date": run_date.isoformat()})


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_search_results(
    api_base_url: str,
    query: str,
    as_of: date,
) -> list[dict[str, Any]]:
    """Fetch matching geographies for the sidebar search box."""
    return _api_get(
        api_base_url,
        "/scores/search",
        {"q": query, "as_of": as_of.isoformat(), "limit": 25},
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(
    api_base_url: str,
    geo_id: str,
    window: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Fetch one score history series for the selected geography."""
    return _api_get(
        api_base_url,
        f"/scores/{geo_id}/history",
        {
            "window": window,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_score(api_base_url: str, geo_id: str) -> dict[str, Any]:
    """Fetch the latest score payload for one geography."""
    return _api_get(api_base_url, f"/scores/{geo_id}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dashboard_health(api_base_url: str) -> dict[str, Any]:
    """Fetch the combined dashboard health payload."""
    return _api_get(api_base_url, "/health/dashboard")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_explanation(
    api_base_url: str,
    geo_id: str,
    refresh_token: int,
) -> dict[str, Any]:
    """Fetch the structured explanation payload for one geography."""
    _ = refresh_token
    return _api_get(api_base_url, f"/explain/{geo_id}", timeout=_EXPLANATION_TIMEOUT)


def _normalize_date_range(value: Any) -> tuple[date, date]:
    """Normalize Streamlit's date_input return shape to a two-date tuple."""
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, date):
        return value, value

    today = date.today()
    return today - timedelta(days=30), today


def _show_api_error(message: str) -> None:
    """Render a clear API failure panel with startup instructions."""
    st.error(message)
    st.markdown("Start the API in the project root:")
    st.code("uvicorn src.api.main:app --reload")
    st.markdown(
        "If the API starts but returns empty data, run the scoring pipeline first "
        "and confirm `DUCKDB_PATH` points to the DuckDB file that contains `ess_scores`."
    )


def _selected_geo_label(api_base_url: str, geo_id: str) -> str:
    """Return a readable label for the selected geography."""
    try:
        score = fetch_score(api_base_url, geo_id)
    except Exception:
        return geo_id
    return f"{score.get('geo_name', geo_id)} ({geo_id})"


def main() -> None:
    """Render the PulseIQ dashboard."""
    st.set_page_config(
        page_title="PulseIQ — Economic Stress Monitor",
        layout="wide",
    )
    st.title("PulseIQ — Economic Stress Monitor")

    api_base_url = _DEFAULT_API_BASE
    today = date.today()
    default_range = (today - timedelta(days=30), today)

    with st.sidebar:
        page = st.radio("Page", ["Monitor", "Pipeline Health"])
        geo_query = st.text_input("Geo search")
        raw_range = st.date_input("Date range", value=default_range)
        start_date, end_date = _normalize_date_range(raw_range)
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"API: {api_base_url}")

        if geo_query.strip():
            try:
                matches = fetch_search_results(api_base_url, geo_query.strip(), end_date)
            except Exception as exc:
                _show_api_error(str(exc))
                st.stop()

            if matches:
                labels = [f"{item['geo_name']} ({item['geo_id']})" for item in matches]
                label_to_id = {
                    f"{item['geo_name']} ({item['geo_id']})": item["geo_id"]
                    for item in matches
                }
                current_geo = st.session_state.get("selected_geo_id")
                current_label = next(
                    (label for label, geo_id in label_to_id.items() if geo_id == current_geo),
                    labels[0],
                )
                selected_label = st.selectbox(
                    "Search results",
                    options=labels,
                    index=labels.index(current_label),
                )
                st.session_state["selected_geo_id"] = label_to_id[selected_label]
            else:
                st.caption("No geographies matched that search.")

    if page == "Pipeline Health":
        try:
            health = fetch_dashboard_health(api_base_url)
        except Exception as exc:
            _show_api_error(str(exc))
            st.stop()

        render_health_page(health)
        return

    try:
        map_rows = fetch_map_scores(api_base_url, end_date)
    except Exception as exc:
        _show_api_error(str(exc))
        st.stop()

    clicked_geo_id = render_map(map_rows)
    if clicked_geo_id:
        st.session_state["selected_geo_id"] = clicked_geo_id

    if "selected_geo_id" not in st.session_state and map_rows:
        st.session_state["selected_geo_id"] = map_rows[0]["drilldown_geo_id"]

    selected_geo_id = st.session_state.get("selected_geo_id")
    if not selected_geo_id:
        st.info("Select a geography from the map or sidebar search to view its details.")
        return

    st.caption(f"Selected geography: {_selected_geo_label(api_base_url, selected_geo_id)}")

    trend_col, explanation_col = st.columns([1.35, 1])
    with trend_col:
        render_trend(
            selected_geo_id,
            start_date,
            end_date,
            lambda geo_id, window, start_date, end_date: fetch_history(
                api_base_url,
                geo_id,
                window,
                start_date,
                end_date,
            ),
        )
    with explanation_col:
        render_explanation(
            selected_geo_id,
            lambda geo_id, refresh_token: fetch_explanation(
                api_base_url, geo_id, refresh_token
            ),
        )


if __name__ == "__main__":
    main()
