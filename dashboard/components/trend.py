"""Trend component for the PulseIQ dashboard."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_ALERT_THRESHOLD: float = 75.0


def _low_confidence_ranges(points: list[dict[str, Any]]) -> list[tuple[date, date]]:
    """Return contiguous low-confidence date ranges."""
    ranges: list[tuple[date, date]] = []
    start: date | None = None
    previous: date | None = None

    for point in points:
        point_date = date.fromisoformat(str(point["date"]))
        if point["confidence"] == "low":
            if start is None:
                start = point_date
            previous = point_date
            continue

        if start is not None and previous is not None:
            ranges.append((start, previous))
            start = None
            previous = None

    if start is not None and previous is not None:
        ranges.append((start, previous))

    return ranges


def render_trend(
    geo_id: str,
    start_date: date,
    end_date: date,
    fetch_history: Callable[[str, str, date, date], dict[str, Any]],
    key_prefix: str = "pulseiq-trend",
) -> None:
    """Render the score history chart and its window selector."""
    window = st.radio(
        "Window",
        options=["7d", "30d", "90d"],
        horizontal=True,
        key=f"{key_prefix}-window",
    )
    lookback_days = {"7d": 7, "30d": 30, "90d": 90}[window]
    requested_start = max(start_date, end_date - timedelta(days=lookback_days))

    try:
        history = fetch_history(geo_id, window, requested_start, end_date)
    except Exception as exc:
        st.error(str(exc))
        return
    points = history.get("points", [])
    if not points:
        st.info("No score history is available for the selected geography and date.")
        return

    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"])
    df["rolling_7d"] = df["ess_score"].rolling(window=7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ess_score"],
            mode="lines+markers",
            name="Raw score",
            line={"width": 1.5, "color": "#5a6b7a"},
            marker={"size": 5, "color": "#5a6b7a"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rolling_7d"],
            mode="lines",
            name="7-day average",
            line={"width": 3.5, "color": "#c44e1a"},
        )
    )
    fig.add_hline(
        y=_ALERT_THRESHOLD,
        line_dash="dash",
        line_color="#a7281c",
        annotation_text="Alert threshold",
        annotation_position="top left",
    )

    for start, end in _low_confidence_ranges(points):
        fig.add_vrect(
            x0=start,
            x1=end + timedelta(days=1),
            fillcolor="#f7d774",
            opacity=0.22,
            line_width=0,
            layer="below",
        )

    for point in points:
        if point["missing_sources"]:
            point_date = pd.to_datetime(point["date"])
            fig.add_annotation(
                x=point_date,
                y=point["ess_score"],
                text="⚠️",
                showarrow=False,
                yshift=18,
                font={"size": 14},
            )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        xaxis_title="Date",
        yaxis_title="ESS score",
        yaxis={"range": [0, 100]},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Trend: {history.get('trend', 'stable')}")
