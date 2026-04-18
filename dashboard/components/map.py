"""Map component for the PulseIQ dashboard."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

_OPACITY = {"high": 0.85, "medium": 0.60, "low": 0.30}
_BASE_SCALE = (
    (0.0, (26, 152, 80)),
    (0.5, (254, 224, 139)),
    (1.0, (215, 48, 39)),
)

_STATE_CENTERS: dict[str, tuple[float, float]] = {
    "AL": (32.806671, -86.79113),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564),
    "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),
    "DE": (39.318523, -75.507141),
    "DC": (38.9072, -77.0369),
    "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337),
    "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),
    "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526),
    "KS": (38.5266, -96.726486),
    "KY": (37.66814, -84.670067),
    "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927),
    "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106),
    "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),
    "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368),
    "MT": (46.921925, -110.454353),
    "NE": (41.12537, -98.268082),
    "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482),
    "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.51178),
    "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),
    "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686),
    "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494),
    "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508),
    "WY": (42.755966, -107.30249),
}


def _selection_points(selection: Any) -> list[dict[str, Any]]:
    """Return Plotly selection points across Streamlit versions."""
    if not selection:
        return []
    if isinstance(selection, dict):
        return selection.get("selection", {}).get("points", [])

    try:
        payload = dict(selection)
    except Exception:
        payload = {}
    return payload.get("selection", {}).get("points", [])


def _confidence_colorscale(confidence: str) -> list[list[str | float]]:
    """Return the shared green-yellow-red scale with confidence alpha baked in."""
    alpha = _OPACITY[confidence]
    return [
        [stop, f"rgba({red},{green},{blue},{alpha})"]
        for stop, (red, green, blue) in _BASE_SCALE
    ]


def render_map(map_rows: list[dict[str, Any]], chart_key: str = "pulseiq-map") -> str | None:
    """Render the US choropleth and return the clicked drill-down geo_id."""
    if not map_rows:
        st.info("No score snapshot is available yet. Run the scoring pipeline, then refresh.")
        return None

    latest_date = max(row["run_date"] for row in map_rows)
    state_lookup = {row["state_code"]: row["drilldown_geo_id"] for row in map_rows}

    fig = go.Figure()
    show_scale = True

    for confidence in ("high", "medium", "low"):
        rows = [row for row in map_rows if row["confidence"] == confidence]
        if not rows:
            continue

        customdata = [
            [
                row["drilldown_geo_id"],
                row["geo_name"],
                row["ess_score"],
                row.get("delta_7d"),
                row["confidence"],
                row["run_date"],
                ", ".join(row["missing_sources"]) or "None",
                "Yes" if row["granularity_warning"] else "None",
            ]
            for row in rows
        ]

        fig.add_trace(
            go.Choropleth(
                locations=[row["state_code"] for row in rows],
                z=[row["ess_score"] for row in rows],
                locationmode="USA-states",
                zmin=0,
                zmax=100,
                ids=[row["drilldown_geo_id"] for row in rows],
                customdata=customdata,
                # Plotly choropleths do not support a top-level `opacity` field,
                # so confidence is encoded directly into the fill colors.
                colorscale=_confidence_colorscale(confidence),
                marker={"line": {"color": "#f7f7f7", "width": 0.8}},
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Score: %{customdata[2]:.1f}<br>"
                    "Delta 7d: %{customdata[3]}<br>"
                    "Confidence: %{customdata[4]}<br>"
                    "Run date: %{customdata[5]}<br>"
                    "Missing sources: %{customdata[6]}<br>"
                    "Granularity warning: %{customdata[7]}"
                    "<extra></extra>"
                ),
                colorbar=(
                    {
                        "title": "ESS",
                        "tickvals": [0, 50, 100],
                        "ticktext": ["0", "50", "100"],
                    }
                    if show_scale
                    else None
                ),
                showscale=show_scale,
                name=confidence.title(),
            )
        )
        show_scale = False

    hatched_rows = [row for row in map_rows if row["missing_sources"]]
    if hatched_rows:
        latitudes: list[float] = []
        longitudes: list[float] = []
        labels: list[str] = []

        for row in hatched_rows:
            center = _STATE_CENTERS.get(row["state_code"])
            if center is None:
                continue
            latitudes.append(center[0])
            longitudes.append(center[1])
            labels.append("///")

        if latitudes:
            fig.add_trace(
                go.Scattergeo(
                    lat=latitudes,
                    lon=longitudes,
                    text=labels,
                    mode="text",
                    textfont={"size": 16, "color": "#444444"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        geo={
            "scope": "usa",
            "projection": {"type": "albers usa"},
            "showland": True,
            "landcolor": "#f8f7f2",
            "bgcolor": "rgba(0,0,0,0)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select",
        legend={"orientation": "h", "y": -0.06},
    )

    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        key=chart_key,
        on_select="rerun",
        selection_mode="points",
    )
    caption = (
        f"Data as of {latest_date} \u00b7 Opacity = confidence \u00b7 "
        "Hatching = missing sources"
    )
    st.caption(caption)

    points = _selection_points(selection)
    if not points:
        return None

    selected = points[-1]
    customdata = selected.get("customdata")
    if isinstance(customdata, list) and customdata:
        return str(customdata[0])

    location = selected.get("location")
    if location in state_lookup:
        return state_lookup[location]

    point_index = selected.get("point_index")
    curve_number = selected.get("curve_number")
    if isinstance(point_index, int) and isinstance(curve_number, int):
        try:
            trace = fig.data[curve_number]
            if hasattr(trace, "ids") and trace.ids:
                return str(trace.ids[point_index])
        except Exception:
            return None

    return None
