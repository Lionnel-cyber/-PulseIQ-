"""Pipeline health component for the PulseIQ dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

_STATUS_ICON = {"ok": "✅ OK", "slow": "⚠️ SLOW", "down": "🔴 DOWN"}


def _format_timestamp(value: str | None) -> str:
    """Format an API timestamp string for dashboard display."""
    if not value:
        return "—"
    return str(value).replace("T", " ").replace("+00:00", " UTC")


def _format_latency(value: float | None) -> str:
    """Format latency seconds with a compact suffix."""
    if value is None:
        return "—"
    return f"{value:.1f}s"


def render_health_page(health: dict[str, Any]) -> None:
    """Render the Pipeline Health page."""
    source_rows = list(health.get("source_health") or [])
    model_info = dict(health.get("model_info") or {})
    benchmark = dict(model_info.get("benchmark") or {})
    pipeline = dict(health.get("pipeline_info") or {})

    table = pd.DataFrame(
        [
            {
                "source": row.get("source", "unknown"),
                "last_run": _format_timestamp(row.get("last_run")),
                "status": _STATUS_ICON.get(row.get("status", "down"), "🔴 DOWN"),
                "records": int(row.get("records", 0) or 0),
                "latency": _format_latency(row.get("latency_seconds")),
                "7d trend": row.get("trend_7d", "unknown"),
            }
            for row in source_rows
        ]
    )

    st.subheader("Source Health")
    if table.empty:
        st.info("No source health rows are available yet.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Model Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Version", model_info.get("version", "unknown"))
        st.metric("Trained date", _format_timestamp(model_info.get("trained_at")))
    with col2:
        st.metric("Calibrated", "Yes" if model_info.get("calibrated") else "No")
        verdict = benchmark.get("verdict", "unknown")
        improvement = benchmark.get("improvement_pct")
        if improvement is None:
            benchmark_label = verdict
        else:
            benchmark_label = f"{verdict} ({improvement:.2f}% vs baseline)"
        st.metric("Benchmark result", benchmark_label)

    if benchmark.get("warning"):
        st.warning(str(benchmark["warning"]))

    st.subheader("Pipeline Info")
    st.write(f"Last ingest run: {_format_timestamp(pipeline.get('last_ingest_run'))}")
    st.write(f"Last transform run: {_format_timestamp(pipeline.get('last_transform_run'))}")
    st.write(f"Last score run: {_format_timestamp(pipeline.get('last_score_run'))}")
    st.write(f"Anomaly flags count: {int(pipeline.get('anomaly_flags_count', 0) or 0)}")

    failures = list(pipeline.get("failures") or [])
    if failures:
        for failure in failures:
            st.warning(str(failure))
    else:
        st.success(f"Pipeline status: {pipeline.get('status', 'unknown')}")
