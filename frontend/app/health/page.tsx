"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "@/components/Sidebar";
import {
  getHealth,
  getHealthDashboard,
  getModelVersion,
  getPipelineStatus,
  getSourceFreshness,
} from "@/lib/api";
import {
  DEFAULT_FILTER_STATE,
  type HealthDashboardResponse,
  type HealthResponse,
  type ModelVersionResponse,
  type PipelineStatusResponse,
  type SourceFreshnessPayload,
} from "@/lib/types";

interface HealthPageData {
  freshness: SourceFreshnessPayload[];
  health: HealthResponse;
  model: ModelVersionResponse;
  pipeline: PipelineStatusResponse;
  dashboard: HealthDashboardResponse;
}

type StatusTone = "ok" | "degraded" | "down";

const POLL_INTERVAL_MS = 30_000;

function formatDate(value: string | null, includeTime = false): string {
  if (!value) {
    return "--";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    ...(includeTime
      ? { hour: "2-digit", minute: "2-digit" }
      : {}),
  });
}

function formatLatency(value: number | null): string {
  return value == null ? "--" : `${value.toFixed(1)}s`;
}

function getStatusColor(status: StatusTone): string {
  switch (status) {
    case "ok":
      return "var(--accent)";
    case "degraded":
      return "var(--warn)";
    default:
      return "var(--danger)";
  }
}

function getFreshnessSortOrder(status: SourceFreshnessPayload["freshness_status"]): number {
  switch (status) {
    case "critical":
      return 0;
    case "stale":
      return 1;
    case "unknown":
      return 2;
    default:
      return 3;
  }
}

function getFreshnessBadge(status: SourceFreshnessPayload["freshness_status"]) {
  switch (status) {
    case "ok":
      return { tone: "ok" as const, label: "OK" };
    case "stale":
      return { tone: "degraded" as const, label: "STALE" };
    case "critical":
      return { tone: "down" as const, label: "CRITICAL" };
    default:
      return { tone: "down" as const, label: "UNKNOWN" };
  }
}

function formatFreshness(row: {
  days_since_fetch: number | null;
  freshness_status: SourceFreshnessPayload["freshness_status"];
}): string {
  if (row.days_since_fetch == null) return "Never fetched";
  switch (row.freshness_status) {
    case "ok":      return "Up to date";
    case "stale":   return `${row.days_since_fetch}d since last fetch`;
    case "critical": return `${row.days_since_fetch}d overdue`;
    default:        return "Unknown";
  }
}

const SOURCE_PURPOSE: Record<string, string> = {
  bls:         "Unemployment claims (Tier 1)",
  fred:        "Interest rates & delinquency (Tier 1–2)",
  census:      "Poverty baseline — annual (Tier 1)",
  news:        "News sentiment (Tier 3)",
  openweather: "Weather events",
  rss:         "Economic RSS feeds (Tier 3)",
  trends:      "Google search trends (Tier 3)",
};

function derivePipelineStatus(
  pipeline: PipelineStatusResponse
): { tone: StatusTone; text: string } {
  if (pipeline.status !== "down") {
    return { tone: pipeline.status as StatusTone, text: pipeline.status };
  }
  const isStale =
    pipeline.failures.length > 0 &&
    pipeline.failures.every((f) => /not run in|hours|stale/i.test(f));
  return isStale
    ? { tone: "degraded", text: "stale" }
    : { tone: "down", text: "down" };
}

function deriveModelStatus(model: ModelVersionResponse): StatusTone {
  if (
    model.model_version === "unknown" ||
    model.feature_version === "unknown" ||
    !model.trained_at
  ) {
    return "down";
  }

  if (!model.calibrated || model.calibration_samples < 60) {
    return "degraded";
  }

  return "ok";
}

function ApiErrorPanel({ message }: { message: string }) {
  return (
    <div
      style={{
        minHeight: "calc(100vh - 40px)",
        display: "grid",
        gridTemplateColumns: "200px minmax(0, 1fr)",
        background: "var(--bg)",
      }}
    >
      <Sidebar
        value={DEFAULT_FILTER_STATE}
        onFiltersChange={() => undefined}
        showMonitorControls={false}
      />
      <div
        style={{
          padding: "24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            maxWidth: 720,
            border: "1px solid rgba(229,62,62,0.35)",
            background: "rgba(229,62,62,0.08)",
            borderRadius: "6px",
            padding: "18px 20px",
            color: "var(--text)",
            fontFamily: "var(--mono)",
          }}
        >
          <div style={{ fontSize: "18px", fontWeight: 700, marginBottom: 10 }}>
            Could not reach the PulseIQ API
          </div>
          <div style={{ fontSize: "11px", lineHeight: 1.8, color: "var(--text2)" }}>
            {message}
          </div>
          <div style={{ marginTop: 14, fontSize: "11px" }}>Start it with:</div>
          <pre
            style={{
              marginTop: 8,
              padding: "10px 12px",
              background: "#0a0c0f",
              border: "1px solid var(--border)",
              borderRadius: "4px",
              color: "var(--accent)",
              fontSize: "12px",
              overflowX: "auto",
            }}
          >
            uvicorn src.api.main:app --reload
          </pre>
        </div>
      </div>
    </div>
  );
}

function SectionFrame({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "6px",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "10px 12px",
          borderBottom: "1px solid var(--border)",
          fontFamily: "var(--mono)",
          fontSize: "10px",
          color: "var(--text3)",
          textTransform: "uppercase",
          letterSpacing: "0.12em",
        }}
      >
        {title}
      </div>
      <div style={{ padding: "14px 16px" }}>{children}</div>
    </section>
  );
}

function StatusPill({
  label,
  status,
  statusText,
}: {
  label: string;
  status: StatusTone;
  statusText?: string;
}) {
  const color = getStatusColor(status);

  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "8px",
        padding: "8px 12px",
        borderRadius: "999px",
        border: `1px solid ${color}`,
        background:
          status === "ok"
            ? "rgba(0,212,170,0.08)"
            : status === "degraded"
            ? "rgba(245,166,35,0.12)"
            : "rgba(229,62,62,0.12)",
        color,
        fontFamily: "var(--mono)",
        fontSize: "11px",
        letterSpacing: "0.05em",
        textTransform: "uppercase",
      }}
    >
      <span>{label}</span>
      <span>|</span>
      <span>{statusText ?? status}</span>
    </div>
  );
}

function StatusDot({ status }: { status: StatusTone }) {
  return (
    <span
      style={{
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: getStatusColor(status),
        display: "inline-block",
        flexShrink: 0,
      }}
    />
  );
}

export default function HealthPage() {
  const [data, setData] = useState<HealthPageData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const firstLoadRef = useRef(true);

  useEffect(() => {
    let cancelled = false;

    async function fetchHealthData() {
      if (firstLoadRef.current && !cancelled) {
        setLoading(true);
      }

      try {
        const [freshness, health, model, pipeline, dashboard] = await Promise.all([
          getSourceFreshness(),
          getHealth(),
          getModelVersion(),
          getPipelineStatus(),
          getHealthDashboard(),
        ]);

        if (!cancelled) {
          setData({ freshness, health, model, pipeline, dashboard });
          setError(null);
          setLoading(false);
          firstLoadRef.current = false;
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(
            fetchError instanceof Error
              ? fetchError.message
              : "Failed to load pipeline health."
          );
          setLoading(false);
        }
      }
    }

    void fetchHealthData();
    const timer = setInterval(() => void fetchHealthData(), POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const rows = useMemo(() => {
    if (!data) {
      return [];
    }

    const latencyBySource = new Map(
      data.dashboard.source_health.map((row) => [row.source, row])
    );

    return [...data.freshness]
      .sort((left, right) => {
        const diff =
          getFreshnessSortOrder(left.freshness_status) -
          getFreshnessSortOrder(right.freshness_status);
        return diff !== 0 ? diff : left.source.localeCompare(right.source);
      })
      .map((row) => ({
        ...row,
        latency_seconds: latencyBySource.get(row.source)?.latency_seconds ?? null,
        last_run: latencyBySource.get(row.source)?.last_run ?? null,
        records: latencyBySource.get(row.source)?.records ?? row.records_last_run,
      }));
  }, [data]);

  if (error) {
    return <ApiErrorPanel message={error} />;
  }

  const dbStatus: StatusTone = data ? "ok" : "down";
  const modelStatus: StatusTone = data ? deriveModelStatus(data.model) : "down";
  const pipelineDerived = data
    ? derivePipelineStatus(data.pipeline)
    : { tone: "down" as StatusTone, text: "down" };
  const lastEvaluated = data?.pipeline.last_score_run ?? data?.health.checked_at ?? null;

  return (
    <div
      style={{
        minHeight: "calc(100vh - 40px)",
        display: "grid",
        gridTemplateColumns: "200px minmax(0, 1fr)",
        background: "var(--bg)",
        fontFamily: "var(--mono)",
      }}
    >
      <Sidebar
        value={DEFAULT_FILTER_STATE}
        onFiltersChange={() => undefined}
        showMonitorControls={false}
      />

      <div style={{ padding: "18px 20px 28px", minWidth: 0 }}>
        <div style={{ maxWidth: 1180, margin: "0 auto", display: "grid", gap: "14px" }}>
          <header
            style={{
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "space-between",
              gap: "16px",
              flexWrap: "wrap",
            }}
          >
            <div>
              <div
                style={{
                  fontSize: "10px",
                  letterSpacing: "0.14em",
                  color: "var(--text3)",
                  textTransform: "uppercase",
                  marginBottom: "6px",
                }}
              >
                Pipeline Health
              </div>
              <h1
                style={{
                  margin: 0,
                  fontSize: "24px",
                  color: "var(--text)",
                  fontWeight: 700,
                }}
              >
                PulseIQ operational status
              </h1>
            </div>

            <div
              style={{
                color: "var(--text3)",
                fontSize: "11px",
                letterSpacing: "0.05em",
              }}
            >
              {data ? `Last checked ${formatDate(data.health.checked_at, true)}` : "Loading..."}
            </div>
          </header>

          <SectionFrame title="System Status">
            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
              <StatusPill label="DB" status={dbStatus} />
              <StatusPill label="Model" status={modelStatus} />
              <StatusPill
                label="Pipeline"
                status={pipelineDerived.tone}
                statusText={pipelineDerived.text}
              />
            </div>
            {data?.pipeline.failures && data.pipeline.failures.length > 0 && (
              <ul
                style={{
                  margin: "10px 0 0",
                  padding: "0 0 0 16px",
                  fontSize: "10px",
                  color: "var(--text3)",
                  lineHeight: 1.8,
                }}
              >
                {data.pipeline.failures.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            )}
          </SectionFrame>

          <SectionFrame title="How to read this page">
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "12px 32px",
                fontSize: "11px",
                color: "var(--text2)",
                lineHeight: 1.7,
              }}
            >
              <div>
                <div style={{ color: "var(--text3)", fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  DB badge
                </div>
                Connected to the DuckDB score store. OK = reachable and has rows.
              </div>
              <div>
                <div style={{ color: "var(--text3)", fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  Model badge
                </div>
                XGBoost + isotonic calibration. Degraded = not yet calibrated on ≥60 confirmed events.
              </div>
              <div>
                <div style={{ color: "var(--text3)", fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  Pipeline badge
                </div>
                OK = all three stages (ingest, transform, score) ran within 24h. Stale = timing gap only, not a crash.
              </div>
              <div>
                <div style={{ color: "var(--text3)", fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  Source freshness
                </div>
                Stale after 1× expected cadence. Critical after 2×. A critical source downgrades nearby geo scores to confidence=low.
              </div>
            </div>
          </SectionFrame>

          <SectionFrame title="Source Health">
            {loading && !data ? (
              <div
                style={{
                  color: "var(--text3)",
                  fontSize: "11px",
                  lineHeight: 1.8,
                }}
              >
                Loading source health...
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: "11px",
                    color: "var(--text)",
                  }}
                >
                  <thead>
                    <tr style={{ color: "var(--text3)", textAlign: "left" }}>
                      <th style={{ padding: "0 12px 10px 0" }}>Source</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Purpose</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Last fetch</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Status</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Records</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Latency</th>
                      <th style={{ padding: "0 0 10px" }}>Freshness</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => {
                      const badge = getFreshnessBadge(row.freshness_status);

                      return (
                        <tr key={row.source} style={{ borderTop: "1px solid var(--border)" }}>
                          <td style={{ padding: "11px 12px 11px 0" }}>{row.source}</td>
                          <td style={{ padding: "11px 12px 11px 0", color: "var(--text2)" }}>
                            {SOURCE_PURPOSE[row.source] ?? "—"}
                          </td>
                          <td style={{ padding: "11px 12px 11px 0", color: "var(--text2)" }}>
                            {formatDate(row.last_successful_fetch)}
                          </td>
                          <td style={{ padding: "11px 12px 11px 0" }}>
                            <span
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: "8px",
                              }}
                            >
                              <StatusDot status={badge.tone} />
                              <span style={{ color: getStatusColor(badge.tone) }}>
                                {badge.label}
                              </span>
                            </span>
                          </td>
                          <td style={{ padding: "11px 12px 11px 0" }}>{row.records}</td>
                          <td style={{ padding: "11px 12px 11px 0", color: "var(--text2)" }}>
                            {formatLatency(row.latency_seconds)}
                          </td>
                          <td style={{ padding: "11px 0", color: "var(--text2)" }}>
                            {formatFreshness(row)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </SectionFrame>

          <SectionFrame title="Model Info">
            {loading && !data ? (
              <div
                style={{
                  color: "var(--text3)",
                  fontSize: "11px",
                  lineHeight: 1.8,
                }}
              >
                Loading model metadata...
              </div>
            ) : data ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                  gap: "12px 24px",
                  fontSize: "11px",
                }}
              >
                <div>
                  <div style={{ color: "var(--text3)", marginBottom: 4 }}>Model version</div>
                  <div style={{ color: "var(--text)" }}>{data.model.model_version}</div>
                </div>
                <div>
                  <div style={{ color: "var(--text3)", marginBottom: 4 }}>Feature version</div>
                  <div style={{ color: "var(--text)" }}>{data.model.feature_version}</div>
                </div>
                <div>
                  <div style={{ color: "var(--text3)", marginBottom: 4 }}>Trained date</div>
                  <div style={{ color: "var(--text)" }}>{formatDate(data.model.trained_at)}</div>
                </div>
                <div>
                  <div style={{ color: "var(--text3)", marginBottom: 4 }}>Calibrated</div>
                  <div style={{ color: "var(--text)" }}>{data.model.calibrated ? "yes" : "no"}</div>
                </div>
                <div>
                  <div style={{ color: "var(--text3)", marginBottom: 4 }}>Last evaluated</div>
                  <div style={{ color: "var(--text)" }}>{formatDate(lastEvaluated, true)}</div>
                </div>
              </div>
            ) : null}
          </SectionFrame>
        </div>
      </div>
    </div>
  );
}
