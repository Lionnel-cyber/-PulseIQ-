"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { getModelVersion } from "@/lib/api";
import { DEFAULT_FILTER_STATE, type ModelVersionResponse } from "@/lib/types";

type StatusTone = "ok" | "degraded" | "down";

function getStatusColor(tone: StatusTone): string {
  if (tone === "ok") return "var(--accent)";
  if (tone === "degraded") return "var(--warn)";
  return "var(--danger)";
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

function MetaRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div
        style={{
          color: "var(--text3)",
          fontSize: "10px",
          textTransform: "uppercase",
          letterSpacing: "0.1em",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ color: "var(--text)", fontSize: "11px" }}>{value}</div>
    </div>
  );
}

const PSI_THRESHOLDS = [
  {
    range: "< 0.1",
    label: "STABLE",
    tone: "ok" as StatusTone,
    action: "No action needed — model is behaving as trained.",
  },
  {
    range: "0.1 – 0.2",
    label: "MONITOR",
    tone: "degraded" as StatusTone,
    action: "Log warning and increase monitoring frequency.",
  },
  {
    range: "> 0.2",
    label: "RETRAIN",
    tone: "down" as StatusTone,
    action: "Population has shifted significantly — retraining recommended.",
  },
];

function formatDate(value: string | null): string {
  if (!value) return "--";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export default function DriftPage() {
  const [model, setModel] = useState<ModelVersionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const data = await getModelVersion();
        setModel(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load model version.");
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  const calibrationTone: StatusTone =
    !model || !model.calibrated || model.calibration_samples < 60
      ? "degraded"
      : "ok";

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
        <div
          style={{ maxWidth: 1180, margin: "0 auto", display: "grid", gap: "14px" }}
        >
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
                ML Observability
              </div>
              <h1
                style={{
                  margin: 0,
                  fontSize: "24px",
                  color: "var(--text)",
                  fontWeight: 700,
                }}
              >
                MODEL DRIFT MONITOR
              </h1>
            </div>
          </header>

          <SectionFrame title="Deployed Model">
            {loading ? (
              <div style={{ color: "var(--text3)", fontSize: "11px", lineHeight: 1.8 }}>
                Loading model metadata...
              </div>
            ) : error ? (
              <div style={{ color: "var(--danger)", fontSize: "11px" }}>{error}</div>
            ) : model ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                  gap: "12px 24px",
                }}
              >
                <MetaRow label="Model version" value={model.model_version} />
                <MetaRow label="Feature version" value={model.feature_version} />
                <MetaRow label="Trained" value={formatDate(model.trained_at)} />
                <MetaRow
                  label="Calibrated"
                  value={
                    <span
                      style={{
                        color: getStatusColor(calibrationTone),
                      }}
                    >
                      {model.calibrated
                        ? `yes · ${model.calibration_samples} events`
                        : `no${model.calibration_samples > 0 ? ` · ${model.calibration_samples}/60 events` : ""}`}
                    </span>
                  }
                />
                {model.mlflow_run_id && (
                  <MetaRow
                    label="MLflow run"
                    value={
                      <span
                        style={{
                          color: "var(--text2)",
                          wordBreak: "break-all",
                          fontSize: "10px",
                        }}
                      >
                        {model.mlflow_run_id}
                      </span>
                    }
                  />
                )}
              </div>
            ) : null}
          </SectionFrame>

          <SectionFrame title="PSI Drift Thresholds">
            <div
              style={{
                fontSize: "11px",
                color: "var(--text2)",
                marginBottom: "14px",
                lineHeight: 1.7,
              }}
            >
              Population Stability Index (PSI) measures how much the scoring
              population has shifted since training. Drift is computed daily after
              scoring via <code style={{ color: "var(--accent)" }}>src/models/monitor.py</code>.
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {PSI_THRESHOLDS.map((t) => {
                const color = getStatusColor(t.tone);
                return (
                  <div
                    key={t.label}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "80px 80px 1fr",
                      alignItems: "center",
                      gap: "12px",
                      padding: "10px 12px",
                      border: `1px solid ${color}`,
                      borderRadius: "4px",
                      background:
                        t.tone === "ok"
                          ? "rgba(0,212,170,0.05)"
                          : t.tone === "degraded"
                          ? "rgba(245,166,35,0.07)"
                          : "rgba(229,62,62,0.07)",
                    }}
                  >
                    <span
                      style={{
                        color: "var(--text3)",
                        fontSize: "10px",
                        letterSpacing: "0.04em",
                      }}
                    >
                      PSI {t.range}
                    </span>
                    <span
                      style={{
                        color,
                        fontWeight: 700,
                        fontSize: "10px",
                        letterSpacing: "0.08em",
                      }}
                    >
                      {t.label}
                    </span>
                    <span style={{ color: "var(--text2)", fontSize: "11px" }}>
                      {t.action}
                    </span>
                  </div>
                );
              })}
            </div>
          </SectionFrame>

          <SectionFrame title="Live Drift Data">
            <div
              style={{
                color: "var(--text3)",
                fontSize: "11px",
                padding: "24px 0",
                textAlign: "center",
                lineHeight: 1.8,
              }}
            >
              Live PSI readings are written to DuckDB by{" "}
              <code style={{ color: "var(--accent)" }}>dag_score_and_alert</code>{" "}
              after each scoring run.
              <br />
              Query the <code style={{ color: "var(--accent)" }}>drift_metrics</code>{" "}
              table directly or trigger a scoring run to populate this view.
            </div>
          </SectionFrame>
        </div>
      </div>
    </div>
  );
}
