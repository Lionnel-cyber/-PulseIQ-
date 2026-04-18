"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { getAlerts } from "@/lib/api";
import { DEFAULT_FILTER_STATE, type AlertPayload } from "@/lib/types";

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

const ALERT_TYPE_COLOR: Record<string, string> = {
  threshold_breach: "var(--danger)",
  rapid_rise: "var(--warn)",
  sustained_high: "var(--danger)",
  ingestion_failure: "var(--danger)",
  record_count_drop: "var(--warn)",
  latency_spike: "var(--warn)",
  source_stale: "var(--warn)",
};

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertPayload[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getAlerts();
        setAlerts(data);
      } catch {
        setAlerts([]);
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

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
                Alerts
              </div>
              <h1
                style={{
                  margin: 0,
                  fontSize: "24px",
                  color: "var(--text)",
                  fontWeight: 700,
                }}
              >
                ALERT HISTORY
              </h1>
            </div>
          </header>

          <SectionFrame title="Alert Log">
            {loading ? (
              <div style={{ color: "var(--text3)", fontSize: "11px", lineHeight: 1.8 }}>
                Loading alerts...
              </div>
            ) : alerts.length === 0 ? (
              <div
                style={{
                  border: "1px solid var(--border)",
                  padding: "24px",
                  display: "grid",
                  gap: "18px",
                }}
              >
                <div
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    width: "fit-content",
                    padding: "8px 12px",
                    borderRadius: "999px",
                    background: "rgba(0,212,170,0.08)",
                    color: "var(--accent)",
                    border: "1px solid rgba(0,212,170,0.2)",
                    fontSize: "11px",
                    letterSpacing: "2px",
                    fontFamily: "var(--mono)",
                    fontWeight: 700,
                  }}
                >
                  NO ALERTS FIRED
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                    gap: "24px",
                  }}
                >
                  <div style={{ display: "grid", gap: "10px" }}>
                    <div
                      style={{
                        fontSize: "9px",
                        color: "var(--text3)",
                        letterSpacing: "0.14em",
                        fontFamily: "var(--mono)",
                      }}
                    >
                      ALERT CONDITIONS
                    </div>
                    {[
                      "ESS score exceeds threshold (75)",
                      "Score rises 10+ pts in 7 days",
                      "Tier 1 source confirms Tier 3 spike",
                    ].map((item) => (
                      <div
                        key={item}
                        style={{ display: "flex", alignItems: "center", gap: "10px" }}
                      >
                        <span
                          style={{
                            width: "8px",
                            height: "8px",
                            borderRadius: "999px",
                            background: "var(--warn)",
                            flexShrink: 0,
                          }}
                        />
                        <span style={{ fontSize: "11px", color: "var(--text2)" }}>
                          {item}
                        </span>
                      </div>
                    ))}
                  </div>

                  <div style={{ display: "grid", gap: "10px" }}>
                    <div
                      style={{
                        fontSize: "9px",
                        color: "var(--text3)",
                        letterSpacing: "0.14em",
                        fontFamily: "var(--mono)",
                      }}
                    >
                      SUPPRESSION RULES
                    </div>
                    {[
                      "3-day cooldown after threshold alert",
                      "No re-alert if score moved < 5 pts",
                      "Sustained high: 7-day cooldown",
                    ].map((item) => (
                      <div
                        key={item}
                        style={{ display: "flex", alignItems: "center", gap: "10px" }}
                      >
                        <span
                          style={{
                            width: "8px",
                            height: "8px",
                            borderRadius: "999px",
                            background: "var(--border)",
                            flexShrink: 0,
                          }}
                        />
                        <span style={{ fontSize: "11px", color: "var(--text2)" }}>
                          {item}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div
                  style={{
                    fontSize: "11px",
                    color: "var(--text2)",
                    fontFamily: "var(--mono)",
                    lineHeight: 1.7,
                  }}
                >
                  Current threshold: 75 {"\u00b7"} 0 geos above threshold {"\u00b7"}{" "}
                  Alerts will appear here automatically
                </div>
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
                      <th style={{ padding: "0 12px 10px 0" }}>Triggered</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Region</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Type</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Score</th>
                      <th style={{ padding: "0 12px 10px 0" }}>Confidence</th>
                      <th style={{ padding: "0 0 10px" }}>Summary</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alerts.map((alert) => (
                      <tr
                        key={alert.alert_id}
                        style={{ borderTop: "1px solid var(--border)" }}
                      >
                        <td
                          style={{
                            padding: "11px 12px 11px 0",
                            color: "var(--text2)",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {new Date(alert.triggered_at).toLocaleString()}
                        </td>
                        <td style={{ padding: "11px 12px 11px 0" }}>
                          {alert.region_name}
                        </td>
                        <td
                          style={{
                            padding: "11px 12px 11px 0",
                            color:
                              ALERT_TYPE_COLOR[alert.alert_type] ?? "var(--warn)",
                          }}
                        >
                          {alert.alert_type}
                        </td>
                        <td style={{ padding: "11px 12px 11px 0" }}>
                          {alert.current_score.toFixed(1)}
                        </td>
                        <td style={{ padding: "11px 12px 11px 0" }}>
                          {alert.confidence}
                        </td>
                        <td
                          style={{
                            padding: "11px 0",
                            color: "var(--text2)",
                            maxWidth: 320,
                          }}
                        >
                          {alert.explanation_summary}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </SectionFrame>
        </div>
      </div>
    </div>
  );
}
