"use client";

import { useEffect, useMemo, useState } from "react";
import { getHistory, getScore } from "@/lib/api";
import type {
  HistoryWindow,
  ScoreBand,
  ScoreResponse,
  TimeSeriesPoint,
  TimeSeriesResponse,
} from "@/lib/types";
import ConfidenceBadge from "./ConfidenceBadge";
import ShapWaterfall from "./ShapWaterfall";
import TierBars from "./TierBars";

interface IntelPanelProps {
  selectedGeoId: string | null;
}

const ALERT_THRESHOLD = 75;

// ── helpers ─────────────────────────────────────────────────────────────────

function formatShortDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatDelta(delta: number | null): string {
  if (delta === null) return "No 7d delta";
  const prefix = delta > 0 ? "+" : "";
  return `${prefix}${delta.toFixed(1)} pts / 7d`;
}

function scoreBandColor(band: ScoreBand | undefined): string {
  switch (band) {
    case "low":
      return "var(--accent)";
    case "elevated":
      return "var(--warn)";
    case "high":
    case "critical":
      return "var(--danger)";
    default:
      return "var(--text)";
  }
}

function deltaColor(delta: number | null): string {
  if (delta === null) return "var(--text3)";
  return delta > 0 ? "var(--danger)" : "var(--accent)";
}

function computeRollingAverage(points: TimeSeriesPoint[]): number[] {
  return points.map((_, index) => {
    const start = Math.max(0, index - 6);
    const slice = points.slice(start, index + 1);
    const total = slice.reduce((sum, p) => sum + p.ess_score, 0);
    return total / slice.length;
  });
}

function getTrendColor(trend: TimeSeriesResponse["trend"]): string {
  switch (trend) {
    case "deteriorating":
      return "var(--danger)";
    case "improving":
      return "var(--accent)";
    case "volatile":
      return "var(--warn)";
    default:
      return "var(--text2)";
  }
}

function parseSseEvent(eventText: string): string | null {
  const lines = eventText.split("\n");
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^\s/, ""));
      continue;
    }

    // The backend currently sends a multi-line block in one SSE event, but
    // only prefixes the first line with `data:`. Treat later lines as part
    // of that same event so section bodies stay intact on the client.
    if (dataLines.length > 0) {
      dataLines.push(line);
    }
  }

  if (dataLines.length === 0) return null;
  return dataLines.join("\n").trim();
}

function readSseEvents(buffer: string): { events: string[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const chunks = normalized.split("\n\n");
  const rest = chunks.pop() ?? "";

  return {
    events: chunks
      .map((chunk) => parseSseEvent(chunk))
      .filter((chunk): chunk is string => Boolean(chunk)),
    rest,
  };
}

// ── small components ────────────────────────────────────────────────────────

function ScoreBandPill({ band }: { band: ScoreBand }) {
  const styles: Record<
    ScoreBand,
    { color: string; bg: string; border: string }
  > = {
    low: {
      color: "var(--accent)",
      bg: "rgba(0,212,170,0.15)",
      border: "rgba(0,212,170,0.30)",
    },
    elevated: {
      color: "var(--warn)",
      bg: "rgba(245,166,35,0.15)",
      border: "rgba(245,166,35,0.30)",
    },
    high: {
      color: "var(--danger)",
      bg: "rgba(229,62,62,0.15)",
      border: "rgba(229,62,62,0.30)",
    },
    critical: {
      color: "var(--danger)",
      bg: "rgba(229,62,62,0.20)",
      border: "rgba(229,62,62,0.55)",
    },
  };
  const s = styles[band];
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 7px",
        borderRadius: 999,
        border: `1px solid ${s.border}`,
        background: s.bg,
        color: s.color,
        fontFamily: "var(--mono)",
        fontSize: "9px",
        fontWeight: 700,
        letterSpacing: "1.5px",
        verticalAlign: "middle",
        marginLeft: 8,
      }}
    >
      {band.toUpperCase()}
    </span>
  );
}

function Section({
  children,
  flex,
}: {
  children: React.ReactNode;
  flex?: boolean;
}) {
  return (
    <div
      style={{
        padding: "12px 16px",
        borderBottom: flex ? "none" : "1px solid var(--border)",
        ...(flex
          ? { flex: 1, display: "flex", flexDirection: "column" as const }
          : {}),
      }}
    >
      {children}
    </div>
  );
}

function SectionLabel({ text }: { text: string }) {
  return (
    <div
      style={{
        fontFamily: "var(--mono)",
        fontSize: "9px",
        color: "var(--text3)",
        letterSpacing: "2px",
        textTransform: "uppercase",
        marginBottom: 10,
      }}
    >
      {text}
    </div>
  );
}

function WindowButton({
  value,
  current,
  onSelect,
}: {
  value: HistoryWindow;
  current: HistoryWindow;
  onSelect: (w: HistoryWindow) => void;
}) {
  const active = value === current;
  return (
    <button
      onClick={() => onSelect(value)}
      style={{
        padding: "4px 8px",
        borderRadius: 2,
        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
        color: active ? "var(--accent)" : "var(--text2)",
        background: active ? "rgba(0,212,170,0.08)" : "transparent",
        fontFamily: "var(--mono)",
        fontSize: "10px",
        letterSpacing: "0.04em",
        cursor: "pointer",
      }}
    >
      {value}
    </button>
  );
}

function TrendChart({
  history,
  loading,
}: {
  history: TimeSeriesResponse | null;
  loading: boolean;
}) {
  const points = useMemo(() => history?.points ?? [], [history]);
  const averaged = useMemo(() => computeRollingAverage(points), [points]);

  if (loading) {
    return (
      <div
        style={{
          height: 170,
          border: "1px solid var(--border)",
          background: "var(--surface)",
          borderRadius: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text3)",
          fontFamily: "var(--mono)",
          fontSize: "10px",
        }}
      >
        Loading history...
      </div>
    );
  }

  if (!history || points.length === 0) {
    return (
      <div
        style={{
          height: 170,
          border: "1px solid var(--border)",
          background: "var(--surface)",
          borderRadius: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text3)",
          fontFamily: "var(--mono)",
          fontSize: "10px",
          textAlign: "center",
          padding: "0 16px",
        }}
      >
        No history available for the selected geography.
      </div>
    );
  }

  const width = 248;
  const height = 170;
  const left = 14;
  const right = 10;
  const top = 12;
  const bottom = 22;
  const uw = width - left - right;
  const uh = height - top - bottom;

  const xAt = (i: number): number =>
    points.length === 1
      ? left + uw / 2
      : left + (uw * i) / (points.length - 1);

  const yAt = (v: number): number =>
    top + uh - (Math.max(0, Math.min(v, 100)) / 100) * uh;

  const rawLine = points
    .map((p, i) => `${xAt(i)},${yAt(p.ess_score)}`)
    .join(" ");

  const avgLine = averaged
    .map((v, i) => `${xAt(i)},${yAt(v)}`)
    .join(" ");

  const lowRanges: Array<{ start: number; end: number }> = [];
  let rs: number | null = null;
  points.forEach((p, i) => {
    if (p.confidence === "low") {
      if (rs === null) rs = i;
      return;
    }
    if (rs !== null) {
      lowRanges.push({ start: rs, end: i - 1 });
      rs = null;
    }
  });
  if (rs !== null) lowRanges.push({ start: rs, end: points.length - 1 });

  const labelIdxs = Array.from(
    new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])
  );

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        background: "var(--surface)",
        borderRadius: 4,
        padding: "8px 8px 6px",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 8,
          fontFamily: "var(--mono)",
          fontSize: "9px",
          color: "var(--text3)",
        }}
      >
        <span>Raw</span>
        <span>7d avg</span>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: "100%", height: 170, display: "block" }}
        role="img"
        aria-label="Economic stress trend chart"
      >
        {lowRanges.map((r) => (
          <rect
            key={`${r.start}-${r.end}`}
            x={xAt(r.start) - 6}
            y={top}
            width={Math.max(12, xAt(r.end) + 6 - (xAt(r.start) - 6))}
            height={uh}
            fill="rgba(247,215,116,0.2)"
          />
        ))}

        {[0, 25, 50, 75, 100].map((tick) => (
          <line
            key={tick}
            x1={left}
            x2={width - right}
            y1={yAt(tick)}
            y2={yAt(tick)}
            stroke={tick === ALERT_THRESHOLD ? "#a7281c" : "#1e2730"}
            strokeWidth={1}
            strokeDasharray={tick === ALERT_THRESHOLD ? "5 5" : "0"}
          />
        ))}

        {rawLine && (
          <polyline
            fill="none"
            stroke="#5a6b7a"
            strokeWidth={1.8}
            points={rawLine}
          />
        )}
        {avgLine && (
          <polyline
            fill="none"
            stroke="#c44e1a"
            strokeWidth={3}
            points={avgLine}
          />
        )}

        {points.map((p, i) => (
          <g key={`${p.date}-${i}`}>
            <circle
              cx={xAt(i)}
              cy={yAt(p.ess_score)}
              r={3.5}
              fill="#5a6b7a"
            />
            {p.missing_sources.length > 0 && (
              <text
                x={xAt(i)}
                y={yAt(p.ess_score) - 10}
                textAnchor="middle"
                fontFamily="monospace"
                fontSize="10"
                fill="#f5a623"
              >
                !
              </text>
            )}
          </g>
        ))}

        {labelIdxs.map((i) => (
          <text
            key={i}
            x={xAt(i)}
            y={height - 6}
            textAnchor="middle"
            fontFamily="monospace"
            fontSize="9"
            fill="#6a8099"
          >
            {formatShortDate(points[i].date)}
          </text>
        ))}
      </svg>

      <div
        style={{
          marginTop: 6,
          fontFamily: "var(--mono)",
          fontSize: "10px",
          color: getTrendColor(history.trend),
        }}
      >
        Trend: {history.trend}
      </div>
    </div>
  );
}

function EmptyPanel() {
  return (
    <aside
      style={{
        width: 280,
        minWidth: 280,
        height: "100%",
        background: "var(--surface)",
        borderLeft: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: "11px",
          color: "var(--text3)",
          letterSpacing: "2px",
        }}
      >
        SELECT A GEOGRAPHY
      </div>
    </aside>
  );
}

// ── main export ──────────────────────────────────────────────────────────────

export default function IntelPanel({ selectedGeoId }: IntelPanelProps) {
  const [historyWindow, setHistoryWindow] = useState<HistoryWindow>("30d");
  const [score, setScore] = useState<ScoreResponse | null>(null);
  const [history, setHistory] = useState<TimeSeriesResponse | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [scoreLoading, setScoreLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [explanation, setExplanation] = useState("");
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  useEffect(() => {
    setExplanation("");
    setExplanationError(null);
    setHistoryWindow("30d");

    if (!selectedGeoId) {
      setScore(null);
      setHistory(null);
      setScoreError(null);
      setHistoryError(null);
      return;
    }

    let cancelled = false;
    setScoreLoading(true);
    setScoreError(null);

    void getScore(selectedGeoId)
      .then((s) => {
        if (!cancelled) setScore(s);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setScore(null);
          setScoreError(
            e instanceof Error ? e.message : "Failed to load score."
          );
        }
      })
      .finally(() => {
        if (!cancelled) setScoreLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedGeoId]);

  useEffect(() => {
    if (!selectedGeoId) return;

    let cancelled = false;
    setHistoryLoading(true);
    setHistoryError(null);

    void getHistory(selectedGeoId, historyWindow)
      .then((h) => {
        if (!cancelled) setHistory(h);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setHistory(null);
          setHistoryError(
            e instanceof Error ? e.message : "Failed to load history."
          );
        }
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedGeoId, historyWindow]);

  async function handleExplain() {
    if (!selectedGeoId) return;
    setExplanationLoading(true);
    setExplanationError(null);
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";
      const res = await fetch(
        `${baseUrl}/explain/${encodeURIComponent(selectedGeoId)}/stream`
      );

      if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new Error(`PulseIQ API ${res.status}: ${detail}`);
      }

      if (!res.body) {
        throw new Error("Streaming response body unavailable.");
      }

      setExplanation("");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullExplanation = "";

      const appendEvents = (events: string[]) => {
        if (events.length === 0) return;
        fullExplanation = [fullExplanation, ...events]
          .filter(Boolean)
          .join("\n\n");
        setExplanation(fullExplanation);
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const { events, rest } = readSseEvents(buffer);
        buffer = rest;
        appendEvents(events);
      }

      buffer += decoder.decode();
      const { events, rest } = readSseEvents(`${buffer}\n\n`);
      appendEvents(events);

      const trailingEvent = parseSseEvent(rest);
      if (trailingEvent) {
        appendEvents([trailingEvent]);
      }
    } catch (e) {
      setExplanation("");
      setExplanationError(
        e instanceof Error ? e.message : "Failed to generate explanation."
      );
    } finally {
      setExplanationLoading(false);
    }
  }

  if (!selectedGeoId) return <EmptyPanel />;

  const headlineScore = score?.ess_score.toFixed(1) ?? "--";

  return (
    <aside
      style={{
        width: 280,
        minWidth: 280,
        height: "100%",
        background: "var(--surface)",
        borderLeft: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
      }}
    >
      {/* 1 · HEADER */}
      <Section>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            gap: 8,
          }}
        >
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: "9px",
              color: "var(--text3)",
              letterSpacing: "2px",
              textTransform: "uppercase",
              flexShrink: 0,
            }}
          >
            INTEL PANEL
          </span>
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: "10px",
              color: "var(--warn)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              textAlign: "right",
            }}
            title={score?.geo_name ?? selectedGeoId}
          >
            {score?.geo_name ?? selectedGeoId}
          </span>
        </div>
      </Section>

      {/* 2 · ESS SCORE */}
      <Section>
        <SectionLabel text="ESS SCORE" />
        <div style={{ display: "flex", alignItems: "baseline" }}>
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: "36px",
              fontWeight: 700,
              lineHeight: 1,
              color: scoreLoading
                ? "var(--text3)"
                : scoreBandColor(score?.score_band),
            }}
          >
            {scoreLoading ? "..." : headlineScore}
          </span>
          {score?.score_band && !scoreLoading && (
            <ScoreBandPill band={score.score_band} />
          )}
        </div>
        <div
          style={{
            marginTop: 8,
            fontFamily: "var(--mono)",
            fontSize: "11px",
            color: scoreLoading
              ? "var(--text3)"
              : deltaColor(score?.delta_7d ?? null),
          }}
        >
          {scoreLoading
            ? "Loading..."
            : score
              ? formatDelta(score.delta_7d)
              : "\u2014"}
        </div>
        {scoreError && (
          <div
            style={{
              marginTop: 8,
              color: "var(--danger)",
              fontFamily: "var(--mono)",
              fontSize: "10px",
            }}
          >
            {scoreError}
          </div>
        )}
      </Section>

      {/* 3 · SIGNAL TIERS */}
      <Section>
        <SectionLabel text="SIGNAL TIERS" />
        <TierBars
          tier1={score?.tier1_score ?? null}
          tier2={score?.tier2_score ?? null}
          tier3={score?.tier3_score ?? null}
        />
      </Section>

      {/* 4 · TOP DRIVERS */}
      <Section>
        <SectionLabel text="TOP DRIVERS" />
        <ShapWaterfall
          shapValues={score?.shap_values ?? {}}
          loading={scoreLoading}
        />
      </Section>

      {/* 5 · DATA STATUS */}
      <Section>
        <SectionLabel text="DATA STATUS" />
        {scoreLoading ? (
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: "10px",
              color: "var(--text3)",
            }}
          >
            &mdash;
          </span>
        ) : (
          <ConfidenceBadge
            confidence={score?.confidence ?? "low"}
            missingSources={score?.missing_sources ?? []}
            staleSources={score?.stale_sources ?? []}
          />
        )}
      </Section>

      {/* TREND */}
      <Section>
        <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
          {(["7d", "30d", "90d"] as HistoryWindow[]).map((opt) => (
            <WindowButton
              key={opt}
              value={opt}
              current={historyWindow}
              onSelect={setHistoryWindow}
            />
          ))}
        </div>
        <TrendChart history={history} loading={historyLoading} />
        {historyError && (
          <div
            style={{
              marginTop: 8,
              color: "var(--danger)",
              fontFamily: "var(--mono)",
              fontSize: "10px",
            }}
          >
            {historyError}
          </div>
        )}
      </Section>

      {/* 6 · AI BRIEFING */}
      <Section flex>
        <SectionLabel text="AI BRIEFING" />

        <button
          onClick={handleExplain}
          disabled={explanationLoading}
          style={{
            padding: "8px 12px",
            borderRadius: 4,
            border: "1px solid var(--accent)",
            background: explanationLoading
              ? "rgba(0,212,170,0.08)"
              : "transparent",
            color: "var(--accent)",
            fontFamily: "var(--mono)",
            fontSize: "11px",
            cursor: explanationLoading ? "wait" : "pointer",
            alignSelf: "flex-start",
          }}
        >
          {explanationLoading ? "Generating..." : "Generate explanation"}
        </button>

        {explanationError && (
          <div
            style={{
              marginTop: 10,
              border: "1px solid rgba(229,62,62,0.35)",
              background: "rgba(229,62,62,0.08)",
              borderRadius: 4,
              padding: "10px 12px",
              color: "var(--danger)",
              fontFamily: "var(--mono)",
              fontSize: "10px",
              lineHeight: 1.7,
            }}
          >
            {explanationError}
          </div>
        )}

        {!explanation && !explanationLoading && !explanationError && (
          <div
            style={{
              marginTop: 10,
              color: "var(--text3)",
              fontFamily: "var(--mono)",
              fontSize: "10px",
              lineHeight: 1.8,
            }}
          >
            Generate explanation to load the four-section model answer and
            supporting articles.
          </div>
        )}

        {explanation && (
          <pre
            style={{
              marginTop: 12,
              padding: "10px 12px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "rgba(255,255,255,0.02)",
              color: "var(--text)",
              fontFamily: "var(--mono)",
              fontSize: "10px",
              lineHeight: 1.8,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {explanation}
          </pre>
        )}

        {!explanation && explanationLoading && (
          <pre
            style={{
              marginTop: 12,
              padding: "10px 12px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "rgba(255,255,255,0.02)",
              color: "var(--text3)",
              fontFamily: "var(--mono)",
              fontSize: "10px",
              lineHeight: 1.8,
              whiteSpace: "pre-wrap",
            }}
          >
            Awaiting model output...
          </pre>
        )}
      </Section>
    </aside>
  );
}
