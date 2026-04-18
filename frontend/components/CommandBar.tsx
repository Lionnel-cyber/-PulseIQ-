"use client";

import { useEffect, useState } from "react";
import { getHealth, getTopScores } from "@/lib/api";
import type { HealthResponse, ScoreResponse } from "@/lib/types";

interface BarStats {
  geosScored: number;
  elevated: number;
  critical: number;
  alertsFired: number;
  modelVersion: string;
  status: HealthResponse["status"];
}

function utcNow(): string {
  const date = new Date();
  const hours = date.getUTCHours().toString().padStart(2, "0");
  const minutes = date.getUTCMinutes().toString().padStart(2, "0");
  return `${hours}:${minutes} UTC`;
}

function deriveStats(
  health: HealthResponse,
  scores: ScoreResponse[]
): BarStats {
  return {
    geosScored: scores.length,
    elevated: scores.filter(
      (score) => score.score_band === "elevated" || score.score_band === "high"
    ).length,
    critical: scores.filter((score) => score.score_band === "critical").length,
    alertsFired: health.stale_sources.length,
    modelVersion: scores[0]?.model_version ?? "unknown",
    status: health.status,
  };
}

function StatPill({
  label,
  value,
  alertColor,
}: {
  label: string;
  value: number | undefined;
  alertColor?: string;
}) {
  const active = alertColor && value !== undefined && value > 0;

  return (
    <span
      style={{
        background: "var(--surface2)",
        border: `1px solid ${active ? alertColor : "var(--border)"}`,
        borderRadius: "2px",
        padding: "2px 8px",
        fontSize: "11px",
        letterSpacing: "0.04em",
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ color: "var(--text3)" }}>{label}:</span>{" "}
      <span style={{ color: active ? alertColor : "var(--text)" }}>
        {value !== undefined ? value : "--"}
      </span>
    </span>
  );
}

export default function CommandBar() {
  const [stats, setStats] = useState<BarStats | null>(null);
  const [clock, setClock] = useState<string>(utcNow());

  useEffect(() => {
    async function fetchStats() {
      try {
        const [health, scores] = await Promise.all([getHealth(), getTopScores(100)]);
        setStats(deriveStats(health, scores));
      } catch {
        setStats(null);
      }
    }

    void fetchStats();

    const dataTimer = setInterval(() => void fetchStats(), 60_000);
    const clockTimer = setInterval(() => setClock(utcNow()), 60_000);

    return () => {
      clearInterval(dataTimer);
      clearInterval(clockTimer);
    };
  }, []);

  return (
    <header
      style={{
        height: "40px",
        background: "var(--surface)",
        borderBottom: "1px solid var(--border)",
        fontFamily: "var(--mono)",
        flexShrink: 0,
      }}
      className="flex items-center px-4 gap-3 text-xs"
    >
      <div className="flex items-center gap-2 shrink-0">
        <span
          style={{
            color: "var(--accent)",
            letterSpacing: "0.15em",
            fontWeight: "bold",
            fontSize: "13px",
          }}
        >
          PULSEIQ
        </span>
        <span className="flex items-center gap-1">
          <span className="pulse-dot" />
          <span
            style={{
              color: "var(--accent)",
              fontSize: "10px",
              letterSpacing: "0.1em",
            }}
          >
            LIVE
          </span>
        </span>
      </div>

      <span style={{ color: "var(--border2)", userSelect: "none" }}>|</span>

      <div className="flex items-center gap-2 flex-1 overflow-hidden">
        <StatPill label="GEOS" value={stats?.geosScored} />
        <StatPill label="ELEVATED" value={stats?.elevated} alertColor="var(--warn)" />
        <StatPill
          label="CRITICAL"
          value={stats?.critical}
          alertColor="var(--danger)"
        />
        <StatPill label="ALERTS" value={stats?.alertsFired} alertColor="var(--warn)" />
      </div>

      <div
        className="flex items-center gap-3 shrink-0 ml-auto"
        style={{ color: "var(--text2)", fontSize: "11px" }}
      >
        {stats && (
          <span>
            mdl: <span style={{ color: "var(--text)" }}>{stats.modelVersion.slice(0, 12)}</span>
          </span>
        )}
        <span>{clock}</span>
      </div>
    </header>
  );
}
