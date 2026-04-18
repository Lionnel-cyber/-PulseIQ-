"use client";

import type { ScoreBand, ScoreResponse } from "@/lib/types";

interface EventCardsProps {
  scores: ScoreResponse[];
  loading?: boolean;
  selectedGeoId?: string | null;
  onGeoSelect: (geoId: string) => void;
}

interface EventCardProps {
  score: ScoreResponse;
  selected: boolean;
  onClick: () => void;
}

function getBandColor(band: ScoreBand): string {
  switch (band) {
    case "critical":
      return "var(--danger)";
    case "high":
      return "var(--warn)";
    case "elevated":
      return "var(--warn)";
    default:
      return "var(--accent2)";
  }
}

function getBandLabel(band: ScoreBand): string {
  switch (band) {
    case "critical":
      return "CRITICAL";
    case "high":
      return "HIGH";
    case "elevated":
      return "ELEVATED";
    default:
      return "MONITOR";
  }
}

function formatDelta(delta: number | null): { text: string; color: string } {
  if (delta === null) {
    return { text: "NO DELTA", color: "var(--text3)" };
  }

  if (delta > 0) {
    return { text: `+${Math.abs(delta).toFixed(1)} / 7D`, color: "var(--danger)" };
  }

  if (delta < 0) {
    return { text: `-${Math.abs(delta).toFixed(1)} / 7D`, color: "var(--accent)" };
  }

  return { text: "FLAT / 7D", color: "var(--text2)" };
}

function SkeletonCard() {
  return (
    <div
      style={{
        flex: 1,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderLeft: "3px solid var(--border2)",
        borderRadius: "3px",
        padding: "8px 10px",
        minWidth: 0,
      }}
    >
      <div
        style={{
          width: "60%",
          height: "8px",
          background: "var(--border2)",
          borderRadius: 2,
          marginBottom: 6,
        }}
      />
      <div
        style={{
          width: "85%",
          height: "10px",
          background: "var(--border2)",
          borderRadius: 2,
          marginBottom: 8,
        }}
      />
      <div
        style={{
          width: "40%",
          height: "20px",
          background: "var(--border2)",
          borderRadius: 2,
        }}
      />
    </div>
  );
}

function EventCard({ score, selected, onClick }: EventCardProps) {
  const color = getBandColor(score.score_band);
  const delta = formatDelta(score.delta_7d);
  const missingCount = score.missing_sources.length;
  const geoTag = score.geo_level.toUpperCase();

  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        background: "var(--surface)",
        border: `1px solid ${selected ? "var(--accent)" : "var(--border)"}`,
        borderLeft: `3px solid ${selected ? "var(--accent)" : color}`,
        borderRadius: "3px",
        padding: "8px 10px",
        cursor: "pointer",
        minWidth: 0,
        textAlign: "left",
        transition: "border-color 0.15s",
      }}
    >
      <div
        style={{
          fontSize: "9px",
          letterSpacing: "1px",
          color,
          fontFamily: "var(--mono)",
          marginBottom: "4px",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {getBandLabel(score.score_band)} | {geoTag}
      </div>

      <div
        style={{
          fontSize: "11px",
          color: "var(--text)",
          fontWeight: "bold",
          fontFamily: "var(--mono)",
          marginBottom: "4px",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {score.geo_name}
      </div>

      <div
        style={{
          fontSize: "20px",
          fontWeight: "bold",
          fontFamily: "var(--mono)",
          color,
          lineHeight: 1,
          marginBottom: "4px",
        }}
      >
        {score.ess_score.toFixed(1)}
      </div>

      <div
        style={{
          fontSize: "10px",
          fontFamily: "var(--mono)",
          color: delta.color,
          marginBottom: "4px",
        }}
      >
        {delta.text}
      </div>

      <div
        style={{
          fontSize: "9px",
          color: "var(--text3)",
          fontFamily: "var(--mono)",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {score.confidence.toUpperCase()}
        {missingCount > 0 ? ` | ${missingCount} missing` : ""}
      </div>
    </button>
  );
}

export default function EventCards({
  scores,
  loading = false,
  selectedGeoId,
  onGeoSelect,
}: EventCardsProps) {
  const visible = [...scores]
    .sort((left, right) => right.ess_score - left.ess_score)
    .slice(0, 6);

  return (
    <div
      style={{
        display: "flex",
        gap: "6px",
        padding: "0 12px 12px",
        minHeight: "112px",
      }}
    >
      {loading ? (
        Array.from({ length: 6 }, (_, index) => <SkeletonCard key={index} />)
      ) : visible.length === 0 ? (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "10px",
            color: "var(--text3)",
            fontFamily: "var(--mono)",
            letterSpacing: "0.05em",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "3px",
          }}
        >
          NO EVENTS MATCH CURRENT FILTERS
        </div>
      ) : (
        visible.map((score) => (
          <EventCard
            key={score.geo_id}
            score={score}
            selected={score.geo_id === selectedGeoId}
            onClick={() => onGeoSelect(score.geo_id)}
          />
        ))
      )}
    </div>
  );
}
