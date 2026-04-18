"use client";

import type { Confidence } from "@/lib/types";

interface ConfidenceBadgeProps {
  confidence: Confidence;
  missingSources: string[];
  staleSources: string[];
}

const CONFIDENCE_STYLES: Record<
  Confidence,
  { color: string; bg: string; border: string; label: string }
> = {
  high: {
    color: "var(--accent)",
    bg: "rgba(0,212,170,0.15)",
    border: "rgba(0,212,170,0.30)",
    label: "HIGH",
  },
  medium: {
    color: "var(--warn)",
    bg: "rgba(245,166,35,0.15)",
    border: "rgba(245,166,35,0.30)",
    label: "MEDIUM",
  },
  low: {
    color: "var(--danger)",
    bg: "rgba(229,62,62,0.15)",
    border: "rgba(229,62,62,0.30)",
    label: "LOW",
  },
};

export default function ConfidenceBadge({
  confidence,
  missingSources,
  staleSources,
}: ConfidenceBadgeProps) {
  const s = CONFIDENCE_STYLES[confidence];
  const hasTags = missingSources.length > 0 || staleSources.length > 0;

  return (
    <div>
      <span
        style={{
          display: "inline-block",
          padding: "3px 8px",
          borderRadius: 999,
          border: `1px solid ${s.border}`,
          background: s.bg,
          color: s.color,
          fontFamily: "var(--mono)",
          fontSize: "10px",
          fontWeight: 700,
          letterSpacing: "1.5px",
        }}
      >
        {s.label}
      </span>

      {hasTags && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 4,
            marginTop: 8,
          }}
        >
          {missingSources.map((src) => (
            <span
              key={`missing-${src}`}
              style={{
                fontFamily: "var(--mono)",
                fontSize: "9px",
                padding: "2px 6px",
                borderRadius: 3,
                border: "1px solid rgba(229,62,62,0.25)",
                background: "rgba(229,62,62,0.10)",
                color: "var(--danger)",
              }}
            >
              {src}
            </span>
          ))}
          {staleSources.map((src) => (
            <span
              key={`stale-${src}`}
              style={{
                fontFamily: "var(--mono)",
                fontSize: "9px",
                padding: "2px 6px",
                borderRadius: 3,
                border: "1px solid rgba(245,166,35,0.25)",
                background: "rgba(245,166,35,0.10)",
                color: "var(--warn)",
              }}
            >
              {src}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
