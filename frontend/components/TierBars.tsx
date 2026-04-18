"use client";

interface TierBarsProps {
  tier1: number | null;
  tier2: number | null;
  tier3: number | null;
}

interface TierRowProps {
  label: string;
  value: number | null;
  fillColor: string;
}

function TierRow({ label, value, fillColor }: TierRowProps) {
  const pct = value == null ? 0 : Math.min(Math.max(value / 100, 0), 1) * 100;
  const display = value == null ? "--" : value.toFixed(1);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span
        style={{
          fontFamily: "var(--mono)",
          fontSize: "9px",
          color: "var(--text3)",
          textTransform: "uppercase",
          letterSpacing: "1.5px",
          width: 80,
          flexShrink: 0,
        }}
      >
        {label}
      </span>

      <div
        style={{
          flex: 1,
          height: 3,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 2,
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: 3,
            width: `${pct}%`,
            background: fillColor,
            borderRadius: 2,
          }}
        />
      </div>

      <span
        style={{
          fontFamily: "var(--mono)",
          fontSize: "10px",
          color: "var(--text2)",
          width: 32,
          textAlign: "right",
          flexShrink: 0,
        }}
      >
        {display}
      </span>
    </div>
  );
}

export default function TierBars({ tier1, tier2, tier3 }: TierBarsProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <TierRow label="Tier 1 · Hard"   value={tier1} fillColor="var(--danger)"  />
      <TierRow label="Tier 2 · Med"    value={tier2} fillColor="var(--warn)"    />
      <TierRow label="Tier 3 · Soft"   value={tier3} fillColor="var(--accent2)" />
    </div>
  );
}
