"use client";

interface ShapWaterfallProps {
  shapValues: Record<string, number>;
  loading: boolean;
}

function truncate(name: string, max: number): string {
  return name.length > max ? name.slice(0, max) + "\u2026" : name;
}

export default function ShapWaterfall({ shapValues, loading }: ShapWaterfallProps) {
  if (loading) {
    return (
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: "10px",
          color: "var(--text3)",
        }}
      >
        Loading SHAP...
      </div>
    );
  }

  const entries = Object.entries(shapValues)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3);

  if (entries.length === 0) {
    return (
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: "10px",
          color: "var(--text3)",
        }}
      >
        No SHAP data.
      </div>
    );
  }

  const maxAbs = Math.max(Math.abs(entries[0][1]), 1e-9);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {entries.map(([feature, value]) => {
        const pct = (Math.abs(value) / maxAbs) * 100;
        const fillColor = value >= 0 ? "var(--danger)" : "var(--accent)";
        const sign = value >= 0 ? "+" : "";

        return (
          <div
            key={feature}
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            <span
              title={feature}
              style={{
                fontFamily: "var(--mono)",
                fontSize: "9px",
                color: "var(--text3)",
                width: 80,
                flexShrink: 0,
                overflow: "hidden",
                whiteSpace: "nowrap",
              }}
            >
              {truncate(feature, 20)}
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
                color: fillColor,
                width: 40,
                textAlign: "right",
                flexShrink: 0,
              }}
            >
              {sign}{value.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
