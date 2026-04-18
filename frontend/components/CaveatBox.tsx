"use client";

interface CaveatBoxProps {
  caveats: string[];
}

export default function CaveatBox({ caveats }: CaveatBoxProps) {
  const isEmpty =
    caveats.length === 0 ||
    (caveats.length === 1 && caveats[0].toLowerCase() === "none identified");

  return (
    <div
      style={{
        border: "1px solid rgba(229,62,62,0.30)",
        background: "rgba(229,62,62,0.08)",
        borderRadius: 4,
        padding: "10px 12px",
      }}
    >
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: "9px",
          color: "var(--text3)",
          textTransform: "uppercase",
          letterSpacing: "1.5px",
          marginBottom: 6,
        }}
      >
        CAVEATS
      </div>

      {isEmpty ? (
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: "11px",
            color: "var(--text3)",
          }}
        >
          None identified
        </div>
      ) : (
        <ul
          style={{
            margin: 0,
            paddingLeft: 14,
            color: "var(--text2)",
            fontSize: "11px",
            lineHeight: 1.7,
          }}
        >
          {caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
