"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { Confidence, FilterState } from "@/lib/types";

interface SidebarProps {
  value: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  showMonitorControls?: boolean;
}

const SECTION_LABEL: React.CSSProperties = {
  fontSize: "9px",
  letterSpacing: "2px",
  color: "var(--text3)",
  fontFamily: "var(--mono)",
  textTransform: "uppercase",
  padding: "14px 16px 5px",
  display: "block",
};

function Divider() {
  return (
    <div
      style={{ borderTop: "1px solid var(--border)", margin: "6px 0" }}
    />
  );
}

function NavItem({ href, label }: { href: string; label: string }) {
  const pathname = usePathname();
  const active = pathname === href;

  return (
    <Link
      href={href}
      style={{
        display: "block",
        padding: "7px 16px",
        fontSize: "12px",
        color: active ? "var(--accent)" : "var(--text2)",
        background: active ? "rgba(0,212,170,0.06)" : "transparent",
        borderLeft: `2px solid ${active ? "var(--accent)" : "transparent"}`,
        fontFamily: "var(--mono)",
        textDecoration: "none",
        letterSpacing: "0.03em",
        transition: "color 0.15s, background 0.15s",
      }}
    >
      {label}
    </Link>
  );
}

function TierRow({ dot, label }: { dot: string; label: string }) {
  return (
    <div
      style={{
        padding: "5px 16px",
        display: "flex",
        alignItems: "center",
        gap: "8px",
        fontSize: "11px",
        color: "var(--text2)",
        fontFamily: "var(--mono)",
        userSelect: "none",
      }}
    >
      <span
        style={{
          width: "6px",
          height: "6px",
          borderRadius: "50%",
          background: dot,
          display: "inline-block",
          flexShrink: 0,
        }}
      />
      {label}
    </div>
  );
}

function FilterGroup({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ padding: "6px 16px 10px" }}>
      <div
        style={{
          fontSize: "9px",
          letterSpacing: "1.5px",
          color: "var(--text3)",
          fontFamily: "var(--mono)",
          marginBottom: "6px",
          textTransform: "uppercase",
        }}
      >
        {label}
      </div>
      <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
        {children}
      </div>
    </div>
  );
}

function FilterToggle({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "2px 8px",
        fontSize: "11px",
        borderRadius: "2px",
        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
        color: active ? "var(--accent)" : "var(--text2)",
        background: active ? "rgba(0,212,170,0.08)" : "transparent",
        fontFamily: "var(--mono)",
        cursor: "pointer",
        letterSpacing: "0.03em",
        transition: "border-color 0.15s, color 0.15s",
      }}
    >
      {label}
    </button>
  );
}

export default function Sidebar({
  value,
  onFiltersChange,
  showMonitorControls = true,
}: SidebarProps) {
  const { geoLevels, minScore, confidenceLevels } = value;

  function toggleGeo(level: "metro" | "county" | "zip") {
    const next = geoLevels.includes(level)
      ? geoLevels.filter((geo) => geo !== level)
      : [...geoLevels, level];
    onFiltersChange({ geoLevels: next, minScore, confidenceLevels });
  }

  function setThreshold(score: 0 | 60 | 75) {
    onFiltersChange({ geoLevels, minScore: score, confidenceLevels });
  }

  function toggleConfidence(level: Confidence) {
    const next = confidenceLevels.includes(level)
      ? confidenceLevels.filter((confidence) => confidence !== level)
      : [...confidenceLevels, level];
    onFiltersChange({ geoLevels, minScore, confidenceLevels: next });
  }

  return (
    <nav
      style={{
        width: "200px",
        minWidth: "200px",
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
        height: "100%",
      }}
    >
      <span style={SECTION_LABEL}>View</span>
      <NavItem href="/" label="Monitor" />
      <NavItem href="/health" label="Pipeline health" />
      <NavItem href="/alerts" label="Alert history" />
      <NavItem href="/drift" label="Model drift" />

      {showMonitorControls && (
        <>
          <Divider />

          <span style={SECTION_LABEL}>Signal Tiers</span>
          <TierRow dot="var(--danger)" label="Tier 1 | BLS + FRED" />
          <TierRow dot="var(--warn)" label="Tier 2 | Prices" />
          <TierRow dot="var(--text3)" label="Tier 3 | Search" />

          <Divider />

          <span style={SECTION_LABEL}>Filters</span>

          <FilterGroup label="Geo Level">
            <FilterToggle
              label="Metro"
              active={geoLevels.includes("metro")}
              onClick={() => toggleGeo("metro")}
            />
            <FilterToggle
              label="County"
              active={geoLevels.includes("county")}
              onClick={() => toggleGeo("county")}
            />
            <FilterToggle
              label="ZIP"
              active={geoLevels.includes("zip")}
              onClick={() => toggleGeo("zip")}
            />
          </FilterGroup>

          <FilterGroup label="Threshold">
            <FilterToggle
              label="All"
              active={minScore === 0}
              onClick={() => setThreshold(0)}
            />
            <FilterToggle
              label=">=60"
              active={minScore === 60}
              onClick={() => setThreshold(60)}
            />
            <FilterToggle
              label=">=75"
              active={minScore === 75}
              onClick={() => setThreshold(75)}
            />
          </FilterGroup>

          <FilterGroup label="Confidence">
            <FilterToggle
              label="High"
              active={confidenceLevels.includes("high")}
              onClick={() => toggleConfidence("high")}
            />
            <FilterToggle
              label="Medium"
              active={confidenceLevels.includes("medium")}
              onClick={() => toggleConfidence("medium")}
            />
            <FilterToggle
              label="Low"
              active={confidenceLevels.includes("low")}
              onClick={() => toggleConfidence("low")}
            />
          </FilterGroup>
        </>
      )}
    </nav>
  );
}
