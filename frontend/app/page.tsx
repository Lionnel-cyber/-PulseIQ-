"use client";

import { useEffect, useMemo, useState } from "react";
import EventCards from "@/components/EventCards";
import IntelPanel from "@/components/IntelPanel";
import Sidebar from "@/components/Sidebar";
import WorldMap from "@/components/WorldMap";
import { getMapScores, getScoreSnapshot } from "@/lib/api";
import type {
  FilterState,
  MapScoreResponse,
  MappableScore,
  ScoreResponse,
} from "@/lib/types";
import { DEFAULT_FILTER_STATE as DEFAULT_FILTERS } from "@/lib/types";

function matchesScoreFilters(score: ScoreResponse, filters: FilterState): boolean {
  const matchesGeoLevel =
    filters.geoLevels.length === 0 ||
    filters.geoLevels.includes(score.geo_level as "metro" | "county" | "zip");

  const matchesScore = score.ess_score >= filters.minScore;
  const matchesConfidence =
    filters.confidenceLevels.length === 0 ||
    filters.confidenceLevels.includes(score.confidence);

  return matchesGeoLevel && matchesScore && matchesConfidence;
}

function matchesMapFilters(score: MapScoreResponse, filters: FilterState): boolean {
  const matchesScore = score.ess_score >= filters.minScore;
  const matchesConfidence =
    filters.confidenceLevels.length === 0 ||
    filters.confidenceLevels.includes(score.confidence);

  return matchesScore && matchesConfidence;
}

function extractStateCode(score: ScoreResponse): string | null {
  const geoName = score.geo_name.toUpperCase();
  const geoId = score.geo_id.toUpperCase();

  const geoNameMatch = geoName.match(/,\s*([A-Z]{2})(?:-[A-Z]{2})*$/);
  if (geoNameMatch) {
    return geoNameMatch[1];
  }

  const geoIdMatch = geoId.match(/-([A-Z]{2})(?:-[A-Z]{2})*$/);
  if (geoIdMatch) {
    return geoIdMatch[1];
  }

  return null;
}

function deriveFallbackMapScores(
  snapshotScores: ScoreResponse[]
): MapScoreResponse[] {
  const derived = new Map<string, MapScoreResponse>();

  snapshotScores.forEach((score) => {
    const stateCode = extractStateCode(score);
    if (!stateCode) {
      return;
    }

    const current = derived.get(stateCode);
    if (!current || score.ess_score > current.ess_score) {
      derived.set(stateCode, {
        state_code: stateCode,
        geo_id: stateCode,
        geo_name: `${stateCode} drilldown`,
        geo_level: "state",
        run_date: score.run_date,
        ess_score: score.ess_score,
        delta_7d: score.delta_7d,
        confidence: score.confidence,
        missing_sources: score.missing_sources,
        granularity_warning: score.granularity_warning,
        drilldown_geo_id: score.geo_id,
      });
    }
  });

  return Array.from(derived.values());
}

function ApiErrorPanel({ message }: { message: string }) {
  return (
    <div
      style={{
        minHeight: "calc(100vh - 40px)",
        display: "grid",
        gridTemplateColumns: "200px minmax(0, 1fr) 280px",
        background: "var(--bg)",
      }}
    >
      <Sidebar value={DEFAULT_FILTERS} onFiltersChange={() => undefined} />
      <div
        style={{
          padding: "18px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            maxWidth: 640,
            width: "100%",
            border: "1px solid rgba(229,62,62,0.35)",
            background: "rgba(229,62,62,0.08)",
            borderRadius: "6px",
            padding: "18px 20px",
            color: "var(--text)",
          }}
        >
          <div
            style={{
              fontSize: "18px",
              fontWeight: 700,
              marginBottom: 10,
            }}
          >
            Could not reach the PulseIQ API
          </div>
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: "11px",
              lineHeight: 1.8,
              color: "var(--text2)",
            }}
          >
            {message}
          </div>
          <div
            style={{
              marginTop: 14,
              fontFamily: "var(--mono)",
              fontSize: "11px",
              color: "var(--text)",
            }}
          >
            Start it with:
          </div>
          <pre
            style={{
              marginTop: 8,
              padding: "10px 12px",
              background: "#0a0c0f",
              border: "1px solid var(--border)",
              borderRadius: "4px",
              color: "var(--accent)",
              fontFamily: "var(--mono)",
              fontSize: "12px",
              overflowX: "auto",
            }}
          >
            uvicorn src.api.main:app --reload
          </pre>
        </div>
      </div>
      <IntelPanel selectedGeoId={null} />
    </div>
  );
}

function SectionFrame({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <section
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "6px",
        display: "flex",
        flexDirection: "column",
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
        {label}
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>{children}</div>
    </section>
  );
}

export default function Home() {
  const [selectedGeoId, setSelectedGeoId] = useState<string | null>(null);
  const [sidebarFilters, setSidebarFilters] =
    useState<FilterState>(DEFAULT_FILTERS);
  const [snapshotScores, setSnapshotScores] = useState<MappableScore[]>([]);
  const [mapScores, setMapScores] = useState<MapScoreResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchMonitorData() {
      try {
        if (!cancelled) {
          setLoading(true);
          setError(null);
        }

        const [nextSnapshotScores, nextMapScores] = await Promise.all([
          getScoreSnapshot(500),
          getMapScores(),
        ]);

        if (!cancelled) {
          setSnapshotScores(nextSnapshotScores);
          setMapScores(nextMapScores);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(
            fetchError instanceof Error
              ? fetchError.message
              : "Failed to load monitor data."
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void fetchMonitorData();
    const timer = setInterval(() => void fetchMonitorData(), 60_000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const filteredSnapshotScores = useMemo(() => {
    return snapshotScores.filter((score) => matchesScoreFilters(score, sidebarFilters));
  }, [sidebarFilters, snapshotScores]);

  const resolvedMapScores = useMemo(() => {
    return mapScores.length > 0 ? mapScores : deriveFallbackMapScores(snapshotScores);
  }, [mapScores, snapshotScores]);

  const filteredMapScores = useMemo(() => {
    return resolvedMapScores.filter((score) => matchesMapFilters(score, sidebarFilters));
  }, [resolvedMapScores, sidebarFilters]);

  if (error) {
    return <ApiErrorPanel message={error} />;
  }

  return (
    <div
      style={{
        minHeight: "calc(100vh - 40px)",
        display: "grid",
        gridTemplateColumns: "200px minmax(0, 1fr) 280px",
        background: "var(--bg)",
      }}
    >
      <Sidebar value={sidebarFilters} onFiltersChange={setSidebarFilters} />

      <div
        style={{
          padding: "12px",
          minWidth: 0,
          display: "grid",
          gridTemplateRows: "minmax(360px, 1fr) auto",
          gap: "12px",
        }}
      >
        <SectionFrame label="Map">
          <WorldMap
            scores={filteredSnapshotScores}
            mapScores={filteredMapScores}
            selectedGeoId={selectedGeoId}
            onGeoSelect={setSelectedGeoId}
          />
        </SectionFrame>

        <SectionFrame label="Event Cards">
          <EventCards
            scores={filteredSnapshotScores}
            loading={loading}
            selectedGeoId={selectedGeoId}
            onGeoSelect={setSelectedGeoId}
          />
        </SectionFrame>
      </div>

      <IntelPanel selectedGeoId={selectedGeoId} />
    </div>
  );
}
