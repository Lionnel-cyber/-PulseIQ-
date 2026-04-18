"use client";

import dynamic from "next/dynamic";
import { useMemo, useState, useSyncExternalStore } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import type {
  Confidence,
  MapScoreResponse,
  MappableScore,
  ScoreBand,
} from "@/lib/types";
import type { MapPoint } from "./DeckGLMap";

const DeckGLMap = dynamic(() => import("./DeckGLMap"), { ssr: false });

interface WorldMapProps {
  scores: MappableScore[];
  mapScores: MapScoreResponse[];
  selectedGeoId?: string | null;
  onGeoSelect?: (geoId: string) => void;
}

const US_TOPO = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json";

const FIPS_TO_STATE: Record<string, string> = {
  "01": "AL",
  "02": "AK",
  "04": "AZ",
  "05": "AR",
  "06": "CA",
  "08": "CO",
  "09": "CT",
  "10": "DE",
  "11": "DC",
  "12": "FL",
  "13": "GA",
  "15": "HI",
  "16": "ID",
  "17": "IL",
  "18": "IN",
  "19": "IA",
  "20": "KS",
  "21": "KY",
  "22": "LA",
  "23": "ME",
  "24": "MD",
  "25": "MA",
  "26": "MI",
  "27": "MN",
  "28": "MS",
  "29": "MO",
  "30": "MT",
  "31": "NE",
  "32": "NV",
  "33": "NH",
  "34": "NJ",
  "35": "NM",
  "36": "NY",
  "37": "NC",
  "38": "ND",
  "39": "OH",
  "40": "OK",
  "41": "OR",
  "42": "PA",
  "44": "RI",
  "45": "SC",
  "46": "SD",
  "47": "TN",
  "48": "TX",
  "49": "UT",
  "50": "VT",
  "51": "VA",
  "53": "WA",
  "54": "WV",
  "55": "WI",
  "56": "WY",
};

const STATE_CENTROIDS: Record<string, { longitude: number; latitude: number }> = {
  AL: { longitude: -86.7911, latitude: 32.8067 },
  AK: { longitude: -152.4044, latitude: 61.3707 },
  AZ: { longitude: -111.4312, latitude: 33.7298 },
  AR: { longitude: -92.3731, latitude: 34.9697 },
  CA: { longitude: -119.6816, latitude: 36.1162 },
  CO: { longitude: -105.3111, latitude: 39.0598 },
  CT: { longitude: -72.7554, latitude: 41.5978 },
  DE: { longitude: -75.5071, latitude: 39.3185 },
  DC: { longitude: -77.0369, latitude: 38.9072 },
  FL: { longitude: -81.6868, latitude: 27.7663 },
  GA: { longitude: -83.6431, latitude: 33.0406 },
  HI: { longitude: -157.4983, latitude: 21.0943 },
  ID: { longitude: -114.4788, latitude: 44.2405 },
  IL: { longitude: -88.9861, latitude: 40.3495 },
  IN: { longitude: -86.2583, latitude: 39.8494 },
  IA: { longitude: -93.2105, latitude: 42.0115 },
  KS: { longitude: -96.7265, latitude: 38.5266 },
  KY: { longitude: -84.6701, latitude: 37.6681 },
  LA: { longitude: -91.8678, latitude: 31.1695 },
  ME: { longitude: -69.3819, latitude: 44.6939 },
  MD: { longitude: -76.8021, latitude: 39.0639 },
  MA: { longitude: -71.5301, latitude: 42.2302 },
  MI: { longitude: -84.5361, latitude: 43.3266 },
  MN: { longitude: -93.9002, latitude: 45.6945 },
  MS: { longitude: -89.6787, latitude: 32.7416 },
  MO: { longitude: -92.2884, latitude: 38.4561 },
  MT: { longitude: -110.4544, latitude: 46.9219 },
  NE: { longitude: -98.2681, latitude: 41.1254 },
  NV: { longitude: -117.0554, latitude: 38.3135 },
  NH: { longitude: -71.5639, latitude: 43.4525 },
  NJ: { longitude: -74.521, latitude: 40.2989 },
  NM: { longitude: -106.2485, latitude: 34.8405 },
  NY: { longitude: -74.9481, latitude: 42.1657 },
  NC: { longitude: -79.8064, latitude: 35.6301 },
  ND: { longitude: -99.784, latitude: 47.5289 },
  OH: { longitude: -82.7649, latitude: 40.3888 },
  OK: { longitude: -96.9289, latitude: 35.5653 },
  OR: { longitude: -122.0709, latitude: 44.572 },
  PA: { longitude: -77.2098, latitude: 40.5908 },
  RI: { longitude: -71.5118, latitude: 41.6809 },
  SC: { longitude: -80.945, latitude: 33.8569 },
  SD: { longitude: -99.4388, latitude: 44.2998 },
  TN: { longitude: -86.6923, latitude: 35.7478 },
  TX: { longitude: -97.5635, latitude: 31.0545 },
  UT: { longitude: -111.8624, latitude: 40.15 },
  VT: { longitude: -72.7107, latitude: 44.0459 },
  VA: { longitude: -78.17, latitude: 37.7693 },
  WA: { longitude: -121.4905, latitude: 47.4009 },
  WV: { longitude: -80.9545, latitude: 38.4912 },
  WI: { longitude: -89.6165, latitude: 44.2685 },
  WY: { longitude: -107.3025, latitude: 42.756 },
};

function hasCoordinates(
  score: MappableScore
): score is MappableScore & { longitude: number; latitude: number } {
  return typeof score.longitude === "number" && typeof score.latitude === "number";
}

function useIsHydrated(): boolean {
  return useSyncExternalStore(
    () => () => undefined,
    () => true,
    () => false
  );
}

function getOpacity(confidence: Confidence): number {
  return confidence === "high" ? 0.85 : confidence === "medium" ? 0.6 : 0.3;
}

function interpolateChannel(start: number, end: number, ratio: number): number {
  return Math.round(start + (end - start) * ratio);
}

function getScoreColor(score: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(score, 100));

  if (clamped <= 50) {
    const ratio = clamped / 50;
    return [
      interpolateChannel(26, 245, ratio),
      interpolateChannel(152, 166, ratio),
      interpolateChannel(80, 35, ratio),
    ];
  }

  const ratio = (clamped - 50) / 50;
  return [
    interpolateChannel(245, 229, ratio),
    interpolateChannel(166, 62, ratio),
    interpolateChannel(35, 62, ratio),
  ];
}

function getFill(score: MapScoreResponse | undefined): string {
  if (!score) {
    return "#0f1318";
  }

  const [red, green, blue] = getScoreColor(score.ess_score);
  return `rgba(${red},${green},${blue},${getOpacity(score.confidence)})`;
}

function getScoreBand(score: number): ScoreBand {
  if (score >= 85) {
    return "critical";
  }
  if (score >= 75) {
    return "high";
  }
  if (score >= 60) {
    return "elevated";
  }
  return "low";
}

function formatDelta(delta: number | null): string {
  if (delta === null) {
    return "No 7d delta";
  }

  const prefix = delta > 0 ? "+" : "";
  return `${prefix}${delta.toFixed(1)} vs 7d`;
}

function StateOverviewMap({
  scores,
  selectedGeoId,
  onGeoSelect,
}: {
  scores: MapScoreResponse[];
  selectedGeoId?: string | null;
  onGeoSelect?: (geoId: string) => void;
}) {
  const byState = useMemo(() => {
    return new Map(scores.map((score) => [score.state_code, score]));
  }, [scores]);

  const latestRunDate = scores[0]?.run_date ?? "--";
  const selectedScore =
    scores.find((score) => score.drilldown_geo_id === selectedGeoId) ?? null;

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <ComposableMap
        projection="geoAlbersUsa"
        style={{ width: "100%", height: "100%", background: "var(--bg)" }}
      >
        <Geographies geography={US_TOPO}>
          {({ geographies }) =>
            geographies.map((geo) => {
              const stateCode = FIPS_TO_STATE[String(geo.id).padStart(2, "0")];
              const score = stateCode ? byState.get(stateCode) : undefined;
              const selected =
                score?.drilldown_geo_id != null &&
                score.drilldown_geo_id === selectedGeoId;

              return (
                <g
                  key={geo.rsmKey}
                  onClick={() => {
                    if (score) {
                      onGeoSelect?.(score.drilldown_geo_id);
                    }
                  }}
                  style={{ cursor: score ? "pointer" : "default" }}
                >
                  <Geography
                    geography={geo}
                    style={{
                      default: {
                        fill: getFill(score),
                        stroke: selected ? "#00d4aa" : "#1e2730",
                        strokeWidth: selected ? 1.6 : 0.7,
                        outline: "none",
                      },
                      hover: {
                        fill: score ? getFill(score) : "#141920",
                        stroke: selected ? "#00d4aa" : "#3a5068",
                        strokeWidth: selected ? 1.6 : 1,
                        outline: "none",
                      },
                      pressed: {
                        fill: score ? getFill(score) : "#141920",
                        stroke: "#00d4aa",
                        strokeWidth: 1.6,
                        outline: "none",
                      },
                    }}
                  />
                </g>
              );
            })
          }
        </Geographies>
      </ComposableMap>

      {selectedScore && (
        <div
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            background: "rgba(15,19,24,0.96)",
            border: "1px solid var(--border)",
            borderRadius: "4px",
            padding: "8px 10px",
            fontFamily: "var(--mono)",
            fontSize: "10px",
            color: "var(--text)",
            lineHeight: 1.6,
            maxWidth: 220,
          }}
        >
          <div style={{ fontWeight: "bold", marginBottom: 2 }}>
            {selectedScore.geo_name}
          </div>
          <div>ESS: {selectedScore.ess_score.toFixed(1)}</div>
          <div>{formatDelta(selectedScore.delta_7d)}</div>
          <div>Confidence: {selectedScore.confidence}</div>
          <div>
            Missing:{" "}
            {selectedScore.missing_sources.length > 0
              ? selectedScore.missing_sources.join(", ")
              : "none"}
          </div>
        </div>
      )}

      <div
        style={{
          position: "absolute",
          top: 10,
          right: 12,
          fontSize: "9px",
          letterSpacing: "1px",
          fontFamily: "var(--mono)",
          color: "var(--text3)",
          border: "1px solid var(--border)",
          padding: "3px 10px",
          textTransform: "uppercase",
        }}
      >
        State Overview
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 8,
          left: 12,
          fontSize: "9px",
          color: "var(--text3)",
          fontFamily: "var(--mono)",
          pointerEvents: "none",
          userSelect: "none",
          letterSpacing: "0.04em",
        }}
      >
        Data as of {latestRunDate} | Opacity = confidence | Click a highlighted
        state to drill down
      </div>
    </div>
  );
}

export default function WorldMap({
  scores,
  mapScores,
  selectedGeoId,
  onGeoSelect,
}: WorldMapProps) {
  const [viewMode, setViewMode] = useState<"flat" | "globe">("flat");
  const isHydrated = useIsHydrated();

  const pointScores = useMemo<MapPoint[]>(() => {
    const geoPoints = scores.filter(hasCoordinates).map((score) => ({
      geo_id: score.geo_id,
      geo_name: score.geo_name,
      ess_score: score.ess_score,
      score_band: score.score_band,
      confidence: score.confidence,
      missing_sources: score.missing_sources,
      run_date: score.run_date,
      longitude: score.longitude,
      latitude: score.latitude,
      population_at_risk: score.population_at_risk ?? null,
    }));

    if (geoPoints.length > 0) {
      return geoPoints;
    }

    const statePoints: MapPoint[] = [];
    mapScores.forEach((score) => {
      const centroid = STATE_CENTROIDS[score.state_code];
      if (!centroid) {
        return;
      }

      statePoints.push({
          geo_id: score.drilldown_geo_id,
          geo_name: score.geo_name,
          ess_score: score.ess_score,
          score_band: getScoreBand(score.ess_score),
          confidence: score.confidence,
          missing_sources: score.missing_sources,
          run_date: score.run_date,
          longitude: centroid.longitude,
          latitude: centroid.latitude,
          population_at_risk: null,
      });
    });

    return statePoints;
  }, [mapScores, scores]);

  const canUsePointMap = pointScores.length > 0;
  const supportsWebGl = useMemo(() => {
    if (!isHydrated || !canUsePointMap) {
      return false;
    }

    const canvas = document.createElement("canvas");
    const context =
      canvas.getContext("webgl") ?? canvas.getContext("experimental-webgl");

    return Boolean(context);
  }, [canUsePointMap, isHydrated]);

  const useFallback = !canUsePointMap || !supportsWebGl;

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        background: "var(--bg)",
        overflow: "hidden",
      }}
    >
      {!useFallback && canUsePointMap && (
        <div
          style={{
            position: "absolute",
            top: 10,
            right: 12,
            display: "flex",
            gap: "4px",
            zIndex: 10,
          }}
        >
          {(["flat", "globe"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              style={{
                padding: "3px 10px",
                fontSize: "9px",
                letterSpacing: "1px",
                fontFamily: "var(--mono)",
                background: "transparent",
                border: `1px solid ${
                  viewMode === mode ? "var(--accent)" : "var(--border)"
                }`,
                color: viewMode === mode ? "var(--accent)" : "var(--text3)",
                cursor: "pointer",
                textTransform: "uppercase",
              }}
            >
              {mode === "flat" ? "Flat Map" : "3D Globe"}
            </button>
          ))}
        </div>
      )}

      {useFallback ? (
        <StateOverviewMap
          scores={mapScores}
          selectedGeoId={selectedGeoId}
          onGeoSelect={onGeoSelect}
        />
      ) : (
        <DeckGLMap
          key={viewMode}
          points={pointScores}
          viewMode={viewMode}
          selectedGeoId={selectedGeoId}
          onGeoSelect={(geoId) => onGeoSelect?.(geoId)}
        />
      )}
    </div>
  );
}
