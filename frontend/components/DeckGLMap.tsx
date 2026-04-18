import { useEffect, useRef, useState } from "react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { MapView, _GlobeView as GlobeView } from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import { Map } from "react-map-gl/maplibre";
import type { Confidence, ScoreBand } from "@/lib/types";

export interface MapPoint {
  geo_id: string;
  geo_name: string;
  ess_score: number;
  score_band: ScoreBand;
  confidence: Confidence;
  missing_sources: string[];
  run_date: string;
  longitude: number;
  latitude: number;
  population_at_risk?: number | null;
}

interface DeckGLMapProps {
  points: MapPoint[];
  viewMode: "flat" | "globe";
  selectedGeoId?: string | null;
  onGeoSelect: (geoId: string) => void;
}

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
}

function getColor(band: ScoreBand): [number, number, number, number] {
  switch (band) {
    case "critical":
      return [229, 62, 62, 230];
    case "high":
      return [230, 100, 35, 215];
    case "elevated":
      return [245, 166, 35, 215];
    default:
      return [0, 136, 255, 180];
  }
}

function getRadius(population: number | null | undefined): number {
  if (!population) {
    return 12_000;
  }

  const clamped = Math.max(1_000, Math.min(population, 5_000_000));
  return 4_000 + (clamped / 5_000_000) * 76_000;
}

function getOpacity(confidence: Confidence): number {
  return confidence === "high" ? 0.9 : confidence === "medium" ? 0.65 : 0.35;
}

const FLAT_INIT: ViewState = {
  longitude: -98.5795,
  latitude: 39.8283,
  zoom: 3.5,
};

const GLOBE_INIT: ViewState = {
  longitude: -98,
  latitude: 35,
  zoom: 1.5,
};

function getTooltip(info: PickingInfo<MapPoint>) {
  const item = info.object ?? null;
  if (!item) {
    return null;
  }

  const missing = item.missing_sources.length
    ? `<br/>Missing: ${item.missing_sources.join(", ")}`
    : "";

  return {
    html: `<div style="font-family:monospace;font-size:10px;line-height:1.6">
      <b>${item.geo_name}</b><br/>
      ESS: ${item.ess_score.toFixed(1)} | ${item.score_band.toUpperCase()}<br/>
      Confidence: ${item.confidence}${missing}
    </div>`,
    style: {
      background: "#0f1318",
      border: "1px solid #1e2730",
      padding: "6px 10px",
      color: "#c8d8e8",
    },
  };
}

export default function DeckGLMap({
  points,
  viewMode,
  selectedGeoId,
  onGeoSelect,
}: DeckGLMapProps) {
  const [viewState, setViewState] = useState<ViewState>(
    viewMode === "globe" ? GLOBE_INIT : FLAT_INIT
  );

  const lastInteractionRef = useRef<number>(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (viewMode !== "globe") {
      cancelAnimationFrame(rafRef.current);
      return;
    }

    function tick() {
      const idle = Date.now() - lastInteractionRef.current > 3_000;
      if (idle) {
        setViewState((current) => ({
          ...current,
          longitude: current.longitude + 0.03,
        }));
      }
      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [viewMode]);

  const latestRunDate = points[0]?.run_date ?? "--";
  const selectedPoints = points.filter((point) => point.geo_id === selectedGeoId);

  const layers = [
    new ScatterplotLayer<MapPoint>({
      id: "stress-dots",
      data: points,
      getPosition: (point) => [point.longitude, point.latitude],
      getFillColor: (point) => {
        const [red, green, blue, alpha] = getColor(point.score_band);
        return [red, green, blue, Math.round(alpha * getOpacity(point.confidence))];
      },
      getRadius: (point) => getRadius(point.population_at_risk),
      opacity: 1,
      pickable: true,
      radiusMinPixels: 4,
      radiusMaxPixels: 20,
      updateTriggers: {
        getFillColor: points,
        getRadius: points,
      },
    }),
    new ScatterplotLayer<MapPoint>({
      id: "critical-rings",
      data: points.filter((point) => point.score_band === "critical"),
      getPosition: (point) => [point.longitude, point.latitude],
      getFillColor: [0, 0, 0, 0] as [number, number, number, number],
      getLineColor: (point) => {
        const [red, green, blue] = getColor(point.score_band);
        return [red, green, blue, 180] as [number, number, number, number];
      },
      stroked: true,
      filled: false,
      getLineWidth: 1,
      getRadius: (point) => getRadius(point.population_at_risk) * 1.8,
      radiusMinPixels: 6,
      radiusMaxPixels: 36,
      updateTriggers: {
        getLineColor: points,
        getRadius: points,
      },
    }),
    new ScatterplotLayer<MapPoint>({
      id: "selected-ring",
      data: selectedPoints,
      getPosition: (point) => [point.longitude, point.latitude],
      getFillColor: [0, 0, 0, 0] as [number, number, number, number],
      getLineColor: [0, 212, 170, 255] as [number, number, number, number],
      stroked: true,
      filled: false,
      getLineWidth: 2,
      getRadius: (point) => getRadius(point.population_at_risk) * 2.2,
      radiusMinPixels: 10,
      radiusMaxPixels: 44,
    }),
  ];

  const view =
    viewMode === "globe" ? new GlobeView() : new MapView({ repeat: true });

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div className="scan-line" />

      <DeckGL
        views={view}
        viewState={viewState}
        onViewStateChange={({ viewState: nextViewState }) =>
          setViewState(nextViewState as ViewState)
        }
        onDrag={() => {
          lastInteractionRef.current = Date.now();
        }}
        onInteractionStateChange={() => {
          lastInteractionRef.current = Date.now();
        }}
        layers={layers}
        getTooltip={getTooltip}
        onClick={({ object }) => {
          if (object) {
            onGeoSelect((object as MapPoint).geo_id);
          }
        }}
        style={{ width: "100%", height: "100%" }}
      >
        {viewMode === "flat" && (
          <Map mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" />
        )}
      </DeckGL>

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
        Opacity = confidence | Ring = selected | Data as of {latestRunDate}
        {" | "}
        {viewMode === "globe" ? "3D GLOBE" : "FLAT MAP"}
      </div>
    </div>
  );
}
