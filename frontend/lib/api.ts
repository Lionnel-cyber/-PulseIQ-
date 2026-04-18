/**
 * PulseIQ typed API client.
 *
 * All functions read NEXT_PUBLIC_API_URL from process.env and throw a
 * descriptive Error on any non-2xx response. No `any` — strict TypeScript.
 *
 * Usage:
 *   import { getTopScores, getScore } from "@/lib/api";
 */

import type {
  AlertPayload,
  Explanation,
  HealthDashboardResponse,
  HealthResponse,
  MapScoreResponse,
  ModelVersionResponse,
  PipelineStatusResponse,
  ScoreResponse,
  SourceFreshnessPayload,
  TimeSeriesResponse,
} from "./types";

const CONFIGURED_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

const BASE_URL =
  typeof window !== "undefined" && CONFIGURED_BASE_URL?.startsWith("http")
    ? "/api/pulseiq"
    : CONFIGURED_BASE_URL ?? "http://localhost:8000/api/v1";

/**
 * Internal fetch wrapper. Throws a descriptive error on non-2xx responses.
 *
 * @param path - Path relative to BASE_URL, e.g. "/scores/top?limit=5"
 * @returns Parsed JSON as type T
 * @throws Error with status code and API detail message
 */
async function apiFetch<T>(path: string): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`PulseIQ API ${res.status} on ${path}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

/**
 * Return the top N highest-stress geographies for the most recent run date.
 *
 * Calls GET /scores/top?limit={limit}
 *
 * @param limit - Maximum results to return (1–100, default 20)
 * @returns Array of ScoreResponse ordered by ess_score descending
 */
export async function getTopScores(limit = 20): Promise<ScoreResponse[]> {
  return apiFetch<ScoreResponse[]>(`/scores/top?limit=${limit}`);
}

/**
 * Return the latest score snapshot for the most recent run date.
 *
 * Calls GET /scores/snapshot?limit={limit}
 *
 * @param limit - Maximum rows to return (default 500)
 * @returns Array of ScoreResponse rows for the resolved snapshot date
 */
export async function getScoreSnapshot(
  limit = 500
): Promise<ScoreResponse[]> {
  return apiFetch<ScoreResponse[]>(`/scores/snapshot?limit=${limit}`);
}

/**
 * Return the state-level map payload used by the monitor view.
 *
 * Calls GET /scores/map
 *
 * @returns Array of MapScoreResponse rows for the latest snapshot date
 */
export async function getMapScores(): Promise<MapScoreResponse[]> {
  return apiFetch<MapScoreResponse[]>("/scores/map");
}

/**
 * Return the latest score for a single geography.
 *
 * Calls GET /scores/{geoId}
 *
 * @param geoId - Geography identifier, e.g. "MSA-16980" or "IL"
 * @returns ScoreResponse for the requested geography
 * @throws Error with status 404 if the geography is not in the database
 */
export async function getScore(geoId: string): Promise<ScoreResponse> {
  return apiFetch<ScoreResponse>(`/scores/${encodeURIComponent(geoId)}`);
}

/**
 * Return score history for a single geography over the requested window.
 *
 * Calls GET /scores/{geoId}/history?window={window}
 *
 * @param geoId  - Geography identifier
 * @param window - Look-back window: "7d" | "30d" | "90d" (default "30d")
 * @returns TimeSeriesResponse with per-day points and overall trend
 * @throws Error with status 404 if the geography is not in the database
 */
export async function getHistory(
  geoId: string,
  window: "7d" | "30d" | "90d" = "30d"
): Promise<TimeSeriesResponse> {
  return apiFetch<TimeSeriesResponse>(
    `/scores/${encodeURIComponent(geoId)}/history?window=${window}`
  );
}

/**
 * Return the structured RAG explanation for a geography's latest score.
 *
 * Calls GET /explain/{geoId}
 *
 * @param geoId - Geography identifier
 * @returns Four-section Explanation: summary, top_drivers, evidence, caveats
 * @throws Error with status 404 if the geography is not in the database
 */
export async function getExplanation(geoId: string): Promise<Explanation> {
  return apiFetch<Explanation>(`/explain/${encodeURIComponent(geoId)}`);
}

/**
 * Return the freshness status of every PulseIQ data source.
 *
 * Calls GET /health/freshness
 *
 * @returns Array of SourceFreshnessPayload, one per known source
 */
export async function getSourceFreshness(): Promise<SourceFreshnessPayload[]> {
  return apiFetch<SourceFreshnessPayload[]>("/health/freshness");
}

/**
 * Return the aggregate pipeline health (shortcut to /health/freshness).
 *
 * Calls GET /health
 *
 * @returns HealthResponse with per-source freshness and overall_data_quality
 */
export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

/**
 * Return deployed model metadata for the health view.
 *
 * Calls GET /health/model
 */
export async function getModelVersion(): Promise<ModelVersionResponse> {
  return apiFetch<ModelVersionResponse>("/health/model");
}

/**
 * Return pipeline run status for the health view.
 *
 * Calls GET /health/pipeline
 */
export async function getPipelineStatus(): Promise<PipelineStatusResponse> {
  return apiFetch<PipelineStatusResponse>("/health/pipeline");
}

/**
 * Return the combined health dashboard payload.
 *
 * Calls GET /health/dashboard
 */
export async function getHealthDashboard(): Promise<HealthDashboardResponse> {
  return apiFetch<HealthDashboardResponse>("/health/dashboard");
}

/**
 * Return all recorded alerts across all geographies.
 *
 * Calls GET /alerts/history — returns empty array when the endpoint is
 * unavailable or returns 404.
 */
export async function getAlerts(): Promise<AlertPayload[]> {
  return apiFetch<AlertPayload[]>("/alerts/history");
}

/**
 * Return all alerts for a single geography.
 *
 * Calls GET /alerts/history/{geoId}
 *
 * @param geoId - Geography identifier
 * @returns List of AlertPayload ordered by triggered_at descending
 */
export async function getAlertHistory(geoId: string): Promise<AlertPayload[]> {
  return apiFetch<AlertPayload[]>(`/alerts/history/${encodeURIComponent(geoId)}`);
}
