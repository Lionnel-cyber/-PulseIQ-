/**
 * PulseIQ TypeScript contracts.
 *
 * Every interface mirrors a Pydantic model in src/contracts.py exactly.
 * Python `date` and `datetime` fields are serialised to ISO 8601 strings by
 * FastAPI's JSON encoder; they are typed as `string` here.
 * Python `None` becomes `null` in JSON.
 */

// ---------------------------------------------------------------------------
// Literal type aliases
// ---------------------------------------------------------------------------

/** Granularity level of a geography row. Matches GeoLevel in contracts.py. */
export type GeoLevel =
  | "national"
  | "state"
  | "metro"
  | "county"
  | "city"
  | "zip";

/** Confidence in a score, derived from Tier 1 signal coverage and calibration. */
export type Confidence = "high" | "medium" | "low";

/** Freshness of a data source relative to its expected cadence. */
export type FreshnessStatus = "ok" | "stale" | "critical" | "unknown";

/** Discrete ESS band derived from score thresholds (low < 60, elevated 60–74, high 75–84, critical ≥ 85). */
export type ScoreBand = "low" | "elevated" | "high" | "critical";

/** Machine-readable alert categories. Matches AlertType in contracts.py. */
export type AlertType =
  | "threshold_breach"
  | "rapid_rise"
  | "sustained_high"
  | "ingestion_failure"
  | "record_count_drop"
  | "latency_spike"
  | "source_stale";

/** Direction of a score series over a time window. */
export type Trend = "improving" | "stable" | "deteriorating" | "volatile";

/** Window selector for monitor history charts. */
export type HistoryWindow = "7d" | "30d" | "90d";

/**
 * UI-level threat classification — not a backend contract field.
 * Used to colour-code regions on the map and badge alerts.
 */
export type ThreatLevel = "monitor" | "elevated" | "critical";

/**
 * Shared sidebar filter state for the monitor page.
 * Empty arrays mean "show all" for that filter dimension.
 */
export interface FilterState {
  geoLevels: Array<"metro" | "county" | "zip">;
  minScore: 0 | 60 | 75;
  confidenceLevels: Array<Confidence>;
}

export const DEFAULT_FILTER_STATE: FilterState = {
  geoLevels: [],
  minScore: 0,
  confidenceLevels: [],
};

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

/**
 * API response for GET /scores/{geo_id} and GET /scores/top.
 * Mirrors ScoreResponse (extends Prediction) in contracts.py.
 * `is_trustworthy` is a Pydantic computed_field serialised into the response.
 */
export interface ScoreResponse {
  geo_id: string;
  geo_name: string;
  geo_level: GeoLevel;
  /** ISO 8601 date string, e.g. "2026-04-13" */
  run_date: string;
  ess_score: number;
  score_band: ScoreBand;
  delta_7d: number | null;
  delta_30d: number | null;
  confidence: Confidence;
  early_warning: boolean;
  missing_sources: string[];
  stale_sources: string[];
  anomaly_flags: string[];
  granularity_warning: boolean;
  model_version: string;
  feature_version: string;
  calibrated: boolean;
  tier1_score: number;
  tier2_score: number;
  tier3_score: number;
  /** SHAP contributions keyed by FeatureVector field name. */
  shap_values: Record<string, number>;
  /** True when confidence is high, calibrated, no granularity warning, no anomaly flags. */
  is_trustworthy: boolean;
}

/**
 * Optional frontend extension for score payloads that include map centroids.
 * The current FastAPI ScoreResponse contract does not require these fields.
 */
export type MappableScore = ScoreResponse & {
  longitude?: number | null;
  latitude?: number | null;
  population_at_risk?: number | null;
};

/**
 * A single point in a score history series.
 * Mirrors TimeSeriesPoint in contracts.py.
 */
export interface TimeSeriesPoint {
  /** ISO 8601 date string, e.g. "2026-04-13" */
  date: string;
  ess_score: number;
  confidence: Confidence;
  missing_sources: string[];
  anomaly_flag: boolean;
}

/**
 * API response for GET /scores/{geo_id}/history.
 * Mirrors TimeSeriesResponse in contracts.py.
 */
export interface TimeSeriesResponse {
  geo_id: string;
  geo_name: string;
  period_days: number;
  /** Score history ordered by date ascending. */
  points: TimeSeriesPoint[];
  trend: Trend;
}

/**
 * A single news article or document retrieved from ChromaDB.
 * Mirrors RetrievedSource in contracts.py.
 */
export interface RetrievedSource {
  url: string;
  title: string;
  /** ISO 8601 datetime string, or null when publication date is unavailable. */
  published_at: string | null;
  /** Semantic similarity to the query (0.0–1.0). */
  relevance_score: number;
}

/**
 * Structured RAG explanation for a score.
 * Mirrors Explanation in contracts.py — four-section structure is mandatory.
 * `caveats` always contains at least one item; `top_drivers` is max 3.
 */
export interface Explanation {
  geo_id: string;
  geo_name: string;
  /** ISO 8601 date string. */
  run_date: string;
  /** One factual sentence, e.g. "Score rose 8 points over 7 days." */
  summary: string;
  /** Up to 3 top SHAP contributors in plain English. */
  top_drivers: string[];
  shap_breakdown: Record<string, number>;
  retrieved_sources: RetrievedSource[];
  evidence_strength: "strong" | "moderate" | "weak";
  confidence: Confidence;
  missing_sources: string[];
  /** At least one item; ["None identified"] when no caveats apply. */
  caveats: string[];
  /** ISO 8601 datetime string. */
  generated_at: string;
}

/**
 * Freshness summary for a single data source.
 * Mirrors SourceFreshnessPayload in contracts.py.
 */
export interface SourceFreshnessPayload {
  source: string;
  /** ISO 8601 date string, or null if no successful run exists. */
  last_successful_fetch: string | null;
  days_since_fetch: number | null;
  expected_cadence_days: number;
  freshness_status: FreshnessStatus;
  records_last_run: number;
  anomaly_flag: boolean;
}

/**
 * Response for GET /api/v1/health (aggregate) and derived health endpoints.
 * Mirrors HealthResponse in contracts.py.
 */
export interface HealthResponse {
  status: "ok" | "degraded" | "down";
  /** ISO 8601 datetime string. */
  checked_at: string;
  source_freshness: SourceFreshnessPayload[];
  stale_sources: string[];
  /** Mean data_quality_score across the most recent mart run (0.0–1.0). */
  overall_data_quality: number;
}

/**
 * Response for GET /health/model.
 * Mirrors ModelVersionResponse in contracts.py.
 */
export interface ModelVersionResponse {
  model_version: string;
  feature_version: string;
  trained_at: string | null;
  calibrated: boolean;
  calibration_samples: number;
  mlflow_run_id: string | null;
}

/**
 * Response for GET /health/pipeline.
 * Mirrors PipelineStatusResponse in contracts.py.
 */
export interface PipelineStatusResponse {
  status: "ok" | "degraded" | "down";
  last_ingest_run: string | null;
  last_transform_run: string | null;
  last_score_run: string | null;
  checked_at: string;
  failures: string[];
}

/**
 * Compact benchmark summary from GET /health/dashboard.
 * Mirrors BenchmarkSummaryResponse in contracts.py.
 */
export interface BenchmarkSummaryResponse {
  model_rmse: number | null;
  baseline_rmse: number | null;
  improvement_pct: number | null;
  verdict: string;
  warning: string | null;
}

/**
 * One source row from GET /health/dashboard.
 * Mirrors SourceHealthRow in contracts.py.
 */
export interface SourceHealthRow {
  source: string;
  last_run: string | null;
  status: "ok" | "slow" | "down";
  records: number;
  latency_seconds: number | null;
  trend_7d: "improving" | "stable" | "deteriorating" | "unknown";
}

/**
 * Model block from GET /health/dashboard.
 * Mirrors ModelHealthSummary in contracts.py.
 */
export interface ModelHealthSummary {
  version: string;
  trained_at: string | null;
  calibrated: boolean;
  benchmark: BenchmarkSummaryResponse;
}

/**
 * Pipeline block from GET /health/dashboard.
 * Mirrors PipelineHealthSummary in contracts.py.
 */
export interface PipelineHealthSummary {
  status: "ok" | "degraded" | "down";
  last_ingest_run: string | null;
  last_transform_run: string | null;
  last_score_run: string | null;
  anomaly_flags_count: number;
  failures: string[];
}

/**
 * Combined health payload returned by GET /health/dashboard.
 * Mirrors HealthDashboardResponse in contracts.py.
 */
export interface HealthDashboardResponse {
  checked_at: string;
  source_health: SourceHealthRow[];
  model_info: ModelHealthSummary;
  pipeline_info: PipelineHealthSummary;
}

/**
 * State-level dashboard payload used by GET /scores/map.
 * Mirrors MapScoreResponse in contracts.py.
 */
export interface MapScoreResponse {
  state_code: string;
  geo_id: string;
  geo_name: string;
  geo_level: GeoLevel;
  run_date: string;
  ess_score: number;
  delta_7d: number | null;
  confidence: Confidence;
  missing_sources: string[];
  granularity_warning: boolean;
  drilldown_geo_id: string;
}

/**
 * Interpretable alert payload delivered to webhook subscribers.
 * Mirrors AlertPayload in contracts.py.
 */
export interface AlertPayload {
  alert_id: string;
  region_id: string;
  region_name: string;
  /** ISO 8601 datetime string. */
  triggered_at: string;
  current_score: number;
  previous_score: number;
  score_delta: number;
  delta_window_days: number;
  alert_type: AlertType;
  /** Up to 3 plain-English SHAP contributors at alert time. */
  top_drivers: string[];
  explanation_summary: string;
  confidence: Confidence;
  missing_sources: string[];
  model_version: string;
  explanation_url: string;
  /** ISO 8601 datetime string, or null if not suppressed. */
  suppressed_until: string | null;
}
