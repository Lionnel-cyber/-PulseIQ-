"""PulseIQ data contracts — single source of truth for every data shape.

All structured data that crosses a module boundary must use a type defined here.
Never define schemas inline in routes, connectors, or model code.

Usage::

    from src.contracts import FeatureVector, Prediction, ScoreResponse

Pydantic v2 throughout. All models are immutable by default (``model_config``
inherits Pydantic's default frozen=False, but field mutation is discouraged).

Naming note
───────────
``IngestionMetrics`` and ``SourceFreshnessPayload`` also exist as dataclasses in
``src/observability/metrics.py`` (historical). Those dataclasses have slightly
different field names and will be refactored to import from here in a follow-on
session. Import these Pydantic versions for all new code.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

# ---------------------------------------------------------------------------
# Literal type aliases — defined once, reused across all models
# ---------------------------------------------------------------------------

GeoLevel = Literal["national", "state", "metro", "county", "city", "zip"]
"""Granularity level of a geography row."""

ConfidenceLevel = Literal["high", "medium", "low"]
"""Confidence in a score, derived from Tier 1 signal coverage and calibration
status. See CLAUDE.md confidence rules."""

FreshnessStatus = Literal["ok", "stale", "critical", "unknown"]
"""Freshness of a data source relative to its expected cadence.

ok       — within 1.5× expected cadence
stale    — between 1.5× and 3× expected cadence
critical — more than 3× expected cadence
unknown  — no successful run on record
"""

ScoreBand = Literal["low", "elevated", "high", "critical"]
"""Discrete ESS band derived from score thresholds.

low       < 60
elevated  60–74
high      75–84
critical  ≥ 85
"""

AlertType = Literal[
    "threshold_breach",
    "rapid_rise",
    "sustained_high",
    "ingestion_failure",
    "record_count_drop",
    "latency_spike",
    "source_stale",
]
"""Machine-readable alert categories. Matches MTTD rules in CLAUDE.md."""

# ---------------------------------------------------------------------------
# Feature field names — used by FeatureVector validator
# ---------------------------------------------------------------------------

_FEATURE_FIELDS: frozenset[str] = frozenset({
    "bls_jobless_claims_delta",
    "bls_unemployment_rate",
    "fred_delinquency_rate",
    "census_poverty_baseline",
    "census_median_income",
    "fred_cpi_delta",
    "fred_mortgage_rate_delta",
    "trends_search_score",
    "trends_search_delta",
    "news_sentiment_score",
    "news_article_count",
    "openweather_severity_index",
    "data_quality_score",
    "stale_source_count",
})


# ---------------------------------------------------------------------------
# 1. FeatureVector
# ---------------------------------------------------------------------------


class FeatureVector(BaseModel):
    """Validated input to the XGBoost scoring model.

    All 14 feature fields are required and must not be None — null-fill must
    happen at the mart layer (COALESCE in dbt) before constructing this object.

    Feature names use the prefixed convention that maps directly to CLAUDE.md's
    tier structure. ``features.py`` maps mart column names to these field names.

    Args:
        geo_id: Geography identifier (e.g. ``"Detroit-MI"``).
        geo_level: Granularity of the geography row.
        run_date: Date the dbt run that produced this row executed.

        bls_jobless_claims_delta: Tier 1 — month-over-month change in national
            unemployment rate (BLS LNS14000000 proxy).
        bls_unemployment_rate: Tier 1 — national unemployment rate value.
        fred_delinquency_rate: Tier 1 — credit card delinquency rate (FRED
            DRCCLACBS).
        census_poverty_baseline: Tier 1 — ZIP-level poverty rate from Census ACS.
        census_median_income: Tier 1 — ZIP-level median household income.

        fred_cpi_delta: Tier 2 — month-over-month CPI change (FRED CPIAUCSL).
        fred_mortgage_rate_delta: Tier 2 — change in 30-year mortgage rate (FRED
            MORTGAGE30US). Zero until the FRED connector adds this series.

        trends_search_score: Tier 3 — Google Trends distress search index (0–100).
            Zero until the Trends connector is added in Phase 2.
        trends_search_delta: Tier 3 — change in trends_search_score vs prior week.
        news_sentiment_score: Tier 3 — news/Reddit negativity fraction (0.0–1.0).
        news_article_count: Tier 3 — count of relevant news articles / Reddit posts.
        openweather_severity_index: Tier 3 — weather stress index (0.0–1.0).

        data_quality_score: 5-domain source coverage fraction (0.0–1.0).
        stale_source_count: Count of sources whose data exceeds the freshness SLA.

    Raises:
        ValidationError: If any feature field is None or out of bounds.
    """

    # Identity
    geo_id: str
    geo_level: GeoLevel
    run_date: date

    # Tier 1 — High confidence (weight 0.55)
    bls_jobless_claims_delta: float
    bls_unemployment_rate: float
    fred_delinquency_rate: float
    census_poverty_baseline: float
    census_median_income: float

    # Tier 2 — Medium confidence (weight 0.30)
    fred_cpi_delta: float
    fred_mortgage_rate_delta: float

    # Tier 3 — Low confidence (weight 0.15)
    trends_search_score: float
    trends_search_delta: float
    news_sentiment_score: float
    news_article_count: int
    openweather_severity_index: float

    # Observability
    data_quality_score: float = Field(ge=0.0, le=1.0)
    stale_source_count: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def reject_none_features(cls, data: Any) -> Any:
        """Raise ValueError if any feature field is None.

        Fires before Pydantic's type coercion so the error message names the
        offending field(s) explicitly. Silent null handling is forbidden —
        use COALESCE in dbt before constructing FeatureVector.

        Args:
            data: Raw input dict passed to the model constructor.

        Returns:
            The input dict unchanged if no None feature fields are found.

        Raises:
            ValueError: Naming every feature field that is None.
        """
        if not isinstance(data, dict):
            return data
        none_fields = sorted(
            f for f in _FEATURE_FIELDS if f in data and data[f] is None
        )
        if none_fields:
            raise ValueError(
                f"FeatureVector received None for feature field(s): {none_fields}. "
                "Null-fill must happen at the mart layer via COALESCE before "
                "constructing FeatureVector — never pass None features to the model."
            )
        return data


# ---------------------------------------------------------------------------
# 2. Prediction
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    """Full scoring output written by ``predict.py`` to the ``ess_scores`` table.

    Args:
        geo_id: Geography identifier.
        geo_name: Human-readable geography name.
        geo_level: Granularity of the scored geography.
        run_date: Date this prediction was produced.
        ess_score: Economic Stress Score (0–100). Higher = more stressed.
        score_band: Discrete band derived from ess_score thresholds.
        delta_7d: ESS change over the prior 7 days. None on first run.
        delta_30d: ESS change over the prior 30 days. None on first run.
        confidence: Confidence level derived from Tier 1 coverage and
            calibration status. See CLAUDE.md confidence rules.
        early_warning: True when Tier 3 signals spike while Tier 1 is calm.
        missing_sources: Source names with no data for this row.
        stale_sources: Source names whose data exceeds the freshness SLA.
        anomaly_flags: Anomaly codes raised by ``validation/rules.py``.
        granularity_warning: True when national-level proxies are used for a
            sub-national geography.
        model_version: MLflow run ID or semver tag of the scoring model.
        feature_version: Semver tag matching the FeatureVector field contract.
        calibrated: True when isotonic calibration has been applied.
        tier1_score: Weighted Tier 1 z-score contribution.
        tier2_score: Weighted Tier 2 z-score contribution.
        tier3_score: Weighted Tier 3 z-score contribution.
        shap_values: Per-feature SHAP contributions. Keys match FeatureVector
            field names.
    """

    geo_id: str
    geo_name: str
    geo_level: GeoLevel
    run_date: date
    ess_score: float = Field(ge=0.0, le=100.0)
    score_band: ScoreBand
    delta_7d: float | None
    delta_30d: float | None
    confidence: ConfidenceLevel
    early_warning: bool
    missing_sources: list[str]
    stale_sources: list[str]
    anomaly_flags: list[str]
    granularity_warning: bool
    model_version: str
    feature_version: str
    calibrated: bool
    tier1_score: float
    tier2_score: float
    tier3_score: float
    shap_values: dict[str, float]


# ---------------------------------------------------------------------------
# 3. RetrievedSource + Explanation
# ---------------------------------------------------------------------------


class RetrievedSource(BaseModel):
    """A single news article or document retrieved from ChromaDB for an explanation.

    Args:
        url: Source URL.
        title: Article or document title.
        published_at: Publication timestamp. None if not available.
        relevance_score: Semantic similarity to the query (0.0–1.0).
    """

    url: str
    title: str
    published_at: datetime | None
    relevance_score: float = Field(ge=0.0, le=1.0)


class Explanation(BaseModel):
    """Structured RAG explanation for a score.

    Enforces the rigid four-section template required by CLAUDE.md:
    1. summary — one factual sentence
    2. top_drivers — at most 3 plain-English SHAP contributors
    3. retrieved_sources — evidence with URLs
    4. caveats — source gaps, stale data, weak evidence (NEVER OMITTED)

    The ``caveats`` field must contain at least one item. Pass
    ``["None identified"]`` explicitly when no caveats apply — an empty list
    is rejected.

    Args:
        geo_id: Geography identifier.
        geo_name: Human-readable geography name.
        run_date: Date the underlying score was produced.
        summary: One factual sentence (e.g. "Score rose 8 points over 7 days").
        top_drivers: Up to 3 top SHAP contributors in plain English.
        shap_breakdown: Full SHAP dict keyed by FeatureVector field name.
        retrieved_sources: News articles supporting the explanation.
        evidence_strength: Aggregate quality of retrieved sources.
        confidence: Confidence level from the underlying Prediction.
        missing_sources: Source names with no data for this row.
        caveats: Source gaps, staleness notes, or weak evidence statements.
            Must contain at least one item.
        generated_at: Timestamp when this explanation was generated.

    Raises:
        ValidationError: If ``top_drivers`` has more than 3 items, or
            ``caveats`` is empty.
    """

    geo_id: str
    geo_name: str
    run_date: date
    summary: str
    top_drivers: Annotated[list[str], Field(max_length=3)]
    shap_breakdown: dict[str, float]
    retrieved_sources: list[RetrievedSource]
    evidence_strength: Literal["strong", "moderate", "weak"]
    confidence: ConfidenceLevel
    missing_sources: list[str]
    caveats: list[str]
    generated_at: datetime

    @model_validator(mode="after")
    def caveats_must_not_be_empty(self) -> "Explanation":
        """Raise if caveats is empty — the caveats section is never omitted.

        Pass ``["None identified"]`` explicitly when no caveats apply.

        Returns:
            Self, unchanged.

        Raises:
            ValueError: If caveats is an empty list.
        """
        if len(self.caveats) == 0:
            raise ValueError(
                "Explanation.caveats must contain at least one item. "
                'Pass ["None identified"] when no caveats apply — '
                "the caveats section is never omitted per CLAUDE.md."
            )
        return self


# ---------------------------------------------------------------------------
# 4. AlertPayload
# ---------------------------------------------------------------------------


class AlertPayload(BaseModel):
    """Interpretable alert payload delivered to webhook subscribers.

    Every field must be human-readable at delivery time — the recipient must be
    able to act on this payload without making additional API calls.

    Args:
        alert_id: UUID4 uniquely identifying this alert instance.
        region_id: Geography identifier (matches geo_id in Prediction).
        region_name: Human-readable geography name.
        triggered_at: Timestamp when the alert condition was detected.
        current_score: ESS at the time the alert fired.
        previous_score: ESS at the start of the delta window.
        score_delta: current_score − previous_score.
        delta_window_days: Number of days the delta spans.
        alert_type: Machine-readable alert category.
        top_drivers: Up to 3 plain-English SHAP contributors at alert time.
        explanation_summary: One-sentence summary of why the alert fired.
        confidence: Confidence level of the underlying score.
        missing_sources: Source names with no data at alert time.
        model_version: MLflow run ID or semver tag of the model that fired.
        explanation_url: URL to the full explanation in the API.
        suppressed_until: If set, re-alerts are suppressed until this time.
    """

    alert_id: str
    region_id: str
    region_name: str
    triggered_at: datetime
    current_score: float
    previous_score: float
    score_delta: float
    delta_window_days: int
    alert_type: AlertType
    top_drivers: Annotated[list[str], Field(max_length=3)]
    explanation_summary: str
    confidence: ConfidenceLevel
    missing_sources: list[str]
    model_version: str
    explanation_url: str
    suppressed_until: datetime | None


# ---------------------------------------------------------------------------
# 5. IngestionMetrics (Pydantic)
# ---------------------------------------------------------------------------


class IngestionMetrics(BaseModel):
    """Per-run observability record for a connector ingestion.

    Write one of these after every connector run — especially on failure.
    The ``latency_seconds`` field is computed from ``started_at`` and
    ``completed_at`` by the connector, not re-derived here.

    Note: A dataclass with the same name exists in ``src/observability/metrics.py``
    for historical reasons. That dataclass will be refactored to use this model
    in a follow-on session. Use this Pydantic model for all new code.

    Args:
        source: Connector identifier (e.g. ``"bls"``, ``"fred"``, ``"news"``).
        run_date: Calendar date of the ingestion run.
        run_id: UUID4 string uniquely identifying this run.
        started_at: Timestamp when the connector began execution.
        completed_at: Timestamp when the connector finished (success or failure).
        latency_seconds: Wall-clock seconds from first HTTP call to
            ``save_raw()`` completion. Must be ≥ 0.
        records_fetched: Total records returned by the source API.
        records_rejected: Records that failed hard validation.
        records_suspect: Records written with ``anomaly_flag=True``.
        freshness_status: Data freshness relative to the source cadence SLA.
        http_retries: Number of HTTP retries made by the tenacity retry wrapper.
        success: True if the run completed without an unhandled exception.
        error_message: Exception message if success is False, otherwise None.
    """

    source: str
    run_date: date
    run_id: str
    started_at: datetime
    completed_at: datetime
    latency_seconds: float = Field(ge=0.0)
    records_fetched: int = Field(ge=0)
    records_rejected: int = Field(ge=0)
    records_suspect: int = Field(ge=0)
    freshness_status: FreshnessStatus
    http_retries: int = Field(ge=0, default=0)
    success: bool
    error_message: str | None = None


# ---------------------------------------------------------------------------
# 6. SourceFreshnessPayload (Pydantic)
# ---------------------------------------------------------------------------


class SourceFreshnessPayload(BaseModel):
    """Freshness summary for a single data source.

    Returned by the ``/health/freshness`` endpoint and consumed by
    ``HealthResponse``.

    Note: A dataclass with the same name exists in ``src/observability/metrics.py``
    with slightly different field names. It will be refactored in a follow-on
    session. Use this Pydantic model for all new code.

    Args:
        source: Source identifier (e.g. ``"bls"``).
        last_successful_fetch: Date of the most recent successful run.
            None if no successful run exists.
        days_since_fetch: Calendar days since ``last_successful_fetch``.
            None if no successful run exists.
        expected_cadence_days: Expected fetch interval in days.
        freshness_status: Current freshness relative to the cadence SLA.
        records_last_run: Record count from the most recent successful run.
        anomaly_flag: True if the most recent run raised a validation anomaly.
    """

    source: str
    last_successful_fetch: date | None
    days_since_fetch: int | None
    expected_cadence_days: int = Field(ge=0)
    freshness_status: FreshnessStatus
    records_last_run: int = Field(ge=0, default=0)
    anomaly_flag: bool = False


# ---------------------------------------------------------------------------
# 7. ScoreResponse — API response for GET /scores/{geo_id}
# ---------------------------------------------------------------------------


class ScoreResponse(Prediction):
    """API response for the ``GET /scores/{geo_id}`` endpoint.

    Inherits all fields from ``Prediction`` and adds ``is_trustworthy``, a
    computed field that summarises whether the score meets the bar for
    high-confidence production use.

    ``is_trustworthy`` is serialised into the JSON response body automatically
    via Pydantic's ``@computed_field`` mechanism.
    """

    @computed_field  # type: ignore[misc]
    @property
    def is_trustworthy(self) -> bool:
        """True when the score meets all high-confidence criteria.

        A score is trustworthy when ALL of the following hold:
        - ``confidence == "high"`` (Tier 1 coverage ≥ 0.8)
        - ``calibrated`` is True (isotonic calibration has been applied)
        - ``granularity_warning`` is False (no national proxy inflation)
        - ``anomaly_flags`` is empty (no validation anomalies)

        Returns:
            True if all four criteria are met, False otherwise.
        """
        return (
            self.confidence == "high"
            and self.calibrated
            and not self.granularity_warning
            and len(self.anomaly_flags) == 0
        )


# ---------------------------------------------------------------------------
# 8. TimeSeriesPoint
# ---------------------------------------------------------------------------


class TimeSeriesPoint(BaseModel):
    """A single point in a score history series.

    Args:
        date: Signal date this point represents.
        ess_score: ESS value on this date (0–100).
        confidence: Confidence level for this point.
        missing_sources: Source names with no data on this date.
        anomaly_flag: True if any anomaly was raised for this point.
    """

    date: date
    ess_score: float = Field(ge=0.0, le=100.0)
    confidence: ConfidenceLevel
    missing_sources: list[str]
    anomaly_flag: bool


# ---------------------------------------------------------------------------
# 9. TimeSeriesResponse — API response for history endpoints
# ---------------------------------------------------------------------------


class TimeSeriesResponse(BaseModel):
    """API response for score history endpoints.

    Args:
        geo_id: Geography identifier.
        geo_name: Human-readable geography name.
        period_days: Number of days spanned by ``points``.
        points: Score history, one entry per day, ordered by date ascending.
        trend: Direction of the score series over the period.

            improving     — score declining (lower stress)
            stable        — score within noise threshold
            deteriorating — score rising (higher stress)
            volatile      — score oscillating without clear direction
    """

    geo_id: str
    geo_name: str
    period_days: int
    points: list[TimeSeriesPoint]
    trend: Literal["improving", "stable", "deteriorating", "volatile"]


# ---------------------------------------------------------------------------
# 10. HealthResponse — GET /health/freshness
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for the ``GET /health/freshness`` endpoint.

    Args:
        status: Aggregate pipeline health.

            ok       — all sources within SLA
            degraded — one or more sources stale
            down     — critical source missing or pipeline halted

        checked_at: Timestamp when this health snapshot was computed.
        source_freshness: Per-source freshness status for all known sources.
        stale_sources: Source names currently outside their freshness SLA.
        overall_data_quality: Mean ``data_quality_score`` across the most
            recent mart run (0.0–1.0).
    """

    status: Literal["ok", "degraded", "down"]
    checked_at: datetime
    source_freshness: list[SourceFreshnessPayload]
    stale_sources: list[str]
    overall_data_quality: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# 11. ModelVersionResponse — GET /health/model
# ---------------------------------------------------------------------------


class ModelVersionResponse(BaseModel):
    """Response for the ``GET /health/model`` endpoint.

    Args:
        model_version: MLflow run ID or semver tag of the deployed model.
        feature_version: Semver tag matching the FeatureVector field contract.
        trained_at: Timestamp when the model was last trained. None if unknown.
        calibrated: True when isotonic calibration has been applied.
        calibration_samples: Number of ground-truth samples used for
            calibration. Must be ≥ 60 before confidence can be "high".
        mlflow_run_id: MLflow run ID. None if tracking URI is not configured.
    """

    model_version: str
    feature_version: str
    trained_at: datetime | None
    calibrated: bool
    calibration_samples: int = Field(ge=0)
    mlflow_run_id: str | None


# ---------------------------------------------------------------------------
# 12. PipelineStatusResponse — GET /health/pipeline
# ---------------------------------------------------------------------------


class PipelineStatusResponse(BaseModel):
    """Response for the ``GET /health/pipeline`` endpoint.

    Args:
        status: Aggregate pipeline status.

            ok       — all three DAGs completed within expected windows
            degraded — one or more DAGs delayed or partially failed
            down     — scoring pipeline has not run in >24 hours

        last_ingest_run: Timestamp of the most recent dag_ingest_daily completion.
        last_transform_run: Timestamp of the most recent dag_transform_daily
            completion.
        last_score_run: Timestamp of the most recent dag_score_and_alert
            completion.
        checked_at: Timestamp when this status was computed.
        failures: Human-readable descriptions of any active failures.
    """

    status: Literal["ok", "degraded", "down"]
    last_ingest_run: datetime | None
    last_transform_run: datetime | None
    last_score_run: datetime | None
    checked_at: datetime
    failures: list[str]


# ---------------------------------------------------------------------------
# 13. BenchmarkSummaryResponse
# ---------------------------------------------------------------------------


class BenchmarkSummaryResponse(BaseModel):
    """Compact benchmark summary for dashboard and health views.

    Args:
        model_rmse: RMSE of the model forecast. None when unavailable.
        baseline_rmse: RMSE of the naive baseline. None when unavailable.
        improvement_pct: Relative improvement vs baseline. None when unavailable.
        verdict: Human-readable shipping verdict.
        warning: Explanatory warning when benchmark data is insufficient.
    """

    model_rmse: float | None
    baseline_rmse: float | None
    improvement_pct: float | None
    verdict: str
    warning: str | None = None


# ---------------------------------------------------------------------------
# 14. SourceHealthRow
# ---------------------------------------------------------------------------


class SourceHealthRow(BaseModel):
    """Dashboard row summarising one source's operational health.

    Args:
        source: Source identifier.
        last_run: Timestamp of the most recent ingestion run, successful or not.
        status: Dashboard-friendly source status.
        records: Record count from the most recent run.
        latency_seconds: Latest observed latency in seconds.
        trend_7d: Seven-day records trend classification.
    """

    source: str
    last_run: datetime | None
    status: Literal["ok", "slow", "down"]
    records: int = Field(ge=0, default=0)
    latency_seconds: float | None = Field(default=None, ge=0.0)
    trend_7d: Literal["improving", "stable", "deteriorating", "unknown"]


# ---------------------------------------------------------------------------
# 15. ModelHealthSummary
# ---------------------------------------------------------------------------


class ModelHealthSummary(BaseModel):
    """Dashboard summary of the deployed model.

    Args:
        version: Deployed model version identifier.
        trained_at: Timestamp when the model artifacts were last produced.
        calibrated: True when isotonic calibration is active.
        benchmark: Latest benchmark summary.
    """

    version: str
    trained_at: datetime | None
    calibrated: bool
    benchmark: BenchmarkSummaryResponse


# ---------------------------------------------------------------------------
# 16. PipelineHealthSummary
# ---------------------------------------------------------------------------


class PipelineHealthSummary(BaseModel):
    """Dashboard summary of end-to-end pipeline health.

    Args:
        status: Aggregate pipeline status.
        last_ingest_run: Most recent ingestion DAG completion.
        last_transform_run: Most recent transform DAG completion.
        last_score_run: Most recent scoring DAG completion.
        anomaly_flags_count: Count of latest-run score rows with anomaly flags.
        failures: Active pipeline failure descriptions.
    """

    status: Literal["ok", "degraded", "down"]
    last_ingest_run: datetime | None
    last_transform_run: datetime | None
    last_score_run: datetime | None
    anomaly_flags_count: int = Field(ge=0, default=0)
    failures: list[str]


# ---------------------------------------------------------------------------
# 17. HealthDashboardResponse
# ---------------------------------------------------------------------------


class HealthDashboardResponse(BaseModel):
    """Combined operational health payload for the Streamlit dashboard.

    Args:
        checked_at: Timestamp when the payload was assembled.
        source_health: Per-source operational rows.
        model_info: Deployed model metadata and benchmark.
        pipeline_info: DAG-level pipeline status and anomaly counts.
    """

    checked_at: datetime
    source_health: list[SourceHealthRow]
    model_info: ModelHealthSummary
    pipeline_info: PipelineHealthSummary


# ---------------------------------------------------------------------------
# 18. MapScoreResponse
# ---------------------------------------------------------------------------


class MapScoreResponse(BaseModel):
    """State-level score payload used by the dashboard choropleth.

    Args:
        state_code: Two-letter USPS state code used by Plotly.
        geo_id: Underlying source geography identifier.
        geo_name: Human-readable geography name.
        geo_level: Original geography level represented by this row.
        run_date: Snapshot date used for the map.
        ess_score: Economic Stress Score for the mapped geography.
        delta_7d: Seven-day ESS delta, when available.
        confidence: Confidence level mapped to opacity in the UI.
        missing_sources: Sources missing from the underlying geography.
        granularity_warning: True when sub-state precision is inflated.
        drilldown_geo_id: Geography identifier to use for trend/explanation
            drill-down interactions.
    """

    state_code: str = Field(min_length=2, max_length=2)
    geo_id: str
    geo_name: str
    geo_level: GeoLevel
    run_date: date
    ess_score: float = Field(ge=0.0, le=100.0)
    delta_7d: float | None
    confidence: ConfidenceLevel
    missing_sources: list[str]
    granularity_warning: bool
    drilldown_geo_id: str
