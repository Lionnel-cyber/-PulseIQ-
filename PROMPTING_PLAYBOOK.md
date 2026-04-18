# PulseIQ — Claude Code Prompting Playbook
## Definitive edition — all phases, all modules

---

## Golden Rules

1. Always `cd pulseiq` before running `claude`
2. Start every session with the SESSION START prompt
3. Use `/plan` before any task touching more than one file
4. One module at a time — never ask for multiple modules at once
5. Always ask for tests immediately after each module
6. Review every diff before approving — never skip this step
7. Use `/compact` if responses feel vague in long sessions
8. End every session with the SESSION END prompt

---

## Session Start — use every single time

```
Read CLAUDE.md fully.

Then tell me:
1. Project purpose in one sentence
2. Which Build Status items are complete
3. The known issues listed
4. What the logical next task is

Do not write any code yet.
```

---

## Session End — run before closing

```
Update the Build Status checklist in CLAUDE.md.
Mark completed items with [x].
Add a one-line note under each completed item
describing any decisions or deviations from plan.
List any new known issues discovered.
Do not change anything else in CLAUDE.md.
```

---

# PHASE 1B — Reliability Layer
(Complete before Phase 2 — fixes foundation)

---

### STEP 1B-1 — NewsAPI connector (replaces Reddit)

```
/plan

Read src/connectors/base_connector.py and
src/connectors/fred_connector.py as the pattern.

Create src/connectors/news_connector.py.

Requirements:
- Fetch from NewsAPI (read NEWS_API_KEY from env)
- Query terms (fetch all daily):
  "unemployment layoffs", "food bank demand",
  "eviction crisis", "economic hardship",
  "job losses", "foreclosure"
- Pydantic model: NewsArticle with fields:
  title, description, url, publishedAt, source_name
- Pydantic model: NewsResponse with fields:
  articles: list[NewsArticle]
- Method: fetch(query_terms: list[str]) → pd.DataFrame
  Columns: date (datetime), geo_id (str), geo_level ("city"),
  geo_name (str), headline (str), description (str),
  sentiment_score (float, placeholder 0.0 for now),
  source_url (str)
  Extract city from article text using simple regex
  (look for "in {City}, {State}" pattern)
  If no city found: geo_id="US", geo_level="national",
  geo_name="United States"
- Call save_raw() with source_name="news"
- Write IngestionMetrics after run
- Full type hints and docstrings

Then write tests/test_news_connector.py:
- Mock all HTTP calls with responses library
- Test: correct DataFrame schema
- Test: city extraction from article text
- Test: fallback to national when no city found
- Test: retries on HTTP 429
```

---

### STEP 1B-2 — Validation rules

```
/plan

Create src/validation/rules.py.

Read src/contracts.py first (or create placeholder if not yet built).

Define VALIDATION_RULES dict with three tiers per value:
expected_range, hard_limits, spike_threshold_pct.

Sources and rules:

bls / jobless_claims:
  expected_range: (1_000, 1_000_000)
  hard_limits: (0, 10_000_000)
  spike_threshold_pct: 300
  drop_to_zero: "reject"  ← claims never hit 0

fred / cpi:
  expected_range: (200, 400)
  hard_limits: (0, 1_000)
  max_monthly_delta_pct: 5

fred / delinquency_rate:
  expected_range: (0.5, 15.0)
  hard_limits: (0.0, 100.0)
  spike_threshold_pct: 200

news / sentiment_score:
  expected_range: (-1.0, 1.0)
  hard_limits: (-1.0, 1.0)
  spike_threshold_pct: 1_000  ← 10x = possible bot wave

trends / search_score:
  expected_range: (0, 100)
  hard_limits: (0, 100)
  spike_threshold_pct: 500

Create ValidationResult enum:
  VALID    → use normally
  SUSPECT  → use with anomaly_flag=True
  REJECTED → do not write to mart

Create function: validate(source, field, value, previous_value=None)
  → ValidationResult

Write tests/test_validation_rules.py:
- Test: BLS claims = 0 → REJECTED
- Test: news spike 10x → SUSPECT
- Test: normal CPI value → VALID
- Test: value outside hard_limits → REJECTED
```

---

### STEP 1B-3 — Observability: IngestionMetrics

```
/plan

Create src/observability/metrics.py.

Read src/contracts.py for IngestionMetrics schema.
Read src/connectors/base_connector.py to understand
where this gets called from.

Requirements:
- Class: MetricsWriter
- Method: write_ingestion_metrics(metrics: IngestionMetrics) → None
  Writes to DuckDB table "ingestion_metrics"
  Table auto-created if not exists
  Idempotent: unique_key = (source, run_date, run_id)
- Method: get_source_health(source: str, days: int = 7)
  → SourceFreshnessPayload
  Returns freshness status based on last successful fetch
  and expected cadence from VALIDATION_RULES
- Method: get_all_source_health() → list[SourceFreshnessPayload]
  Returns health for all known sources

MTTD alerting rules — fire POST to ALERT_WEBHOOK_URL if:
- success == False → critical
- records_fetched < (7day_avg * 0.5) → warning
- latency_seconds > (7day_avg_latency * 3) → warning
- freshness_status == "critical" → critical

Write tests/test_metrics.py:
- Use in-memory DuckDB
- Test: write succeeds
- Test: idempotent write (same run_id = overwrite not duplicate)
- Test: get_source_health returns correct freshness_status
- Mock webhook POST
```

---

### STEP 1B-4 — Ground truth logging

```
/plan

Create src/observability/ground_truth.py.

Requirements:
- Class: GroundTruthLogger
- Method: log_raw_signal(source, geo_id, run_date,
    raw_value, processed_value, validation_status, anomaly_flag)
  Writes to DuckDB table "raw_signal_log"
  Called after validation, before writing to mart

- Method: log_prediction(geo_id, run_date, ess_score,
    confidence, shap_values, signal_snapshot)
  CRITICAL: signal_snapshot = full feature vector as dict
  Without this we cannot reproduce any prediction
  Writes to DuckDB table "prediction_log"

- Method: log_ground_truth_event(geo_id, event_date,
    event_type, event_source, severity, confirmed_date)
  event_type: "mass_layoff"|"plant_closure"|"bankruptcy_spike"
  event_source: "BLS_WARN_ACT"|"news_confirmed"|"manual"
  Writes to DuckDB table "ground_truth_events"
  This is future training data — treat as gold

All three tables: idempotent writes, never delete rows.

Write tests/test_ground_truth.py:
- Use in-memory DuckDB
- Test: log_prediction stores complete signal_snapshot
- Test: idempotent writes (same prediction = overwrite)
- Test: log_ground_truth_event persists correctly
```

---

### STEP 1B-5 — Rebuild mart_economic_stress

```
/plan

Read all intermediate models first:
src/transforms/models/intermediate/int_geo_weekly_signals.sql
src/transforms/models/intermediate/int_sentiment_scores.sql

Read CLAUDE.md mart requirements section.

Rebuild src/transforms/models/marts/mart_economic_stress.sql
with all required columns:

Geography: geo_id, geo_level, geo_name, run_date
Tier sub-scores:
  tier1_score: weighted avg of Tier 1 feature z-scores * 0.55
  tier2_score: weighted avg of Tier 2 feature z-scores * 0.30
  tier3_score: weighted avg of Tier 3 feature z-scores * 0.15
All 14 features from CLAUDE.md
Reliability columns:
  data_quality_score (pct non-null features)
  granularity_warning (boolean)
  data_granularity_note (string)
  stale_sources (comma-separated)
  anomaly_flags (comma-separated)

dbt config:
  materialized='incremental'
  unique_key=['geo_id', 'run_date']
  on_schema_change='append_new_columns'

Update schema.yml with required tests.
Show me full diff before applying.
After applying: dbt compile — all 8 models must parse.
```

---

# PHASE 2 — AI & ML

---

### STEP 2-1 — contracts.py

```
/plan

Create src/contracts.py.

This is the single source of truth for every data shape.
Read CLAUDE.md contracts section and API standards before writing.

Create these pydantic models in this order:

1. FeatureVector — 14 features + geo_id + run_date
   Validator: no silent nulls (raise on None)
   Literal geo_level type

2. Prediction — full score output
   Fields: geo_id, geo_name, geo_level, run_date,
   ess_score (0-100), score_band, delta_7d, delta_30d,
   confidence, early_warning, missing_sources, stale_sources,
   anomaly_flags, granularity_warning, model_version,
   feature_version, calibrated, tier1_score, tier2_score,
   tier3_score, shap_values

3. Explanation — structured RAG output
   Fields: geo_id, geo_name, run_date, summary,
   top_drivers (list[str] max 3), shap_breakdown,
   retrieved_sources, evidence_strength, confidence,
   missing_sources, caveats, generated_at

4. AlertPayload — interpretable alert format
   Fields: alert_id, region_id, region_name, triggered_at,
   current_score, previous_score, score_delta, delta_window_days,
   alert_type, top_drivers, explanation_summary, confidence,
   missing_sources, model_version, explanation_url, suppressed_until

5. IngestionMetrics — per-run observability
   Fields: source, run_date, run_id, started_at, completed_at,
   latency_seconds, records_fetched, records_rejected,
   records_suspect, freshness_status, http_retries, success,
   error_message

6. SourceFreshnessPayload — per-source health
   Fields: source, last_successful_fetch, days_since_fetch,
   expected_cadence_days, freshness_status, records_last_run,
   anomaly_flag

7. ScoreResponse — API score endpoint response
   All fields from Prediction plus is_trustworthy property

8. TimeSeriesPoint — single point in history
9. TimeSeriesResponse — history endpoint response
   Include trend: improving|stable|deteriorating|volatile

10. HealthResponse — /health endpoint
11. ModelVersionResponse — /health/model endpoint
12. PipelineStatusResponse — /health/pipeline endpoint

Write tests/test_contracts.py:
- Test: FeatureVector raises on None field
- Test: Prediction ess_score must be 0-100
- Test: Explanation top_drivers max 3 items
- Test: ScoreResponse.is_trustworthy logic
```

---

### STEP 2-2 — Feature engineering

```
/plan

Create src/models/features.py.

Read src/contracts.py FeatureVector first.
Read mart_economic_stress.sql for exact column names.
Read CLAUDE.md tier weights section.

Requirements:
- Function: load_features(db_path, run_date=None) → pd.DataFrame
  Reads mart_economic_stress from DuckDB
  Filters: data_quality_score >= 0.7
  Returns 14 feature columns + geo columns + run_date

- Function: engineer_features(df) → pd.DataFrame
  jobless_claims_delta: value minus 4-week rolling mean
  claims_yoy_pct: pct change vs same week last year
  median_income_index: median_income / 56000 (national median)
  All other features pass through unchanged
  No NaN values in output — raise if any remain

- Function: to_feature_vectors(df) → list[FeatureVector]
  Converts DataFrame rows to FeatureVector contracts
  Validates every row — fails loudly on contract violation

- Constants: TIER_WEIGHTS, TIER_1_FEATURES, TIER_2_FEATURES,
  TIER_3_FEATURES matching CLAUDE.md exactly

Write tests/test_features.py:
- Use in-memory DuckDB with synthetic data
- Test: filters low data_quality rows
- Test: engineer_features correct delta calculation
- Test: no NaN in output
- Test: to_feature_vectors raises on null field
```

---

### STEP 2-3 — Training pipeline

```
/plan

Create src/models/train.py.

Read src/models/features.py and src/contracts.py first.

Requirements:
- Function: train(db_path, experiment_name="pulseiq-ess") → str
  Load features via load_features() + engineer_features()

  Build ESS training label:
    raw_label = (
      jobless_claims_delta * 0.35 +
      reddit_negativity_score * 0.25 +
      delinquency_rate * 0.20 +
      poverty_rate * 0.20
    )
    ess_label = normalize to 0-100 range

  Train/test split: 80/20 stratified by income_quartile
  Model: XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8
  )

  MLflow logging: params, RMSE, MAE, R², feature importance
  Save model to models/{run_id}/model.pkl
  Save feature_version (hash of feature column names) alongside
  Return run_id

- if __name__ == "__main__" block

Write tests/test_train.py:
- Synthetic DataFrame 50 rows
- Test: returns string run_id
- Test: model file created at expected path
- Test: feature_version file created
- Mock MLflow
```

---

### STEP 2-4 — Calibration

```
/plan

Create src/models/calibration.py.

Read src/contracts.py Prediction model first.

Requirements:
- Class: PulseIQCalibrator
- Method: fit(predictions, ground_truth_events)
  Uses sklearn IsotonicRegression
  Requires minimum 60 ground truth events to fit
  Raises CalibrationDataError if insufficient data
  Saves calibrator to models/calibration.pkl

- Method: calibrate(prediction: Prediction) → Prediction
  Applies isotonic regression to ess_score
  Sets calibrated=True
  Returns updated Prediction

- Method: confidence_from_coverage(
    ess_score, tier1_coverage, stale_sources) → str
  high: tier1_coverage >= 0.8 and no critical stale
  medium: tier1_coverage >= 0.5
  low: otherwise

- Method: calibration_report() → dict
  Reliability diagram data (predicted buckets vs actual rates)
  This is portfolio evidence — make it informative

- Class: CalibrationDataError(Exception)

- Fallback: if calibration.pkl not found, return prediction
  unchanged with calibrated=False and confidence="low"

Write tests/test_calibration.py:
- Test: raises CalibrationDataError with < 60 events
- Test: calibrate() sets calibrated=True
- Test: fallback when no calibration file
- Test: confidence_from_coverage logic
```

---

### STEP 2-5 — Batch scoring + SHAP

```
/plan

Create src/models/predict.py and src/models/explainer.py.

Read src/models/features.py, src/models/calibration.py,
src/contracts.py, src/observability/ground_truth.py first.

predict.py:
- Function: score_all_geos(db_path, model_run_id) → list[Prediction]
  Load features via load_features() + engineer_features()
  Load model from models/{model_run_id}/model.pkl
  Load calibrator from models/calibration.pkl (fallback gracefully)
  For each geo: score → calibrate → explain (SHAP) → build Prediction
  Compute early_warning: tier3_score > 65 and tier1_score < 45
  Write all Predictions to DuckDB table "ess_scores" (idempotent)
  Log each Prediction via ground_truth.log_prediction()
  Return list of Predictions

explainer.py:
- Function: explain(model, X: pd.DataFrame) → list[dict]
  Uses shap.TreeExplainer
  Returns list of {feature_name: shap_contribution} dicts
  Round to 4 decimal places
  One dict per row

Write tests:
- test_predict.py: mock model + DuckDB, test output schema,
  test idempotency (run twice = same row count)
- test_explainer.py: tiny XGBoost on 20 rows,
  verify explain() length and structure
```

---

### STEP 2-6 — Evaluate

```
/plan

Create src/models/evaluate.py.

Read src/contracts.py and src/observability/ground_truth.py first.

Requirements:
- Class: PulseIQEvaluator
- Method: benchmark(baseline="naive_lag") → dict
  Compare model RMSE vs naive baseline
  (naive: predict next score = current score)
  MUST beat naive or log a clear warning
  Return: {model_rmse, baseline_rmse, improvement_pct, verdict}

- Method: backtest(geo_ids, start_date, end_date) → dict
  Replay predictions against ground_truth_events table
  Return: {early_warning_rate, days_early_avg,
           precision, recall, false_positive_rate}

- Method: threshold_evaluation(thresholds=[60,70,75,80,85]) → dict
  Per threshold: precision, recall, alert_count
  Return as dict keyed by threshold value

- Method: false_positive_review(lookback_days=90) → list[dict]
  Find geos that fired alerts with no confirmed event
  Return: list of {geo_id, alert_date, ess_score, top_signals}

- Method: performance_by_geography() → dict
  Disaggregate RMSE by geo_level (metro vs county vs zip)
  Disaggregate by income_quartile
  This is responsible ML — never skip disaggregation

- Method: performance_by_time_period() → dict
  RMSE per month — detect seasonal degradation
  Flag if any month > 2x overall RMSE

Write tests/test_evaluate.py:
- Use synthetic prediction + ground truth data
- Test: benchmark returns improvement_pct
- Test: threshold_evaluation returns all 5 thresholds
- Test: false_positive_review returns correct structure
```

---

### STEP 2-7 — Monitor

```
/plan

Create src/models/monitor.py.

Read src/contracts.py and src/observability/metrics.py first.

Requirements:
- Class: PulseIQMonitor
- Method: feature_drift(baseline_days=30, current_days=7) → dict
  Compute Population Stability Index (PSI) per feature
  PSI formula: sum((actual_pct - expected_pct) * ln(actual/expected))
  Flag features with PSI > 0.2 as "drift_detected"
  Return: {feature_name: {psi, status}} + overall_status

- Method: score_distribution_drift() → dict
  Compare current week score distribution vs 30-day baseline
  Return: {mean_shift, std_shift, drift_detected}

- Method: missing_source_drift() → dict
  Compare missing source rate this week vs 30-day baseline
  Return: per-source {current_missing_rate, baseline_rate, flagged}

- Method: alert_volume_drift() → dict
  Compare alert count this week vs 30-day baseline
  Return: {current_volume, baseline_volume, pct_change, flagged}

- Method: retraining_recommendation() → dict
  Synthesise all drift signals
  Return: {
    recommendation: "no_action"|"schedule"|"immediate",
    evidence: list[str],
    triggered_by: list[str]
  }
  "immediate" if any PSI > 0.2 or score mean shift > 10 points

Write tests/test_monitor.py:
- Use synthetic DuckDB data
- Test: PSI calculation correct
- Test: retraining_recommendation returns "immediate"
  when feature PSI > 0.2
- Test: "no_action" when all PSI < 0.1
```

---

### STEP 2-8 — RAG layer

```
/plan

Create src/rag/ingest.py, src/rag/retriever.py,
src/rag/explainer.py.

Read src/contracts.py Explanation model first.

ingest.py:
- Use sentence-transformers all-MiniLM-L6-v2 for embeddings
- ChromaDB collection: "pulseiq_news"
- Skip articles already stored (check url in metadata)
- Metadata per article: url, publishedAt, source, geo_id, geo_level
- Function: ingest_news(articles: list[dict]) → int
  Returns count of new articles ingested

retriever.py:
- Class: NewsRetriever
- Method: get_relevant_docs(geo_id, geo_name, days_back=7, top_k=5)
  Filter by publishedAt within days_back
  Return list of {title, description, url, publishedAt, source}

explainer.py (RAG chain):
- Class: StressExplainer
- Method: explain(prediction: Prediction) → Explanation
  Retrieve top 5 docs via NewsRetriever
  Build prompt with:
    geo_id, geo_name, ess_score, delta_7d
    Top 3 SHAP features from prediction.shap_values
    Retrieved news context
  System prompt: enforce four-section structure:
    SUMMARY: one factual sentence
    TOP_DRIVERS: max 3 bullet points
    EVIDENCE: cite retrieved sources
    CAVEATS: missing sources, weak evidence, stale data
  Parse LLM response into Explanation contract
  evidence_strength: "strong" if >= 3 relevant docs,
    "moderate" if 1-2, "weak" if 0
  Never return caveats=[] if missing_sources is non-empty

Write tests:
- test_rag_ingest.py: mock ChromaDB, test deduplication
- test_rag_retriever.py: mock ChromaDB, test date filtering
- test_rag_explainer.py: mock OpenAI + retriever,
  test returns valid Explanation contract,
  test caveats populated when missing_sources present
```

---

# PHASE 3 — API & Dashboard

---

### STEP 3-1 — FastAPI routes

```
/plan

Create src/api/main.py and all route files.
Read src/contracts.py fully first — all responses use those models.
Read CLAUDE.md API standards section.

main.py:
- FastAPI app: title="PulseIQ API", version="1.0.0"
- Include all routers with prefix /api/v1
- Startup: verify DuckDB connection + model artifact exist
- GET /api/v1/health → HealthResponse

routes/scores.py:
- GET /scores/{geo_id} → ScoreResponse
  Latest run_date, 404 if not found
- GET /scores/{geo_id}/history → TimeSeriesResponse
  Query param: window = "7d"|"30d"|"90d"
  Compute trend: improving|stable|deteriorating|volatile
- GET /scores/{geo_id}/drivers → dict
  Top 5 SHAP values + tier breakdown
- GET /scores/top → list[ScoreResponse]
  Query params: limit (default 20, max 100),
  min_confidence (default "low")

routes/explain.py:
- GET /explain/{geo_id} → Explanation
  Call StressExplainer.explain()
- GET /explain/{geo_id}/evidence → list[dict]
  Return raw retrieved docs with relevance scores

routes/alerts.py:
- POST /alerts → AlertPayload
  Validate body, store config in DuckDB
  Return with generated alert_id
- GET /alerts/history/{geo_id} → list[AlertPayload]

routes/health.py:
- GET /health/freshness → list[SourceFreshnessPayload]
- GET /health/model → ModelVersionResponse
- GET /health/pipeline → PipelineStatusResponse

Write tests/test_api.py:
- Use FastAPI TestClient
- Mock DuckDB and StressExplainer
- Test every endpoint: 200 schema, 404 on missing geo
- Test: /health/freshness returns all sources
- Test: /explain/{geo_id} returns Explanation with caveats
```

---

### STEP 3-2 — Streamlit dashboard

```
/plan

Create dashboard/app.py and all component files.

Read src/contracts.py for all response schemas.
The dashboard calls FastAPI endpoints — never queries DuckDB directly.

app.py:
- Page title: "PulseIQ — Economic Stress Monitor"
- Two pages: "Monitor" and "Pipeline Health"
- Sidebar: geo search, date range, refresh button
- st.cache_data ttl=3600 on all API calls
- Clear error if API not running: show instructions

dashboard/components/map.py:
- Plotly choropleth of US coloured by ess_score
- Color scale: green(0) → yellow(50) → red(100)
- Opacity per geo: high=0.85, medium=0.60, low=0.30
- Hatch pattern if missing_sources non-empty
- Tooltip must show: geo_name, score, delta_7d, confidence,
  run_date, missing_sources, granularity_warning
- Global caption: "Data as of {date} · Opacity = confidence
  · Hatching = missing sources"
- On click: populate trend + explanation components

dashboard/components/trend.py:
- Plotly line chart
- Raw score line (thin) + 7-day rolling average (thick)
- Dashed horizontal alert threshold line
- Yellow shading on low-confidence date ranges
- ⚠️ annotation on dates with missing sources
- Window selector: 7d | 30d | 90d (radio buttons)
- Caption: "Trend: {improving|stable|deteriorating|volatile}"

dashboard/components/explanation.py:
- RIGID FOUR-SECTION STRUCTURE — never a single text block:
  Section 1: st.markdown(f"### {summary}")
  Section 2: "Top model drivers" — bullet list, max 3
  Section 3: "Supporting context" — article links
  Section 4: st.warning() with caveats — NEVER OMIT
    even if caveats is empty: show "No caveats identified"
- Evidence strength badge: green|orange|red
- "Generated at {timestamp}" always shown
- Button: "Generate explanation" → calls /explain endpoint

dashboard/components/health.py:
- Table: source, last_run, status, records, latency, 7d trend
- Status icons: ✅ OK, ⚠️ SLOW, 🔴 DOWN
- Model info: version, trained date, calibrated, benchmark result
- Pipeline info: last DAG runs, anomaly flags count
```

---

## Debugging Prompts

### Connector failing
```
The {connector} is failing with:
[paste full error and traceback]

Read src/connectors/{connector}.py and
tests/test_{connector}.py.
Diagnose root cause — read the actual code first.
Fix it and run the tests to confirm.
```

### dbt model failing
```
This dbt model is failing:
[paste error output]

Read the failing model and its upstream dependencies.
Check: column name mismatches, NULL handling, JOIN conditions.
Fix the SQL and explain what was wrong.
```

### Tests failing
```
These tests are failing:
[paste pytest output]

Read the test file and the module it tests.
Fix whichever is wrong — the module or the test.
Explain which was broken and why.
```

### API returning unexpected data
```
GET /api/v1/{endpoint} is returning:
[paste response]

Read routes/{file}.py and contracts.py.
Trace: DuckDB query → pydantic model → response.
Find the mismatch and fix it.
```

### Drift detected in monitor
```
monitor.py returned this drift report:
[paste report]

Read src/models/monitor.py and src/models/evaluate.py.
Determine: is this a data quality issue or genuine drift?
Recommend: no_action | schedule_retrain | retrain_immediately
Show evidence from the report for your recommendation.
```

---

## Progress Check — use weekly

```
Read CLAUDE.md Build Status.

For each incomplete item tell me:
1. What it depends on
2. How complex it is (simple/medium/complex)
3. What prompt to use to build it

Recommend the next three items to build and why.
```
