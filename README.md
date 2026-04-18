# PulseIQ 📡
### Near Real-Time Economic Stress Intelligence Platform

> *"By the time the data arrives, the damage is done."*
> PulseIQ closes the gap between economic reality and lagging indicators.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Airflow](https://img.shields.io/badge/Airflow-2.8-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)](https://airflow.apache.org)
[![dbt](https://img.shields.io/badge/dbt-1.7-FF694B?style=flat-square&logo=dbt&logoColor=white)](https://getdbt.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## ✓ Live Now: 11 US Metros

**PulseIQ is live.** Economic stress intelligence updating daily for Atlanta, Boston, Chicago, Dallas, Detroit, Los Angeles, Miami, New York, Philadelphia, Seattle, and Washington DC.

**Built for:** Nonprofits, city planners, community banks, and emergency response teams who need to know *before* the headlines break.

[→ Live Demo](https://pulseiq.app) | [→ API Docs](https://api.pulseiq.app/docs)

---

## The Problem

Banks, nonprofits, and city planners make critical resource allocation decisions
using lagging economic indicators — unemployment rates 30–60 days old, census
figures years stale, quarterly reports describing a world that no longer exists.

Communities in financial distress send signals right now — in search trends,
in news headlines, in job loss filings. Nobody is listening systematically.

**PulseIQ listens.**

---

## What It Does

PulseIQ fuses five public data streams into a per-geography **Economic Stress
Score (ESS)** that updates daily. A RAG-powered AI layer explains *why* a score
is changing. An observability layer ensures you always know whether to trust it.

The system answers five questions every stakeholder needs:

```
What is the score?        GET /scores/{geo_id}
Why is it that score?     GET /explain/{geo_id}
How fresh is the data?    GET /health/freshness
What model produced it?   GET /health/model
Can I trust this run?     GET /health/pipeline
```

---

## Honest Architecture Decisions

### Near real-time, not real-time

PulseIQ uses the word "near real-time" deliberately. The signal stack is:

```
FAST   (hours)    Google Trends     search query spikes by region
FAST   (hours)    NewsAPI           headline sentiment by city
MEDIUM (daily)    FRED              commodity prices, mortgage rates
LAGGING (weekly)  BLS               jobless claims — confirmation signal
BASELINE (annual) Census ACS        poverty/income structural context
```

Reddit was evaluated and rejected — it skews toward urban, younger,
English-speaking users and systematically underrepresents the populations
most vulnerable to economic stress. Google Trends and NewsAPI cover broader,
more representative populations faster.

### Honest geography — no fake precision

Most data sources are not ZIP-level. Claiming ZIP granularity on county or
metro data fabricates precision that doesn't exist. PulseIQ uses a three-tier
geography model where the level of granularity always matches what the data
actually supports:

```
METRO level    BLS MSA data, Google Trends DMA regions, NewsAPI city mentions
COUNTY level   BLS county claims, Census county baseline
ZIP level      Census ZCTA only — labelled "annual baseline" not real-time
```

If a score is displayed at finer grain than it was computed at, the map
renders it at reduced opacity with an explicit warning badge. Users always
know when they are looking at interpolated data.

### Signal tier weighting — hard signals confirm, soft signals warn

Treating Reddit spikes and jobless claims equally produces garbage scores.
PulseIQ uses a three-tier weighting system:

```
TIER 1 — High confidence     weight: 0.55
  BLS jobless claims         0.25   weekly, government-sourced
  FRED delinquency rate      0.20   monthly, Federal Reserve
  Census poverty baseline    0.10   annual, structural

TIER 2 — Medium confidence   weight: 0.30
  FRED commodity prices      0.15   gas/grocery price proxy
  FRED mortgage rates        0.15   housing stress proxy

TIER 3 — Low confidence      weight: 0.15
  Google Trends              0.08   search behaviour proxy
  NewsAPI sentiment          0.07   headline tone by region
```

The model outputs three values not one:

```python
{
  "ess_score": 71.4,        # weighted composite
  "confidence": "medium",   # driven by which tiers have data today
  "early_warning": True     # Tier 3 spiking but Tier 1 still calm
}
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                             │
│  FRED · BLS · NewsAPI · Census · Google Trends · OpenWeather    │
│  + Validation rules (normal/suspicious/impossible per source)   │
│  + Ingestion metrics (latency, record count, freshness)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Raw JSON → data/raw/YYYY/MM/DD/
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION (Airflow)                        │
│  dag_ingest_daily → dag_transform_daily → dag_score_and_alert   │
│  + Idempotent writes   + As-of joins   + Freshness contracts    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRANSFORMATION (dbt + DuckDB)                      │
│  staging/ → intermediate/ → marts/mart_economic_stress          │
│  geo_id · geo_level · geo_name · signal_tier scores             │
│  data_quality_score · granularity_warning · staleness flags     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTRACTS LAYER                              │
│  contracts.py — single typed source of truth for:              │
│  FeatureVector · Prediction · Explanation · AlertContract       │
│  IngestionMetrics · SourceFreshnessPayload · ScoreResponse      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML LAYER                                   │
│  features.py → train.py → calibration.py → predict.py          │
│  explainer.py (SHAP) → evaluate.py → monitor.py                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OBSERVABILITY LAYER                            │
│  metrics.py — per-run ingestion health                          │
│  ground_truth.py — raw signals · predictions · confirmed events │
│  alerts.py — MTTD alerting · suppression rules                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────┬──────────────────────────────────────┐
│      FastAPI Backend     │        Streamlit Dashboard            │
│  Core · Explain ·        │  Choropleth (opacity = confidence)   │
│  Alerts · Health routes  │  Trend · Explanation · Health page   │
└──────────────────────────┴──────────────────────────────────────┘
```

---

## Data Sources

| Source | Signal | Native granularity | Cadence |
|---|---|---|---|
| [BLS API](https://www.bls.gov/developers/) | Jobless claims | MSA / County | Weekly |
| [FRED API](https://fred.stlouisfed.org/docs/api/) | CPI, delinquency, mortgage rates | State / National | Monthly |
| [NewsAPI](https://newsapi.org) | Headline sentiment | City (named in text) | Daily |
| [Google Trends](https://trends.google.com) | Search query spikes | DMA region | Daily |
| [Census ACS](https://www.census.gov/data/developers/) | Poverty, income, housing burden | ZIP (ZCTA) | Annual |
| [OpenWeather](https://openweathermap.org/api) | Climate stress events | City | Daily |

---

## Project Structure

```
pulseiq/
├── dags/
│   ├── dag_ingest_daily.py
│   ├── dag_transform_daily.py
│   └── dag_score_and_alert.py
│
├── src/
│   ├── contracts.py                   ← single source of truth for all types
│   │
│   ├── connectors/
│   │   ├── base_connector.py          ← retry, logging, save_raw
│   │   ├── bls_connector.py           ← geo_id: MSA/county FIPS
│   │   ├── fred_connector.py          ← geo_id: state/national
│   │   ├── news_connector.py          ← geo_id: city, replaces Reddit
│   │   ├── census_connector.py        ← geo_id: ZCTA, annual baseline
│   │   └── openweather_connector.py   ← geo_id: city
│   │
│   ├── validation/
│   │   └── rules.py                   ← per-source normal/suspicious/impossible
│   │
│   ├── transforms/                    ← dbt project
│   │   └── models/
│   │       ├── staging/               ← stg_{source}.sql × 5
│   │       ├── intermediate/          ← int_geo_weekly_signals.sql
│   │       │                            int_sentiment_scores.sql
│   │       └── marts/                 ← mart_economic_stress.sql
│   │
│   ├── models/
│   │   ├── features.py                ← load + engineer 14 features
│   │   ├── train.py                   ← XGBoost + MLflow
│   │   ├── predict.py                 ← batch scoring
│   │   ├── explainer.py               ← SHAP wrapper
│   │   ├── calibration.py             ← isotonic regression calibration
│   │   ├── evaluate.py                ← backtest · benchmark · threshold
│   │   └── monitor.py                 ← drift · retraining signals
│   │
│   ├── rag/
│   │   ├── ingest.py                  ← news → ChromaDB
│   │   ├── retriever.py               ← semantic search
│   │   └── explainer.py               ← LangChain chain
│   │
│   ├── observability/
│   │   ├── metrics.py                 ← IngestionMetrics per run
│   │   ├── ground_truth.py            ← raw signals · predictions · events
│   │   └── alerts.py                  ← MTTD rules · suppression
│   │
│   └── api/
│       ├── main.py
│       └── routes/
│           ├── scores.py              ← core data routes
│           ├── explain.py             ← explanation routes
│           ├── alerts.py              ← alert history + webhooks
│           └── health.py              ← freshness · model · pipeline
│
├── dashboard/
│   ├── app.py
│   └── components/
│       ├── map.py                     ← opacity = confidence, hatch = missing
│       ├── trend.py                   ← rolling avg · annotations · windows
│       ├── explanation.py             ← rigid structure, never freewrite
│       └── health.py                  ← pipeline health page
│
├── data/
│   ├── raw/                           ← partitioned YYYY/MM/DD/source/
│   └── processed/                     ← DuckDB database
│
├── models/                            ← MLflow artifacts
├── tests/                             ← mirrors src/ structure
├── .env.example
├── docker-compose.yml
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## API Contract

Every endpoint answers at least one of the five trust questions.

```
CORE DATA
GET  /api/v1/scores/{geo_id}              Score + confidence + freshness + provenance
GET  /api/v1/scores/{geo_id}/history      Time series with rolling avg + annotations
GET  /api/v1/scores/{geo_id}/drivers      SHAP breakdown + tier sub-scores
GET  /api/v1/scores/top                   Highest-stress geos (filterable by confidence)
GET  /api/v1/alerts/history/{geo_id}      Alert history per geo

EXPLANATION
GET  /api/v1/explain/{geo_id}             Structured narrative — never freewrite
GET  /api/v1/explain/{geo_id}/evidence    Retrieved news articles with relevance scores

OPERATIONAL
GET  /api/v1/health                       DB + model + pipeline status
GET  /api/v1/health/freshness             Per-source: last fetch, days stale, status
GET  /api/v1/health/model                 Version, trained date, calibrated, benchmark
GET  /api/v1/health/pipeline              Last DAG run, records ingested, anomaly flags

ALERTS
POST /api/v1/alerts                       Register webhook with threshold + cooldown
```

---

## Score Response Schema

The score response expresses meaning, not just shape:

```python
ScoreResponse:
  geo_id, geo_name, geo_level, run_date
  ess_score          # 0-100
  score_band         # low | elevated | high | critical
  delta_7d           # change over 7 days
  delta_30d          # change over 30 days
  confidence         # high | medium | low
  missing_sources    # ["bls"] if stale or failed
  stale_sources      # present but beyond freshness threshold
  anomaly_flags      # signals that tripped validation rules
  granularity_warning  # if displayed finer than computed
  model_version
  feature_version
  calibrated         # was calibration.py applied?
  early_warning      # Tier 3 spike, Tier 1 still calm
  tier1_score        # hard signals sub-score
  tier2_score        # medium signals sub-score
  tier3_score        # soft signals sub-score
```

---

## Alert Format

Alerts are never uninterpretable. Every webhook payload includes:

> "Warning: Detroit Metro rose from 61 to 74 over 7 days.
> Top drivers: job-loss search spike (+2.1x), distress
> headline volume elevated. Confidence: medium.
> Details: https://pulseiq.app/explain/MSA-19820"

Suppression rules prevent alert fatigue:
- `threshold_breach` → 3-day cooldown
- `rapid_rise` → 1-day cooldown
- `sustained_high` → 7-day cooldown
- No re-alert if score moved less than 5 points

---

## Dashboard Design Principles

**The map prevents aesthetic deception:**
- Opacity = confidence (low confidence = visually dim)
- Hatching = missing sources (visible on the map itself)
- Tooltip always shows: score, delta, confidence, freshness, warnings
- Global header: "Data as of {date} · Opacity reflects confidence"

**The trend chart answers one question:**
*Is this noise or sustained deterioration?*
- Raw score + 7-day rolling average always shown together
- Confidence bands shaded where data was weak
- Missing source annotations on affected dates
- 7d / 30d / 90d window selector

**The explanation panel never freewrites:**
```
1. Summary      "Score rose 13 points over 7 days"
2. Top drivers  Top 3 SHAP contributors in plain English
3. Evidence     Retrieved news articles with links
4. Caveats      Source gaps · weak evidence · stale data
```
The caveats section is never omitted.

---

## Reliability Properties

| Property | Implementation |
|---|---|
| Source outages | Freshness contracts · stale flag on scores · confidence degrades |
| Time consistency | As-of joins in dbt — most recent known value, not NULL |
| Value validation | Three-tier rules: valid / suspect / rejected per source |
| Schema stability | Versioned pydantic models with graceful fallback |
| Idempotency | `unique_key=(geo_id, run_date)` in dbt · overwrite in save_raw |
| Observability | Per-run IngestionMetrics · MTTD alerting · source health dashboard |
| Ground truth | Raw signals + predictions + timestamps stored permanently |
| Backtesting | evaluate.py replays predictions against confirmed events |
| Drift detection | monitor.py PSI per feature · score distribution · alert volume |

---

## Getting Started

### Prerequisites

- Python 3.11+, Docker & Docker Compose, Git
- Free API keys: BLS · FRED · NewsAPI · Census · OpenWeather · OpenAI

```bash
git clone https://github.com/yourusername/pulseiq.git
cd pulseiq
cp .env.example .env        # add your API keys
docker-compose up -d
pip install -r requirements.txt
cd src/transforms && dbt deps && dbt seed && cd ../..
airflow dags trigger dag_ingest_daily
streamlit run dashboard/app.py
```

### Tests

```bash
pytest tests/ -v --cov=src
dbt test --project-dir src/transforms
curl http://localhost:8000/api/v1/health
```

### Frontend

`cd frontend && npm run dev → localhost:3000`

---

## Phased Rollout — Geography

```
Phase 1 (now)    Top 50 US metros at MSA level
                 All signals genuinely available at this grain
                 Clean, honest, defensible

Phase 2          County level where BLS county data exists
                 500 most populous counties

Phase 3          Selective ZIP level
                 Only where Census ZCTA + confirmed county stress
                 Clearly labelled as estimated, never equivalent precision
```

---

## Design Decisions

**Why DuckDB?** In-process OLAP, zero infrastructure, native window functions.
Scalability path: Delta Lake on S3, same dbt models, different adapter.

**Why XGBoost over neural?** 14 tabular features — XGBoost's regime.
First-class SHAP support is non-negotiable for stakeholder trust.

**Why calibration?** Raw XGBoost scores are not probabilities. Without
isotonic regression calibration, confidence labels are invented not measured.

**Why evaluate.py separate from train.py?** Training metrics lie.
Geographic and temporal disaggregation requires a dedicated evaluation
module. The benchmark vs naive baseline is the most important question.

**Why ground truth logging from day one?** Every day without logging is
training data you can never recover. Backtesting requires the full feature
vector at prediction time — not just the score.

**Why rigid explanation structure?** Freewriting sounds smarter than it is.
Structured output with mandatory caveats is how trust is built and kept.

---

## Author

Built by **[Lionnel-cyber]** — data + AI engineer.

[LinkedIn](https://linkedin.com) · [Portfolio](https://yoursite.com)

---

*Uses only free public APIs. No PII ingested. All outputs aggregated
at geography level. Near real-time, not real-time — always labelled honestly.*
