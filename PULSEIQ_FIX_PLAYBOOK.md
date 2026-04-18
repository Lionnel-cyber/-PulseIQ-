# PulseIQ — Fix Playbook
## Two targeted improvements to an already working system

---

## Rules for this playbook

1. Fix 1 must be fully complete and tested before starting Fix 2
2. Never touch existing pipeline, ML, or API code
3. Show diff before applying every change
4. Run tests after every module — must stay green
5. Use the Session Start prompt at the beginning of every Claude Code session

---

## Session Start — run every time

```
Read CLAUDE.md fully.

Confirm:
1. Current test suite status (run pytest tests/ --co -q to count)
2. Which fix we are working on
3. What the last completed step was

Do not write any code yet.
```

---

## Session End — run before closing

```
Update CLAUDE.md Build Status with what we completed.
Add a one-line note on any decisions made.
List any new known issues discovered.
Do not change anything else in CLAUDE.md.
```

---

# FIX 1 — RSS Feeds for the RAG Layer

Goal: give the RAG explanation chain enough
grounded regional articles to stop returning
"No supporting articles returned" for US metros.

Estimated time: 3-4 hours
Risk level: low — additive only, no existing code touched
Files created: src/rag/rss_ingest.py, tests/test_rss_ingest.py
Files modified: dags/dag_ingest_daily.py, requirements.txt

---

### STEP 1 — Install feedparser

```
Add feedparser to requirements.txt.

Run: pip install feedparser

Verify: python -c "import feedparser; print(feedparser.__version__)"

Show the version number. Do not proceed until
feedparser imports cleanly.
```

---

### STEP 2 — Build rss_ingest.py

```
/plan

Read these files before writing anything:
src/rag/ingest.py
src/contracts.py
CLAUDE.md

Create src/rag/rss_ingest.py.

Requirements:

CONSTANT: PULSEIQ_RSS_FEEDS
Organised as a dict of lists by category.
Use exactly these feeds:

economic_national:
  https://feeds.reuters.com/reuters/businessNews
  https://feeds.reuters.com/reuters/companyNews
  https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml
  https://feeds.feedburner.com/businessinsider

labour_market:
  https://www.bls.gov/feed/rss.xml
  https://www.dol.gov/rss/releases.xml
  https://rss.nytimes.com/services/xml/rss/nyt/Jobs.xml

regional:
  https://www.detroitnews.com/rss/
  https://www.chicagotribune.com/arcio/rss/
  https://www.houstonchronicle.com/rss/
  https://www.latimes.com/business/rss2.0.xml

housing_debt:
  https://www.housingwire.com/feed/
  https://www.calculatedriskblog.com/feeds/posts/default

food_poverty:
  https://feeds.reuters.com/reuters/domesticNews

FUNCTION: fetch_feed(url: str) -> list[dict]
  Parse a single RSS feed using feedparser
  For each entry extract:
    url:         entry.get("link", "")
    title:       entry.get("title", "")
    description: entry.get("summary", "")
    publishedAt: entry.get("published", "")
    source:      feed.feed.get("title", url)
  Skip entries with empty url or title
  Handle feedparser exceptions — log and return []
  Never raise — a broken feed must not stop others
  Return list of article dicts

FUNCTION: ingest_rss_feeds(
    feed_urls: list[str] | None = None
) -> int
  If feed_urls is None use all feeds from
  PULSEIQ_RSS_FEEDS flattened into a single list
  For each url call fetch_feed()
  For each article call the existing ingest
  function from src/rag/ingest.py
  Deduplication is handled by ingest.py (by url)
  Log per feed: name, fetched count, new count
  Log total: feeds processed, total new articles
  Return total new article count as int

Use structured logging (logging module) not print
Full type hints on every function
Docstring on module and every function

Show me the complete file before creating it.
```

---

### STEP 3 — Write tests

```
/plan

Read src/rag/rss_ingest.py and src/rag/ingest.py.

Create tests/test_rss_ingest.py.

Mock ALL external calls:
- feedparser.parse must be mocked — never hit live feeds
- src/rag/ingest.py ingest function must be mocked

Tests to write:

test_fetch_feed_success:
  Mock feedparser.parse to return 3 valid entries
  Assert fetch_feed returns 3 dicts
  Assert each dict has url, title, description,
  publishedAt, source keys

test_fetch_feed_skips_empty_url:
  Mock feedparser with one entry missing "link"
  Assert fetch_feed returns empty list for that entry

test_fetch_feed_handles_exception:
  Mock feedparser.parse to raise Exception
  Assert fetch_feed returns [] — does not raise

test_ingest_rss_feeds_returns_count:
  Mock fetch_feed to return 5 articles
  Mock ingest function to return successfully
  Assert ingest_rss_feeds returns integer > 0

test_ingest_rss_feeds_uses_all_feeds_when_none:
  Call ingest_rss_feeds(None)
  Assert fetch_feed was called once per feed
  in PULSEIQ_RSS_FEEDS (count the total)

test_ingest_rss_feeds_deduplication_handled_by_ingest:
  Mock ingest to raise on duplicate (or return 0)
  Assert ingest_rss_feeds still returns without raising

Run: pytest tests/test_rss_ingest.py -v
All tests must pass before proceeding.
```

---

### STEP 4 — Wire into Airflow DAG

```
Read dags/dag_ingest_daily.py carefully.
Read CLAUDE.md DAG rules section.

Add one new task to dag_ingest_daily.py:

task_rss_ingest:
  Uses @task decorator (TaskFlow API)
  Calls ingest_rss_feeds() from src/rag/rss_ingest.py
  Runs in the existing "connectors" TaskGroup
  alongside the other connector tasks
  Logs: "RSS ingest complete — {n} new articles"
  on_failure_callback same as other tasks

Important constraints:
- Do not change any existing task
- Do not change the DAG schedule
- Do not change cross-DAG sensor logic
- Only add the new task inside the TaskGroup

Show me the diff for dag_ingest_daily.py only.
Apply only after I approve.

After applying run:
python -c "from dags.dag_ingest_daily import dag; print('DAG loads cleanly')"

Must print cleanly with no import errors.
```

---

### STEP 5 — Smoke test end to end

```
Run the RSS ingest manually to verify it works
against real feeds before relying on Airflow:

python -c "
from src.rag.rss_ingest import ingest_rss_feeds
count = ingest_rss_feeds()
print(f'Ingested {count} new articles')
"

If count > 0: Fix 1 is complete.

If count == 0:
  Check ChromaDB — articles may already exist
  from a previous run (deduplication working correctly)
  Run with a fresh ChromaDB path to confirm:
  CHROMADB_PATH=data/test_chroma python -c "..."

After confirming articles are in ChromaDB:
  Test the RAG explanation endpoint manually:
  curl http://localhost:8000/api/v1/explain/MSA-16980

  The explanation should now reference real news sources.
  If it still says "No supporting articles" check
  that CHROMADB_PATH in .env matches where rss_ingest
  wrote to.

Update CLAUDE.md:
  Mark Fix 1 complete
  Note: "RSS feeds added — feedparser, 14 feeds,
  hourly ingest via dag_ingest_daily TaskGroup"
```

---

# FIX 2 — React/Next.js Frontend

Goal: replace the generic Streamlit dashboard
with a dark military intelligence aesthetic that
consumes the existing FastAPI directly.

Estimated time: 2-3 weeks
Risk level: zero to backend — frontend only
Streamlit: kept but not the primary interface
FastAPI: unchanged — just consumed differently

---

## Pre-flight check before starting Fix 2

```
Confirm FastAPI is running and all endpoints respond:

curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/scores/top?limit=5
curl http://localhost:8000/api/v1/health/freshness

All three must return valid JSON before
writing a single line of frontend code.

If any fail: fix the API issue first.
Do not start the frontend on a broken API.
```

---

### STEP 1 — Scaffold Next.js project

```
/plan

Scaffold a new Next.js 14 project inside the
PulseIQ repo at frontend/.

Run:
cd pulseiq
npx create-next-app@latest frontend \
  --typescript \
  --tailwind \
  --app \
  --no-src-dir \
  --import-alias "@/*"

After scaffolding:

1. Delete the default page content in app/page.tsx
   Replace with a single <div>PulseIQ loading...</div>

2. Delete app/globals.css default content
   Replace with just the Tailwind directives

3. Create .env.local in frontend/ with:
   NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1

4. Verify it runs:
   cd frontend && npm run dev
   Must load at localhost:3000 without errors

Show me the scaffolded structure with:
ls -la frontend/app/
ls -la frontend/components/ (create if missing)
```

---

### STEP 2 — Types and API client

```
/plan

Read src/api/schemas.py and src/contracts.py fully.
Every TypeScript type must match the Pydantic
contract exactly — field names, types, optionals.

Create frontend/lib/types.ts

Include TypeScript interfaces for:
- ScoreResponse
- TimeSeriesPoint
- TimeSeriesResponse
- Explanation
- SourceFreshnessPayload
- HealthResponse
- AlertPayload
- ThreatLevel: "monitor" | "elevated" | "critical"
- Confidence: "high" | "medium" | "low"
- GeoLevel: "national" | "state" | "metro" | "county" | "zip"

Create frontend/lib/api.ts

Typed async functions for every endpoint:
- getTopScores(limit?: number): Promise<ScoreResponse[]>
- getScore(geoId: string): Promise<ScoreResponse>
- getHistory(geoId: string, window: "7d"|"30d"|"90d"): Promise<TimeSeriesResponse>
- getExplanation(geoId: string): Promise<Explanation>
- getSourceFreshness(): Promise<SourceFreshnessPayload[]>
- getHealth(): Promise<HealthResponse>

All functions:
- Read NEXT_PUBLIC_API_URL from process.env
- Throw descriptive errors on non-200 responses
- Never use any — full TypeScript strict mode
- Add JSDoc comment on each function

Show me both files before creating them.
```

---

### STEP 3 — Design system and layout

```
/plan

Read frontend/lib/types.ts before proceeding.

Create frontend/app/globals.css
Add these CSS custom properties after the
Tailwind directives:

:root {
  --bg: #0a0c0f;
  --surface: #0f1318;
  --surface2: #141920;
  --border: #1e2730;
  --border2: #253040;
  --accent: #00d4aa;
  --accent2: #0088ff;
  --warn: #f5a623;
  --danger: #e53e3e;
  --text: #c8d8e8;
  --text2: #6a8099;
  --text3: #3a5068;
  --mono: 'Courier New', monospace;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
}

Create frontend/app/layout.tsx

Root layout that:
- Imports globals.css
- Sets metadata: title "PulseIQ", description
- Renders CommandBar at the top (import from
  components/CommandBar.tsx — stub for now)
- Renders children below
- Full height layout: min-h-screen

Create frontend/components/CommandBar.tsx

Top status bar matching the mockup design:
- Fixed height 40px
- Background var(--surface)
- Border bottom 1px var(--border)
- Left: PULSEIQ logo in var(--accent) monospace
         with pulsing green dot + LIVE text
- Middle: stat pills — Geos scored, Elevated,
          Critical, Alerts fired
          Each fetches from GET /health
- Right: model version + UTC timestamp
         updates every minute
- All text in monospace
- Fetch health data with 60s polling interval

Show me CommandBar.tsx before creating it.
After creating: npm run dev must load cleanly.
```

---

### STEP 4 — Sidebar navigation

```
/plan

Read frontend/app/layout.tsx and frontend/lib/types.ts.

Create frontend/components/Sidebar.tsx

Left navigation panel matching the mockup:
Width: 200px fixed
Background: var(--surface)
Border right: 1px var(--border)

Sections:

VIEW section:
  Monitor (links to /)
  Pipeline health (links to /health)
  Alert history (links to /alerts)
  Model drift (links to /drift — stub page)
  Active nav item: left border var(--accent),
  text var(--accent), faint accent background

SIGNAL TIERS section:
  Tier 1 · BLS + FRED (red dot)
  Tier 2 · Prices (amber dot)
  Tier 3 · Search (gray dot)
  Display only — not clickable

FILTERS section (controlled state):
  Geo level: Metro | County | ZIP toggles
  Threshold: All | ≥60 | ≥75 toggles
  Confidence: High | Medium | Low toggles
  Active filter tags: border var(--accent),
  text var(--accent)

Export filter state via callback prop so
parent page can pass to map and event cards.

All section labels: 9px, letter-spacing 2px,
var(--text3), monospace

Show me the full component before creating.
```

---

### STEP 5 — Dual map engine (flat + 3D globe)

```
/plan

Read frontend/lib/types.ts and frontend/lib/api.ts.

Install deck.gl and dependencies:
cd frontend
npm install @deck.gl/react @deck.gl/layers @deck.gl/core
npm install @deck.gl/geo-layers
npm install react-map-gl maplibre-gl

Also install the SVG fallback:
npm install react-simple-maps

Create frontend/components/WorldMap.tsx

This component has TWO view modes toggled by a
button in the map header:
  viewMode: "flat" | "globe"
  Default: "flat"

--- SHARED DATA LOGIC ---

Fetch top scores from getTopScores(500)
Poll every 60 seconds
Store as scores: ScoreResponse[]

Helper: getColor(threatLevel: string) -> [r,g,b,a]
  critical → [229, 62, 62, 230]
  elevated → [245, 166, 35, 215]
  monitor  → [0, 136, 255, 180]

Helper: getRadius(populationAtRisk: number) -> number
  Scale linearly: min 4000m max 80000m
  (deck.gl uses meters for ScatterplotLayer radius)

Helper: getOpacity(confidence: string) -> number
  high   → 0.9
  medium → 0.65
  low    → 0.35

--- FLAT MODE (default) ---

Use DeckGL from @deck.gl/react with MapView

Base map: MapLibre dark style
  mapStyle: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
  No API key required — CartoCDN is free

Layer 1: ScatterplotLayer
  id: "stress-dots"
  data: scores
  getPosition: d => [d.longitude, d.latitude]
  getFillColor: d => getColor(d.threat_level)
  getRadius: d => getRadius(d.population_at_risk)
  opacity: d => getOpacity(d.confidence)
  pickable: true
  radiusMinPixels: 4
  radiusMaxPixels: 20

Layer 2: ScatterplotLayer (pulsing rings — critical only)
  id: "critical-rings"
  data: scores.filter(s => s.threat_level === "critical")
  getPosition: d => [d.longitude, d.latitude]
  getFillColor: [0, 0, 0, 0]
  getLineColor: d => getColor(d.threat_level)
  stroked: true
  getLineWidth: 1
  getRadius: d => getRadius(d.population_at_risk) * 1.8
  Use CSS animation on the canvas overlay for pulse effect

Tooltip on hover:
  Show: geo_name, ess_score, threat_level,
        confidence, missing_sources (if any)
  Style: dark background, monospace, 10px

On click: call onGeoSelect(score.geo_id) prop

Initial view state:
  longitude: -98.5795
  latitude: 39.8283
  zoom: 3.5
  (centers on continental US)

--- GLOBE MODE (toggle) ---

Same DeckGL component, same layers, same data.
Only the view changes:

import { GlobeView } from '@deck.gl/core'

When viewMode === "globe":
  Replace MapView with GlobeView
  Remove mapStyle (no base map tiles on globe)
  Add a dark sphere background: #0a0c0f
  Keep all layers identical — they reproject automatically
  Initial view: longitude: -98, latitude: 35, zoom: 1.5
  Enable auto-rotation: slow spin when no interaction
    Use requestAnimationFrame to increment longitude
    by 0.03 degrees per frame when user is idle
    Stop rotation on mouse interaction
    Resume after 3 seconds of no interaction

--- TOGGLE BUTTON ---

Position: absolute top-right of map container
  [FLAT MAP] [3D GLOBE]
  Monospace, 9px, letter-spacing 1px
  Active: color var(--accent), border var(--accent)
  Inactive: color var(--text3), border var(--border)

--- FALLBACK ---

Detect WebGL support on mount:
  const canvas = document.createElement('canvas')
  const gl = canvas.getContext('webgl')
  if (!gl) setUseFallback(true)

If useFallback === true:
  Render react-simple-maps ComposableMap instead
  US AlbersUsa projection
  Dark fill #0f1318, stroke #1e2730
  Simple colored circles at state centroids
  No toggle button (fallback is flat only)
  Show notice: "WebGL unavailable — basic map mode"
  9px, var(--text3)

--- SCAN LINE ---

CSS overlay div (position: absolute, pointer-events: none)
over the entire map container regardless of view mode:
  1px horizontal line, rgba(0,212,170,0.08)
  CSS keyframe: top 0 → 100% over 4s, linear, infinite

--- MAP FOOTER ---

Absolute bottom-left over the map:
  "Opacity = confidence · Ring = critical
   · Data as of {run_date} · {viewMode === 'globe'
   ? '3D GLOBE' : 'FLAT MAP'}"
  9px, var(--text3), monospace

--- IMPORTANT NOTES FOR CLAUDE CODE ---

deck.gl with Next.js requires dynamic import
to avoid SSR errors. Wrap the DeckGL component:

const DeckGLMap = dynamic(
  () => import('./DeckGLMap'),
  { ssr: false }
)

Create frontend/components/DeckGLMap.tsx as the
actual deck.gl implementation.
WorldMap.tsx wraps it with the dynamic import.

GlobeView is in @deck.gl/core — no extra install.
The auto-rotation uses a useEffect with
requestAnimationFrame — clean up on unmount.

Show me both files before creating either.
After creating: both modes must render at
localhost:3000 with real score data.
Test the toggle switches between modes cleanly.
```

---

### STEP 6 — Event cards

```
/plan

Read frontend/lib/types.ts.

Create frontend/components/EventCards.tsx

Horizontal row of top-N event cards below the map.
Fetch from getTopScores(6) — show top 6.
Accept filter props from Sidebar.

Each card:
  Width: equal flex distribution
  Background: var(--surface)
  Border: 1px var(--border)
  Border left: 3px colored accent
    critical → var(--danger)
    elevated → var(--warn)
    monitor  → var(--accent2)
  Border radius: 3px
  Padding: 8px 10px
  Cursor: pointer — onClick calls onGeoSelect

Card content (top to bottom):
  Tag: "CRITICAL · METRO" — 9px, letter-spacing 1px,
       colored by threat level
  Location: geo_name — 11px, var(--text), bold,
            truncate with ellipsis
  Score: ess_score — 20px, bold, monospace,
         colored by threat level
  Delta: delta_7d with + or - prefix and arrow
         positive → var(--danger) (stress rising)
         negative → var(--accent) (stress falling)
  Meta: confidence + missing sources count
        9px, var(--text3)

Selected card: border 1px var(--accent)

Show me the full component before creating.
After creating: cards must render with real API data.
```

---

### STEP 7 — Intel panel

```
/plan

Read frontend/lib/types.ts and frontend/lib/api.ts.

Create frontend/components/IntelPanel.tsx

Right panel — 280px fixed width
Background: var(--surface)
Border left: 1px var(--border)
Receives selectedGeoId prop from parent page

When no geo selected:
  Show "SELECT A GEOGRAPHY" centered in muted text

When geo selected, fetch:
  getScore(geoId) → score data
  getExplanation(geoId) → briefing (on demand)

Panel sections (top to bottom, each separated
by 1px var(--border) border):

1. HEADER
   "INTEL PANEL" left, geo_name right in var(--warn)
   9px, letter-spacing 2px, var(--text3), monospace

2. ESS SCORE
   Large score number: 36px, bold, monospace
   Colored by threat level
   Score band badge next to it
   Delta below: "+X.X pts / 7d" in appropriate color

3. SIGNAL TIERS
   Three horizontal bars:
   Tier 1 · Hard — fill var(--danger)
   Tier 2 · Medium — fill var(--warn)
   Tier 3 · Soft — fill var(--accent2)
   Each: label left, value right, 3px track height
   Create as frontend/components/TierBars.tsx

4. TOP SHAP DRIVERS
   Top 3 features from shap_values dict
   Sorted by absolute contribution value
   Each: feature name (truncated 20 chars),
         horizontal bar (3px), signed value
   Positive contributions → var(--danger)
   Negative contributions → var(--accent)
   Create as frontend/components/ShapWaterfall.tsx

5. DATA STATUS
   Confidence badge: HIGH/MEDIUM/LOW
     high   → var(--accent) background tinted
     medium → var(--warn) background tinted
     low    → var(--danger) background tinted
   Missing source tags if any
   Stale source tags if any
   Create as frontend/components/ConfidenceBadge.tsx

6. AI BRIEFING (flex: 1, fills remaining space)
   "Generate explanation" button
   On click: calls getExplanation(geoId)
   Shows loading state while fetching
   When loaded, renders four sections:
     Summary: plain text, 11px var(--text2)
     Top drivers: bullet list, 3 items max
     Evidence: article links, var(--accent2)
     Caveats: red-tinted box — NEVER OMIT
       even if caveats is empty: "None identified"
   Create as frontend/components/CaveatBox.tsx

Show me IntelPanel.tsx before creating.
Then create TierBars, ShapWaterfall,
ConfidenceBadge, CaveatBox one at a time.
```

---

### STEP 8 — Main monitor page

```
/plan

Read all components created so far.

Create frontend/app/page.tsx

Full monitor view assembling all components:

Layout: CSS grid
  Columns: 200px (sidebar) | 1fr (main) | 280px (panel)
  Rows: auto (map) | auto (event cards)

State:
  selectedGeoId: string | null — starts null
  sidebarFilters: FilterState object

Data flow:
  Sidebar → filter state → EventCards + WorldMap
  WorldMap click → selectedGeoId → IntelPanel
  EventCards click → selectedGeoId → IntelPanel

No placeholder data anywhere — all real API calls.

After creating: full page must render at localhost:3000
with real data from the FastAPI.

Take a screenshot or describe what you see —
confirm the four main sections are visible:
CommandBar, Sidebar, Map + EventCards, IntelPanel
```

---

### STEP 9 — Health page

```
Create frontend/app/health/page.tsx

Pipeline health view — single column layout.

Fetch from getSourceFreshness() and getHealth().
Poll every 30 seconds.

Sections:

SYSTEM STATUS header row:
  DB status, model status, pipeline status
  Each as a small status pill
  Green = ok, Amber = degraded, Red = down

SOURCE HEALTH table:
  Columns: Source | Last fetch | Status |
           Records | Latency | Freshness
  One row per source
  Status icons:
    fresh    → green dot + "OK"
    stale    → amber dot + "STALE"
    critical → red dot + "CRITICAL"
  Rows sorted by freshness_status (critical first)

MODEL INFO card:
  Model version, feature version, trained date
  Calibrated: yes/no
  Last evaluated: date

All in monospace, dark theme, same design tokens.
```

---

### STEP 10 — Final wiring and launch

```
Final checks before calling Fix 2 complete:

1. npm run build in frontend/
   Must complete with zero TypeScript errors
   Zero is non-negotiable — fix every error

2. npm run lint
   Fix all lint warnings

3. Full functionality check:
   - CommandBar shows live health data
   - Map renders geo dots from real API scores
   - Clicking a dot populates IntelPanel
   - IntelPanel shows real score, tiers, SHAP
   - "Generate explanation" calls real RAG endpoint
   - Caveats section always rendered
   - Event cards ranked by real score data
   - Health page shows all source statuses

4. Update CLAUDE.md:
   Mark Fix 2 complete
   Add note: "React/Next.js frontend live at
   localhost:3000. Streamlit kept at
   dashboard/app.py as fallback."

5. Update README.md:
   Add Frontend section:
   "cd frontend && npm run dev → localhost:3000"
```

---

## After both fixes — what PulseIQ looks like

```
BEFORE                          AFTER
──────────────────────────────────────────────────
Streamlit generic dark UI       Military intelligence aesthetic
US-only choropleth              deck.gl flat map + 3D globe toggle
Empty explanation panel         Grounded RSS-backed briefings
"LLM not configured" warning    Graceful degradation
Disconnected panels             Unified intel command centre
No visual hierarchy             Critical events surface first
Generic widgets                 Monospace command interface
No view options                 [FLAT MAP] [3D GLOBE] toggle
Static map                      Pulsing rings, scan line, auto-spin
```

Both fixes are additive. The backend — pipelines,
ML model, API — is untouched. You are building
on top of something that already works.

## Map mode summary

```
FLAT MAP (default)
  deck.gl ScatterplotLayer on dark CartoCDN tiles
  Best for: analysis, comparing metros, precise coords
  US-centered at zoom 3.5

3D GLOBE (toggle)
  Same deck.gl layers, GlobeView projection
  Auto-rotates slowly when idle, stops on interaction
  Best for: demos, presentations, portfolio screenshots
  One button — same data, zero extra API calls

FALLBACK (automatic)
  react-simple-maps SVG choropleth
  Activates when WebGL not available
  Works on any device
```
