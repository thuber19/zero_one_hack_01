# Feature Specification: Process Control Room

**Feature Branch**: `004-process-control-room`

**Created**: 2026-05-30

**Status**: Draft

**Input**: Unified web GUI for the Infineon fab-yield model — dark-themed, data-dense, fab-control-room aesthetic for process engineers.

---

## Overview

The Process Control Room is a single-page React application that gives Infineon process engineers a real-time, interactive window into the fab-yield model trained on Leonardo (Spec 001/002). It is emphatically **not** a SaaS marketing page: the aesthetic is dark-mode, data-dense, high-information density — modelled on industrial control rooms.

Five views compose the application. They are specified below as independent user stories ordered by demo priority. The app must be buildable and iterable inside the **Loveable** vibe-coding platform, which constrains the tech stack (see Requirements).

### Demo timing reference (3-minute pitch)

| Time | Action |
|------|--------|
| 0:00–0:30 | Wafer Journey view; pre-loaded "bad batch" — visual drama |
| 0:30–1:15 | Hover flagged step → SHAP panel; "model tells you **why**, not just what" |
| 1:15–2:00 | Drag step → yield drops; move back → recovers |
| 2:00–2:30 | KPI strip ticks up as "good batch" loads |
| 2:30–3:00 | Backend / Leonardo training architecture walk-through |

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Wafer Journey View (Priority: P1)

A process engineer opens the app and immediately sees the full process flow for a selected batch as a horizontal, Sankey-like timeline of ~100 fabrication steps. Steps flagged by the model as high-risk glow red or amber. The engineer hovers a glowing step and sees a popover with raw sensor readings, a confidence interval, and the SHAP contribution for that step.

**Why this priority**: This is the hero visual for the 3-minute demo (0:00–1:15). It is the primary artifact that communicates the model's insight in a way non-ML engineers understand at a glance.

**Independent Test**: Load the app with a hard-coded "bad batch" JSON fixture (no backend required). The timeline renders, red/amber steps are visible, and hovering any step shows a populated popover. Delivers full visual demo value without Story 2–5 being complete.

**Acceptance Scenarios**:

1. **Given** a batch with 100 steps where 8 steps have `risk_score >= 0.7`, **When** the Wafer Journey renders, **Then** those 8 steps glow red (`risk_score >= 0.85`) or amber (`0.7 <= risk_score < 0.85`); all other steps are neutral.
2. **Given** the Wafer Journey is visible, **When** the engineer hovers over step node `step_id=42`, **Then** a popover appears within 100 ms showing: step name, sensor readings (≥1 key:value pair), model confidence interval (lo, hi), and top-3 SHAP contributors (feature name, numeric contribution, sign color).
3. **Given** a batch with >100 steps, **When** the view renders, **Then** the timeline is horizontally scrollable without layout breakage; no step nodes overlap.
4. **Given** a batch with 0 high-risk steps, **When** the view renders, **Then** all steps render in neutral style and no red/amber glow is present.
5. **Given** the backend is unavailable, **When** the page loads, **Then** a clearly visible error banner is shown ("Model API unavailable — displaying cached data" or similar) and the last-cached batch fixture is displayed so the demo can continue.

---

### User Story 2 — Prediction Dashboard / KPI Strip (Priority: P1)

A persistent top bar — always visible regardless of which view is active — displays four animated hero numbers: **Predicted Yield %**, **Risk Steps Detected**, **Model Confidence**, **Anomalous Batches**. Numbers animate (count-up) each time a new batch loads.

**Why this priority**: This strip gives the audience immediate quantitative context during every moment of the demo. It is visible during the "good batch vs bad batch" transition (2:00–2:30 demo window) and reinforces the model's value at a glance.

**Independent Test**: Mount the KPI strip in isolation with two fixture payloads (bad batch → good batch). Verify four numbers render, animate on change, and show correct values. No other views need to function.

**Acceptance Scenarios**:

1. **Given** a batch payload is loaded, **When** the KPI strip mounts or receives new data, **Then** all four numbers animate from their previous value to the new value over ≤800 ms using a count-up easing.
2. **Given** a good batch (`predicted_yield=94.2`, `risk_steps=2`, `confidence=0.91`, `anomalous_batches=0`), **When** the strip renders, **Then** each hero number matches the payload value (within display rounding), uses the correct label, and the yield number is styled green.
3. **Given** a bad batch (`predicted_yield=61.5`, `risk_steps=17`, `confidence=0.73`, `anomalous_batches=3`), **When** the strip renders, **Then** the yield number is styled red/amber and anomalous_batches is non-zero.
4. **Given** the strip is visible and the engineer navigates to a different view, **Then** the strip remains mounted and visible without re-animation.

---

### User Story 3 — SHAP Explainability Panel (Priority: P1)

When the engineer clicks or hovers a step in the Wafer Journey, the SHAP Explainability Panel updates to show a waterfall or beeswarm chart of feature contributions for that step. Feature names are humanized (e.g., `etch_rate_std` → "Etch Rate Variability"). Positive contributions push yield up (green); negative push it down (red).

**Why this priority**: This is the "model tells you **why**, not just what" moment of the demo (0:30–1:15). It is the primary differentiator from a simple anomaly detector.

**Independent Test**: Render the panel with a fixture SHAP payload for a single step. Verify chart renders with correct sign colors, humanized labels, and no JavaScript errors. The Wafer Journey does not need to be interactive for this test.

**Acceptance Scenarios**:

1. **Given** a step is selected and its SHAP data has 10 features, **When** the panel renders, **Then** a waterfall chart shows ≤10 bars sorted by absolute contribution magnitude, with positive bars green and negative bars red.
2. **Given** a feature name `etch_rate_std`, **When** the panel renders, **Then** the label displayed is "Etch Rate Variability" (or equivalent humanized form from the name map); raw snake_case names are never shown to the user.
3. **Given** a step with no SHAP data available (field `shap` is null or empty array), **When** that step is selected, **Then** the panel shows a placeholder: "SHAP data not available for this step" — no crash, no empty chart.
4. **Given** a step is selected, **When** the engineer selects a different step, **Then** the chart transitions smoothly (≤300 ms) to the new step's data without a full remount.

---

### User Story 4 — Step Sequence Optimizer (Priority: P2)

The engineer can drag-and-drop process steps in a kanban/timeline view to reorder the sequence. Within 2 seconds, a new predicted yield score is returned and displayed. This demonstrates that step ordering matters and the model encodes process logic.

**Why this priority**: This is the "drag step → yield drops; move back → recovers" moment (1:15–2:00 demo). It is P2 because it depends on Story 1 (Wafer Journey) being established and requires the `/optimize` endpoint to be live.

**Independent Test**: Render the optimizer with a fixture sequence of 20 steps. Drag one step; verify the reordering state updates in the UI. Wire a mock `/optimize` response; verify the yield score updates within 2 s. The full Wafer Journey rendering is not required for this test.

**Acceptance Scenarios**:

1. **Given** a sequence of steps in the optimizer view, **When** the engineer drags step A from position 5 to position 2, **Then** the sequence reorders immediately in the UI (optimistic) and a loading indicator appears on the yield score.
2. **Given** a reorder is submitted to `POST /optimize`, **When** the server responds within 2 s, **Then** the yield score updates and the loading indicator is replaced.
3. **Given** a reorder results in a lower predicted yield, **When** the score updates, **Then** the yield number animates downward and is styled red/amber.
4. **Given** the engineer drags a step into a position flagged as illegal by the legality oracle (Spec 003), **When** the optimizer renders the result, **Then** the step is placed and scored anyway (legality enforcement is Spec 003's domain); an informational badge "Non-standard order" is shown but no drag is blocked.
5. **Given** the `/optimize` call takes >2 s (slow network), **When** the timeout elapses, **Then** a non-blocking toast notification informs the engineer; the previous score remains visible.

---

### User Story 5 — Anomaly / Batch Inspector (Priority: P2)

A sortable table lists batches ordered by predicted defect probability (descending by default). The engineer clicks a row to open a side drawer showing the full per-step risk breakdown for that batch.

**Why this priority**: Useful for triage but not load-bearing during the 3-minute demo. P2 because it requires both the batch list endpoint and the full per-step data.

**Independent Test**: Render the table with a 10-row fixture (varying defect probabilities). Verify sorting, click to open drawer, drawer shows per-step breakdown. No live API required.

**Acceptance Scenarios**:

1. **Given** `GET /batches` returns 50 batches, **When** the inspector loads, **Then** batches are displayed in a paginated table (20 per page), sorted by `defect_probability` descending by default.
2. **Given** the table is visible, **When** the engineer clicks any column header, **Then** the table re-sorts by that column; a second click reverses order.
3. **Given** the engineer clicks batch row `batch_id=B-042`, **When** the side drawer opens, **Then** it shows batch metadata (ID, timestamp, material) and a per-step risk strip (mini Wafer Journey or list) derived from `GET /batches/{id}`.
4. **Given** a batch row has `defect_probability >= 0.8`, **When** the table renders, **Then** that row is highlighted with a red left border.
5. **Given** the table has more than 200 batches (pagination test), **When** the engineer navigates to page 3, **Then** page 3 loads without re-fetching pages 1–2.

---

### Edge Cases

- **Model API down**: KPI strip and Wafer Journey fall back to the last-cached batch. A banner announces degraded mode. The `/optimize` call shows a toast error. No white screen or uncaught exception.
- **Batch with no SHAP data**: The SHAP panel shows a placeholder message; the Wafer Journey still renders with `risk_score` coloring only.
- **Illegal step position (Spec 003 legality)**: The optimizer accepts the drag and scores it; adds "Non-standard order" badge. No drag blocking.
- **Long batch (>200 steps)**: Wafer Journey timeline is horizontally scrollable; D3 SVG performance must remain responsive (no jank) up to 300 steps. Virtualize or chunk the D3 render if needed.
- **Slow network (>2 s for `/predict` or `/optimize`)**: Show a non-blocking loading indicator; do not block the UI. Toast on timeout.
- **Empty batch list**: Batch Inspector shows a "No batches found" empty state — not a blank table.
- **Browser tab backgrounded during count-up animation**: Animation completes instantly or is skipped; no stale state when tab is re-focused.

---

## Requirements *(mandatory)*

### Hard Tech Stack Constraints (Loveable-compatible)

The following constraints are non-negotiable. They exist to ensure the frontend can be iterated on inside the **Loveable** vibe-coding platform.

| Layer | Technology |
|-------|-----------|
| Frontend build | Vite + React + TypeScript |
| Styling | Tailwind CSS + shadcn/ui components |
| Charts (standard) | Recharts (KPI strip sparklines, SHAP waterfall, batch table charts) |
| Chart (custom SVG) | D3.js — **only** for the Wafer Journey timeline |
| Routing | React Router v6 |
| Server state | TanStack Query (React Query) v5 |
| Backend | FastAPI (Python), single service, REST |
| SHAP computation | Server-side only; shipped as JSON |
| Persistence (optional) | Supabase — batch metadata + saved sequences |
| Frontend hosting | Vercel (for demo deploy) |
| Backend hosting | Fly.io or Render (FastAPI container) |
| **Prohibited** | Streamlit, Gradio, Next.js SSR, Angular, Vue |

### Visual / UX Constraints

- **Theme**: Dark mode only. Background `#0a0f1e` (near-black navy). Accent colors: `#00d4ff` (cyan) for primary actions, `#ff4444` (red) for high-risk, `#ffaa00` (amber) for medium-risk, `#00cc66` (green) for positive yield signals.
- **Typography**: Monospace for all numeric values (sensor readings, yield %, probabilities). Sans-serif for labels.
- **No emojis** anywhere in the UI.
- **No marketing copy**. Labels are terse, technical, and accurate.
- **Information density**: Engineers expect dense UIs. Whitespace is used for grouping, not decoration.

### Functional Requirements

- **FR-001**: The KPI strip MUST be mounted at the top of every view and MUST NOT unmount during client-side navigation.
- **FR-002**: The Wafer Journey MUST render up to 300 steps without visible jank (< 16 ms frame budget per step node during initial paint).
- **FR-003**: The system MUST call `POST /predict` when a batch is selected and cache the response using TanStack Query for the lifetime of the browser session.
- **FR-004**: The SHAP panel MUST display humanized feature names from a static name map bundled with the frontend. The map MUST cover at least all features present in the Infineon demo dataset.
- **FR-005**: The step optimizer MUST submit `POST /optimize` within 200 ms of a drag-end event (debounced — not on every drag-over).
- **FR-006**: All API calls MUST include a 5-second timeout; on timeout the UI MUST surface a non-blocking toast and retain the last valid state.
- **FR-007**: The app MUST be runnable with `npm run dev` against a locally running FastAPI server (`uvicorn main:app --reload`).
- **FR-008**: The app MUST work without Supabase configured (Supabase is optional persistence layer; use env var `VITE_SUPABASE_URL` guard).
- **FR-009**: SHAP background dataset MUST be pre-computed and cached on the FastAPI server at startup; it MUST NOT be recomputed per request.
- **FR-010**: The `/predict` response MUST be cached server-side (in-memory LRU, keyed by `batch_id`) to avoid re-running model inference for repeated requests.

### Data Contracts (exact API shapes)

#### `POST /predict`

**Request**:
```json
{
  "batch_id": "B-042",           // optional — if present, use cached result if available
  "sequence": [                   // optional if batch_id supplied; required otherwise
    {
      "step_id": "S001",
      "step_name": "Gate Oxide Growth",
      "category": "Oxidation",
      "duration_min": 45,
      "sensors": {
        "temperature_c": 950.2,
        "pressure_torr": 0.5,
        "gas_flow_slm": 2.1
      }
    }
  ]
}
```

**Response**:
```json
{
  "batch_id": "B-042",
  "predicted_yield": 0.614,
  "confidence": 0.73,
  "risk_steps_detected": 17,
  "anomalous_batches": 3,
  "per_step": [
    {
      "step_id": "S001",
      "step_name": "Gate Oxide Growth",
      "risk_score": 0.12,
      "confidence_lo": 0.08,
      "confidence_hi": 0.18,
      "shap": [
        { "feature": "temperature_c", "value": 950.2, "contribution": -0.043 },
        { "feature": "pressure_torr", "value": 0.5,   "contribution":  0.011 }
      ]
    }
  ],
  "anomalies": [
    { "step_id": "S042", "anomaly_score": 0.91, "type": "sensor_spike" }
  ]
}
```

**Streaming vs. polling decision**: Use **synchronous REST** (not streaming, not SSE). Justification: the Infineon demo model runs inference in <1 s on a pre-loaded SHAP background; streaming adds frontend complexity for no perceptible user benefit at this scale. If inference ever exceeds 2 s, switch to a polling pattern (`POST /predict/async` → job ID → `GET /predict/{job_id}`). That refactor is out of scope for v1.

---

#### `GET /batches`

**Query params**: `page` (int, default 1), `page_size` (int, default 20), `sort_by` (string, default `defect_probability`), `order` (`asc`|`desc`, default `desc`).

**Response**:
```json
{
  "total": 312,
  "page": 1,
  "page_size": 20,
  "batches": [
    {
      "batch_id": "B-042",
      "timestamp": "2026-05-29T14:22:00Z",
      "material": "Silicon",
      "predicted_yield": 0.614,
      "defect_probability": 0.386,
      "confidence": 0.73,
      "risk_steps_detected": 17
    }
  ]
}
```

---

#### `GET /batches/{batch_id}`

**Response**: Same shape as `POST /predict` response (full `per_step` array + `anomalies`).

---

#### `POST /optimize`

**Request** (smaller payload than `/predict` — no sensor readings needed, just step ordering):
```json
{
  "sequence": [
    { "step_id": "S001", "step_name": "Gate Oxide Growth", "category": "Oxidation" }
  ]
}
```

**Response**:
```json
{
  "predicted_yield": 0.784,
  "confidence": 0.81,
  "cached": false
}
```

**Performance target**: <2 s end-to-end including SHAP. Achieved by: (a) pre-loaded SHAP background at startup, (b) no per-step sensor data needed (ordering signal only), (c) in-memory LRU cache keyed by sequence hash.

---

### Key Entities

- **Batch**: A single fabrication lot. Identified by `batch_id`. Contains an ordered sequence of Steps and a predicted yield + risk summary.
- **Step**: One process operation within a batch. Has `step_id`, `step_name`, `category`, `duration_min`, `sensors` map, `risk_score`, `confidence_lo/hi`, and a `shap` array.
- **SHAP Contribution**: One feature's contribution to the step's risk score. Fields: `feature` (raw name), `value` (raw sensor value), `contribution` (signed float).
- **Anomaly**: A step-level signal that a sensor reading is anomalous. Fields: `step_id`, `anomaly_score`, `type`.
- **FeatureNameMap**: A static TypeScript map bundled in the frontend: `{ [raw_feature: string]: string }`. Example: `{ etch_rate_std: "Etch Rate Variability" }`. Maintained as a `.ts` constant file.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `/predict` and `/optimize` p95 latency < 2,000 ms measured from browser `fetch()` start to JSON parse complete, on a local dev machine with the FastAPI server running locally.
- **SC-002**: Wafer Journey renders a 100-step batch in < 500 ms wall time from API response received to all nodes painted (Chrome DevTools performance panel).
- **SC-003**: KPI strip count-up animation runs at ≥ 60 fps (no dropped frames) for all four numbers simultaneously on a mid-2020 MacBook Pro.
- **SC-004**: The full 3-minute demo flow can be executed without any console errors and without a page reload.
- **SC-005**: A Loveable-platform user can fork the frontend repo and successfully run `npm run dev` within 5 minutes, using only the README instructions.
- **SC-006**: The SHAP panel shows humanized labels for 100% of the features present in the Infineon demo dataset fixture (zero raw snake_case labels visible to the user).
- **SC-007**: Drag-to-reorder in the Step Sequence Optimizer triggers `POST /optimize` exactly once per completed drag (debounce — not on intermediate drag-over events).

---

## Assumptions

- The Infineon fab-yield model (Spec 001/002) is deployed as a callable Python module on the FastAPI server. The FastAPI service wraps model inference; it does not re-train or load weights per request.
- SHAP background dataset is precomputed offline (on Leonardo) and shipped as a `.npy` or `.pkl` file alongside the FastAPI service. It is loaded once at startup (`lifespan` handler).
- The demo runs locally (engineer's laptop, FastAPI on localhost:8000, Vite dev server on localhost:5173). Production deploy targets (Vercel + Fly.io/Render) are considered bonus; local demo is the primary success criterion.
- Authentication is **out of scope**. The API has no auth layer for the hackathon demo.
- The Supabase integration is optional. The app MUST function without it; Supabase is only used to persist saved sequences and batch metadata between demo sessions.
- MLOps, model retraining, CI/CD pipelines, and monitoring dashboards are **out of scope**.
- Step legality enforcement (blocking illegal drags) is **out of scope** — handled by Spec 003. This spec only requires that the optimizer scores and displays illegal orderings without crashing.
- The Step Sequence Optimizer does not need to support sequences longer than 100 steps in the drag UI (longer sequences use the Batch Inspector read-only view).
- Feature name humanization covers the known Infineon demo feature set only. Unknown features (not in the map) fall back to a title-cased, underscore-stripped version of the raw name (e.g., `new_sensor_x` → "New Sensor X").
- No mobile or tablet support is required. The target viewport is a 1920×1080 or 2560×1440 desktop/laptop screen.
