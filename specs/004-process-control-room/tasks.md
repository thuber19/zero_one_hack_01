# Tasks: Process Control Room GUI

**Input**: Design documents from `/specs/004-process-control-room/`

**Prerequisites**: plan.md ✓, spec.md ✓

**Organization**: Tasks grouped by user story. Frontend lives in `frontend/`, backend in `api/`.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Parallelizable (different files, no blocking deps)
- **[Story]**: User story label (US1–US5)

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Replace Next.js boilerplate with Vite stack; scaffold backend.

- [ ] T001 Remove Next.js boilerplate from `frontend/frontend/` and reinitialize `frontend/` as a Vite 5 + React 18 + TypeScript project using `bun create vite@latest . -- --template react-ts`
- [ ] T002 Install frontend dependencies: `react-router-dom@6`, `@tanstack/react-query@5`, `zustand@4`, `recharts@2`, `d3@7`, `@dnd-kit/core`, `@dnd-kit/sortable` in `frontend/package.json`
- [ ] T003 [P] Install frontend dev dependencies: `tailwindcss@3`, `autoprefixer`, `postcss`, `@types/d3`, `vitest` in `frontend/package.json`
- [ ] T004 [P] Initialize shadcn/ui in `frontend/` with dark mode (`npx shadcn@latest init`); configure `components.json` with `baseColor=slate`, `cssVariables=true`
- [ ] T005 Configure Tailwind CSS dark mode + design tokens in `frontend/tailwind.config.ts`: background `#0a0f1e`, accent `#00d4ff`, risk-red `#ff4444`, risk-amber `#ffaa00`, yield-green `#00cc66`
- [ ] T006 [P] Create `frontend/.env.example` with `VITE_API_URL=http://localhost:8000` and `VITE_SUPABASE_URL=` (empty, optional)
- [ ] T007 [P] Configure `frontend/vite.config.ts` with dev proxy: `/api` → `http://localhost:8000`
- [ ] T008 [P] Create `api/` directory scaffold: `api/app/routes/`, `api/app/services/`, `api/app/shap/`, `api/app/models/`, `api/app/fixtures/`
- [ ] T009 [P] Create `api/requirements.txt` with pinned versions: `fastapi==0.110.*`, `pydantic>=2.0`, `uvicorn[standard]`, `torch`, `shap`, `numpy`, `pandas`, `cachetools`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared code both frontend and backend depend on. Must be complete before any user story.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T010 Define all TypeScript types in `frontend/src/types/api.ts`: `Batch`, `Step`, `ShapContribution`, `Anomaly`, `PredictResponse`, `BatchSummary`, `BatchListResponse`, `OptimizeResponse`
- [ ] T011 Create API client with 5s timeout, error handling, and base URL in `frontend/src/lib/api.ts`; all fetch calls go through this wrapper
- [ ] T012 [P] Create Zustand store in `frontend/src/lib/store.ts`: `activeBatch`, `selectedStep`, `isDegradedMode` state slices
- [ ] T013 [P] Create 100-step "bad batch" fixture JSON in `frontend/src/fixtures/bad-batch.json`: `predicted_yield=0.614`, `confidence=0.73`, `risk_steps_detected=17`, `anomalous_batches=3`, 17 steps with `risk_score >= 0.70`, SHAP arrays populated on high-risk steps
- [ ] T014 [P] Create 100-step "good batch" fixture JSON in `frontend/src/fixtures/good-batch.json`: `predicted_yield=0.942`, `confidence=0.91`, `risk_steps_detected=2`, `anomalous_batches=0`, 2 steps with `risk_score >= 0.70`
- [ ] T015 [P] Create static feature name map in `frontend/src/lib/featureNameMap.ts`: maps raw snake_case sensor names (e.g. `etch_rate_std`) to humanized labels; includes fallback `toTitleCase(name.replace(/_/g, ' '))`
- [ ] T016 Re-export fixtures via `frontend/src/lib/fixtures.ts` with typed exports `badBatch` and `goodBatch`
- [ ] T017 Install and configure shadcn/ui Toast component in `frontend/src/components/ui/`; create `useToast` hook for non-blocking notifications
- [ ] T018 [P] Create `ErrorBanner` component in `frontend/src/components/ErrorBanner.tsx`: renders "Model API unavailable — displaying cached data" with yellow warning styling when `isDegradedMode` is true
- [ ] T019 Configure TanStack Query `QueryClient` with `staleTime=Infinity` and `retry=1` in `frontend/src/main.tsx`
- [ ] T020 Create `frontend/src/App.tsx` layout shell: React Router `<BrowserRouter>`, `<Routes>` with 4 routes, `KpiStrip` always mounted above `<Outlet>` (FR-001)
- [ ] T021 Create FastAPI app in `api/app/main.py` with lifespan handler (stubs for model + SHAP pre-warm), CORS for `localhost:5173`, and include all 4 routers
- [ ] T022 [P] Create Pydantic v2 models in `api/app/models/predict.py` (`PredictRequest`, `PredictResponse`, `StepInput`, `StepResult`, `ShapContribution`, `Anomaly`), `api/app/models/optimize.py` (`OptimizeRequest`, `OptimizeResponse`), `api/app/models/batch.py` (`BatchSummary`, `BatchListResponse`)
- [ ] T023 [P] Stub all 4 routes returning fixture data: `api/app/routes/predict.py`, `api/app/routes/optimize.py`, `api/app/routes/batches.py`, `api/app/routes/health.py`
- [ ] T024 [P] Create dependency injection in `api/app/deps.py`: `get_model()`, `get_shap_explainer()`, `get_lru_cache()` as FastAPI dependencies

**Checkpoint**: `bun run dev` (frontend) + `uvicorn app.main:app --reload` (backend) both start without errors; `/health` returns `{"status":"ok"}`

---

## Phase 3: User Story 1 — Wafer Journey View (Priority: P1) 🎯 MVP

**Goal**: D3 horizontal timeline of ~100 fabrication steps. High-risk steps glow red/amber. Hover any step to see sensors, confidence interval, and top-3 SHAP contributors.

**Independent Test**: Load `localhost:5173` with bad-batch fixture hardcoded (no backend). Timeline renders, 17 red/amber steps visible, hovering step `step_id=42` shows populated popover. No backend needed.

- [ ] T025 [US1] Create `usePrediction` hook in `frontend/src/hooks/usePrediction.ts`: TanStack Query mutation wrapping `POST /api/predict`; falls back to fixture on error and sets `isDegradedMode=true` in store
- [ ] T026 [US1] Implement D3 SVG base layout in `frontend/src/components/WaferJourney.tsx`: horizontal timeline, step nodes as `<rect>` or `<circle>`, category-grouped lanes, fixed `height=120px` per lane
- [ ] T027 [US1] Add risk coloring to step nodes in `frontend/src/components/WaferJourney.tsx`: `risk_score >= 0.85` → red `#ff4444` glow, `0.70 <= risk_score < 0.85` → amber `#ffaa00` glow, else neutral `#1e2a3a`
- [ ] T028 [P] [US1] Add horizontal scroll container in `frontend/src/components/WaferJourney.tsx` for batches >100 steps; SVG width scales with step count, no node overlap guaranteed by fixed step width
- [ ] T029 [US1] Create `StepPopover` component in `frontend/src/components/StepPopover.tsx`: shows step name, ≥1 sensor key:value pair (monospace), confidence interval `[lo, hi]`, top-3 SHAP contributors with sign color; appears on `mouseenter`, dismisses on `mouseleave`
- [ ] T030 [US1] Wire `mouseenter`/`mouseleave` D3 events to show `StepPopover` and update `selectedStep` in Zustand store in `frontend/src/components/WaferJourney.tsx`
- [ ] T031 [US1] Create `WaferJourneyPage` in `frontend/src/pages/WaferJourneyPage.tsx`: loads bad-batch fixture by default, calls `usePrediction` on batch select, renders `WaferJourney` + `ErrorBanner` if `isDegradedMode`
- [ ] T032 [US1] Wire route `"/"` to `WaferJourneyPage` in `frontend/src/App.tsx`

**Checkpoint**: Navigate to `/`; 100-step D3 timeline renders in < 500 ms; 17 red/amber steps visible; hover shows popover within 100 ms.

---

## Phase 4: User Story 2 — KPI Strip (Priority: P1)

**Goal**: Persistent top bar with 4 count-up animated hero numbers: Predicted Yield %, Risk Steps Detected, Model Confidence, Anomalous Batches. Always visible during navigation.

**Independent Test**: Mount `KpiStrip` in isolation with bad-batch → good-batch fixture swap. Verify 4 numbers render, animate on data change, correct colors, persists across route change.

- [ ] T033 [US2] Create `useCountUp` hook in `frontend/src/hooks/useCountUp.ts`: uses `requestAnimationFrame`, animates from previous to target value over ≤ 800 ms with easing; handles browser tab background (complete instantly on re-focus)
- [ ] T034 [US2] Implement `KpiStrip` component in `frontend/src/components/KpiStrip.tsx`: 4 hero numbers using `useCountUp`; reads `activeBatch` from Zustand store; labels: "Predicted Yield", "Risk Steps", "Confidence", "Anomalous Batches"
- [ ] T035 [US2] Apply color coding in `frontend/src/components/KpiStrip.tsx`: yield green `#00cc66` if `>= 0.85`, amber if `>= 0.70`, red if `< 0.70`; anomalous_batches red if `> 0`; monospace font for all numeric values
- [ ] T036 [US2] Mount `KpiStrip` in `frontend/src/App.tsx` above `<Outlet>` so it never unmounts during client-side navigation (FR-001)

**Checkpoint**: Navigate between all 4 routes; KpiStrip remains mounted and visible; loading bad then good batch animates all 4 numbers.

---

## Phase 5: User Story 3 — SHAP Explainability Panel (Priority: P1)

**Goal**: When a step is selected in the Wafer Journey, show a waterfall chart of SHAP feature contributions. Positive = green, negative = red. Humanized labels only.

**Independent Test**: Render `ShapPanel` with a fixture SHAP payload for a single step. Waterfall chart with correct sign colors, humanized labels, no JS errors. WaferJourney does not need to be interactive.

- [ ] T037 [US3] Implement `ShapPanel` component in `frontend/src/components/ShapPanel.tsx`: Recharts `BarChart` waterfall sorted by absolute contribution magnitude; positive bars `#00cc66`, negative bars `#ff4444`; max 10 bars
- [ ] T038 [US3] Apply humanized feature labels in `frontend/src/components/ShapPanel.tsx` using `featureNameMap`; unknown features → `toTitleCase(raw.replace(/_/g, ' '))`; zero raw snake_case ever shown (SC-006)
- [ ] T039 [US3] Add 300 ms CSS transition on chart data change in `frontend/src/components/ShapPanel.tsx` (no full remount)
- [ ] T040 [US3] Add empty state in `frontend/src/components/ShapPanel.tsx`: show "SHAP data not available for this step" when `shap` is null or empty array
- [ ] T041 [US3] Update `WaferJourney.tsx` step click handler to update `selectedStep` in Zustand store (in addition to hover popover)
- [ ] T042 [US3] Create `ShapPage` in `frontend/src/pages/ShapPage.tsx`: renders `WaferJourney` (read-only, no hover needed) side-by-side with `ShapPanel` driven by `selectedStep` from store
- [ ] T043 [US3] Wire route `"/shap"` to `ShapPage` in `frontend/src/App.tsx`

**Checkpoint**: Select a step in Wafer Journey; ShapPanel shows waterfall with humanized labels and correct sign colors; selecting another step transitions smoothly.

---

## Phase 6: User Story 4 — Step Sequence Optimizer (Priority: P2)

**Goal**: Drag-and-drop step reordering triggers POST /optimize within 200 ms of drag-end (debounced). Yield score updates within 2 s. Optimistic UI on drag.

**Independent Test**: Render `SequenceOptimizer` with 20-step fixture. Drag one step; UI reorders immediately. Mock `/optimize` response; yield updates within 2 s.

- [ ] T044 [US4] Create `useOptimize` hook in `frontend/src/hooks/useOptimize.ts`: TanStack Query mutation wrapping `POST /api/optimize`; 200 ms debounce on drag-end (FR-005); 5 s timeout with toast on expire (FR-006); returns `{ predictedYield, isLoading, isError }`
- [ ] T045 [US4] Set up dnd-kit `DndContext` + `SortableContext` in `frontend/src/components/SequenceOptimizer.tsx`; implement `useSortable` for each step card
- [ ] T046 [US4] Implement optimistic reorder in `frontend/src/components/SequenceOptimizer.tsx`: update step list order immediately on `onDragEnd`; fire `useOptimize` debounced call
- [ ] T047 [US4] Show loading spinner on yield score while `/optimize` is in-flight in `frontend/src/components/SequenceOptimizer.tsx`
- [ ] T048 [US4] Animate yield score change in `frontend/src/components/SequenceOptimizer.tsx` using `useCountUp`; apply red/amber color if new yield < previous
- [ ] T049 [P] [US4] Add "Non-standard order" informational badge on steps that deviate from canonical sequence in `frontend/src/components/SequenceOptimizer.tsx` (no drag blocking)
- [ ] T050 [US4] Create `OptimizerPage` in `frontend/src/pages/OptimizerPage.tsx`: loads first 20 steps from active batch fixture; renders `SequenceOptimizer`
- [ ] T051 [US4] Wire route `"/optimize"` to `OptimizerPage` in `frontend/src/App.tsx`

**Checkpoint**: Drag step on OptimizerPage; UI reorders immediately; yield updates within 2 s; dragging back restores score; toast appears if mock timeout > 2 s.

---

## Phase 7: User Story 5 — Anomaly / Batch Inspector (Priority: P2)

**Goal**: Sortable, paginated table of batches ordered by defect probability. Click row → side drawer with per-step risk breakdown.

**Independent Test**: Render `BatchInspector` with 10-row fixture. Verify sorting, click opens drawer with per-step breakdown. No live API needed.

- [ ] T052 [US5] Create `useBatches` hook in `frontend/src/hooks/useBatches.ts`: TanStack Query with `page`, `pageSize`, `sortBy`, `order` params wrapping `GET /api/batches`; falls back to fixture on error
- [ ] T053 [US5] Create `useBatchDetail` hook in `frontend/src/hooks/useBatchDetail.ts`: fetches `GET /api/batches/{batch_id}`; cached for session lifetime
- [ ] T054 [US5] Implement paginated + sortable shadcn Table in `frontend/src/components/BatchInspector.tsx`: 20 rows per page; columns: batch_id, timestamp, material, predicted_yield, defect_probability, confidence, risk_steps_detected
- [ ] T055 [US5] Add column header click handler for sort toggle in `frontend/src/components/BatchInspector.tsx` (click = sort asc, second click = sort desc)
- [ ] T056 [US5] Apply red left border to rows with `defect_probability >= 0.8` in `frontend/src/components/BatchInspector.tsx`
- [ ] T057 [US5] Implement shadcn Drawer on row click in `frontend/src/components/BatchInspector.tsx`: shows batch metadata + mini per-step risk strip from `useBatchDetail`
- [ ] T058 [US5] Add "No batches found" empty state in `frontend/src/components/BatchInspector.tsx`
- [ ] T059 [US5] Create `BatchInspectorPage` in `frontend/src/pages/BatchInspectorPage.tsx`: renders `BatchInspector` with 10-row fixture default
- [ ] T060 [US5] Wire route `"/batches"` to `BatchInspectorPage` in `frontend/src/App.tsx`

**Checkpoint**: Navigate to `/batches`; 10-row table renders sorted by defect_probability desc; click row opens drawer with per-step breakdown; pagination and sort toggling work.

---

## Phase 8: Backend — FastAPI Live Implementation

**Purpose**: Replace stub routes with real model inference and SHAP computation. Can run in parallel with Phases 3–7 (frontend uses fixtures).

- [ ] T061 [P] Create backend fixture files: `api/app/fixtures/bad_batch.json` (100 steps, yield=0.614, 17 risk steps) and `api/app/fixtures/good_batch.json` (100 steps, yield=0.942, 2 risk steps)
- [ ] T062 [P] Implement `GET /health` returning model_loaded, shap_background_loaded, cache_size, status in `api/app/routes/health.py`
- [ ] T063 Implement SHAP background `.npy` loader (loaded once at startup) in `api/app/shap/background.py`; path configurable via `SHAP_BACKGROUND_PATH` env var
- [ ] T064 Implement SHAP explainer in `api/app/shap/explainer.py`: try `shap.DeepExplainer` first, fall back to `shap.KernelExplainer`; initialized once in lifespan (FR-009)
- [ ] T065 Implement LRU cache in `api/app/shap/cache.py` using `cachetools.LRUCache`; keyed by `batch_id` for predict, `hash(tuple(step_ids))` for optimize
- [ ] T066 Implement inference service in `api/app/services/inference.py`: loads Spec 001 PyTorch checkpoint from `MODEL_PATH` env var; exposes `predict(sequence) -> PredictResponse`; uses pre-loaded SHAP explainer
- [ ] T067 Implement optimize service in `api/app/services/optimize_service.py`: runs inference on reordered sequence (no sensor data needed, ordering signal only); returns yield + confidence
- [ ] T068 Implement `POST /predict` with LRU cache check (keyed by `batch_id`) and SHAP computation in `api/app/routes/predict.py`; serve fixture JSON if model not loaded (FR-010)
- [ ] T069 Implement `POST /optimize` with sequence-hash LRU cache in `api/app/routes/optimize.py`; 200 ms budget (backend side)
- [ ] T070 Implement `GET /batches` with pagination (`page`, `page_size`) and sorting (`sort_by`, `order`) in `api/app/routes/batches.py`; serve from in-memory fixture list if no Supabase
- [ ] T071 Implement `GET /batches/{batch_id}` delegating to inference service (or fixture) in `api/app/routes/batches.py`
- [ ] T072 Wire lifespan handler in `api/app/main.py` to load model, background, and pre-warm SHAP explainer at startup
- [ ] T073 [P] Create `api/Dockerfile` for Fly.io/Render deploy: Python 3.11 slim, copy requirements + app, `CMD uvicorn app.main:app --host 0.0.0.0 --port 8000`
- [ ] T074 [P] Create `api/docker-compose.yml` for local dev: `api` service on port 8000 with model volume mount

---

## Phase 9: Polish & Cross-Cutting Concerns

- [ ] T075 [P] Add navigation header/sidebar in `frontend/src/App.tsx` with links to all 4 routes (WaferJourney, SHAP Panel, Optimizer, Batch Inspector)
- [ ] T076 [P] Update `frontend/index.html` with page title "Process Control Room" and `#0a0f1e` theme-color meta
- [ ] T077 Validate all FR requirements: FR-001 (KpiStrip always mounted), FR-002 (300-step D3 no jank), FR-005 (optimize debounce 200 ms), FR-006 (5 s timeout toast), FR-007 (`npm run dev` works), FR-008 (works without Supabase)
- [ ] T078 [P] Write `frontend/README.md` with `npm run dev` 5-minute setup instructions (SC-005); include `VITE_API_URL` env var setup
- [ ] T079 [P] Write `api/README.md` with `uvicorn app.main:app --reload` startup instructions and `MODEL_PATH` / `SHAP_BACKGROUND_PATH` env vars
- [ ] T080 Verify 3-minute demo flow: bad batch loads → 17 red steps → hover popover → SHAP waterfall → drag step → yield drops → good batch KPI animation. Zero console errors.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **User Stories (Phases 3–7)**: All depend on Phase 2; can run in priority order (US1 → US2 → US3 → US4 → US5)
- **Backend (Phase 8)**: Can run in parallel with Phases 3–7 (frontend uses fixtures)
- **Polish (Phase 9)**: Depends on all target stories being complete

### User Story Dependencies

- **US1 (Wafer Journey)**: Foundational only — no story deps
- **US2 (KPI Strip)**: Foundational only; integrates with US1 Zustand store
- **US3 (SHAP Panel)**: Depends on US1 (WaferJourney step selection via Zustand)
- **US4 (Sequence Optimizer)**: Foundational only; uses US1 fixture data
- **US5 (Batch Inspector)**: Foundational only; independent of other stories

---

## Parallel Example: Phase 2 (Foundational)

```bash
# These can run in parallel (different files):
Task: T012 Create Zustand store in frontend/src/lib/store.ts
Task: T013 Create bad-batch fixture JSON in frontend/src/fixtures/bad-batch.json
Task: T014 Create good-batch fixture JSON in frontend/src/fixtures/good-batch.json
Task: T015 Create featureNameMap in frontend/src/lib/featureNameMap.ts
Task: T022 Create Pydantic models in api/app/models/
Task: T023 Stub API routes in api/app/routes/
```

---

## Implementation Strategy

### MVP First (User Stories 1–3 Only, ~6 h)

1. Complete Phase 1: Setup (~1 h)
2. Complete Phase 2: Foundational (~2 h)
3. Complete Phase 3: US1 Wafer Journey (~2 h)
4. Complete Phase 4: US2 KPI Strip (~1 h)
5. Complete Phase 5: US3 SHAP Panel (~1 h)
6. **STOP and VALIDATE**: Full 0:00–1:15 demo segment works with fixtures

### Incremental Delivery

- MVP (US1–3): Hero visual + SHAP explainability — covers first 75 s of demo
- Add US4 (Optimizer): Adds "drag → yield drops" demo moment (1:15–2:00)
- Add US5 (Batch Inspector): Adds triage table — useful but not load-bearing for demo
- Wire Backend (Phase 8): Upgrade from fixtures to live inference

---

## Notes

- All frontend paths relative to repo root; `frontend/` is the Vite SPA root
- After Phase 1, `frontend/frontend/` (Next.js boilerplate) should be deleted
- `[P]` tasks = different files, no blocking dependencies — safe to parallelize
- `[Story]` label maps each task to its user story for traceability
- Commit after each checkpoint to preserve working state
- shadcn/ui components go in `frontend/src/components/ui/` (auto-generated via CLI)
- All monospace numerics: use CSS `font-family: 'JetBrains Mono', monospace` or `font-mono` Tailwind class
