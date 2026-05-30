# Process Control Room

Dark-mode React SPA for Infineon fab-yield model visualization.

## Quick Start (5 minutes)

### Prerequisites
- [Bun](https://bun.sh) (or Node 18+)

### Frontend

```bash
cd frontend
bun install
cp .env.example .env
bun run dev
```

Open http://localhost:5173

### Backend (optional)

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API runs at http://localhost:8000. Frontend works without the backend — all views fall back to hard-coded fixtures.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | (empty) | Backend URL, e.g. `http://localhost:8000` |
| `MODEL_PATH` | (empty) | Path to BERT MLM checkpoint `.pt` file |
| `THRESHOLD_PATH` | (empty) | Path to `threshold.json` from calibration |
| `SHAP_BACKGROUND_PATH` | (empty) | Path to `shap_background.npy` |

## Demo Flow (3 minutes)

1. **Wafer Journey** (`/`) — Load bad batch → 17 red/amber steps visible → hover any step for sensor + SHAP popover
2. **SHAP Panel** (`/shap`) — Click a step → waterfall chart of feature contributions
3. **Optimizer** (`/optimize`) — Drag steps to reorder → yield score updates
4. **Batch Inspector** (`/batches`) — Sort by defect probability → click row for detail drawer
