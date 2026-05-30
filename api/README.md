# Process Control Room API

FastAPI backend wrapping the Infineon BERT MLM fab-yield model.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API available at http://localhost:8000. Swagger UI at http://localhost:8000/docs.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_PATH` | No | Path to BERT MLM checkpoint (e.g. `checkpoints/002/checkpoint_best.pt`) |
| `THRESHOLD_PATH` | No | Path to `threshold.json` from `scripts/calibrate_threshold.py` |
| `SHAP_BACKGROUND_PATH` | No | Path to precomputed `shap_background.npy` |

All variables are optional — the API falls back to hard-coded fixtures when the model is unavailable.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check — reports model/SHAP status |
| `POST` | `/api/predict` | Run anomaly detection on a batch sequence |
| `POST` | `/api/optimize` | Predict yield for a reordered sequence |
| `GET` | `/api/batches` | List batches (paginated, sortable) |
| `GET` | `/api/batches/{batch_id}` | Get full batch detail |

## With Real Model (on Leonardo HPC)

```bash
# 1. Calibrate threshold (run on Leonardo SLURM)
python scripts/calibrate_threshold.py \
    --checkpoint $WORK/checkpoints/002/checkpoint_best.pt \
    --splits $WORK/data/fab_sequences/splits.json \
    --data-dir $TMPDIR/fab_sequences \
    --output checkpoints/002/threshold.json

# 2. Set env vars
export MODEL_PATH=checkpoints/002/checkpoint_best.pt
export THRESHOLD_PATH=checkpoints/002/threshold.json

# 3. Start API
uvicorn app.main:app --reload
```
