import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env from api/ directory if present
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parents[1] / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        print(f"Loaded env from {_env_file}")
except ImportError:
    pass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import predict, optimize, batches, health, eval_results, infer
from app.services.inference import load_model_if_available
from app.shap.cache import get_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Process Control Room API starting up...")

    # Load ProcessPredictor (optional — falls back to fixture if MODEL_DIR not set)
    predictor = load_model_if_available()
    app.state.predictor = predictor

    # Init LRU cache
    app.state.lru_cache = get_cache()

    print(f"Startup complete. Predictor loaded: {predictor is not None}")
    yield
    print("Shutting down...")


app = FastAPI(title="Process Control Room API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(optimize.router, prefix="/api")
app.include_router(batches.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(eval_results.router, prefix="/api")
app.include_router(infer.router, prefix="/api")
