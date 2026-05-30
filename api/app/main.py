import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env from api/ directory if present
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parents[2] / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        print(f"Loaded env from {_env_file}")
except ImportError:
    pass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import predict, optimize, batches, health
from app.services.inference import load_model_if_available
from app.shap.background import load_background
from app.shap.explainer import init_explainer
from app.shap.cache import get_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Process Control Room API starting up...")

    # Load model (optional — falls back to fixture if MODEL_PATH not set)
    model, tokenizer = load_model_if_available()
    app.state.model = model
    app.state.tokenizer = tokenizer

    # Load SHAP background (optional)
    background = load_background()
    app.state.shap_background = background

    # Init SHAP explainer (optional)
    if model is not None and background is not None:
        init_explainer(model, background)
    app.state.shap_explainer = None  # store reference if needed

    # Init LRU cache
    app.state.lru_cache = get_cache()

    print(f"Startup complete. Model loaded: {model is not None}, SHAP background: {background is not None}")
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
