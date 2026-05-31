import csv
import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[3]
_RESULTS_DIR = _REPO_ROOT / "tracks/industrial-infineon/solution/procseq_base_d20000_s16001_seed11101"
_PARTICIPANT_DIR = _REPO_ROOT / "tracks/industrial-infineon/participant_files"

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
_METRICS: dict = {}
_COMPLETIONS: list[dict] = []
_COMPLETIONS_BY_ID: dict[str, dict] = {}
_ANOMALIES: list[dict] = []
_ANOMALIES_BY_ID: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return list of row dicts. Returns [] if file missing."""
    if not path.exists():
        logger.warning("CSV file not found: %s", path)
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_metrics() -> None:
    global _METRICS

    # Load decoder metrics
    metrics_path = _RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        logger.warning("metrics.json not found: %s", metrics_path)
        raw = {}
    else:
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Rename task3_anomaly → task3_encoder for the response
    _METRICS = {
        "task1_nextstep": raw.get("task1_nextstep", {}),
        "task2_completion": raw.get("task2_completion", {}),
        "task2_logic_validity": raw.get("task2_logic_validity", 1.0),
        "task3_encoder": raw.get("task3_anomaly", {}),
    }

    # Load hybrid metrics and merge
    hybrid_path = _RESULTS_DIR / "metrics_hybrid.json"
    if not hybrid_path.exists():
        logger.warning("metrics_hybrid.json not found: %s", hybrid_path)
        hybrid_raw = {}
    else:
        hybrid_raw = json.loads(hybrid_path.read_text(encoding="utf-8"))

    _METRICS["task1_hybrid"] = hybrid_raw.get("task1_nextstep", {})
    _METRICS["task3_hybrid"] = hybrid_raw.get("task3_anomaly", {})


def _load_completions() -> None:
    global _COMPLETIONS, _COMPLETIONS_BY_ID

    # Load eval_input_valid from participant_files (uses valid_XXXX IDs that match nextstep/completion)
    valid_rows = _read_csv(_PARTICIPANT_DIR / "eval_input_valid.csv")
    valid_map: dict[str, dict] = {r["EXAMPLE_ID"]: r for r in valid_rows}

    nextstep_rows = _read_csv(_RESULTS_DIR / "nextstep.csv")
    nextstep_map: dict[str, dict] = {r["EXAMPLE_ID"]: r for r in nextstep_rows}

    completion_rows = _read_csv(_RESULTS_DIR / "completion.csv")
    completion_map: dict[str, dict] = {r["EXAMPLE_ID"]: r for r in completion_rows}

    items: list[dict] = []
    for eid, v in valid_map.items():
        ns = nextstep_map.get(eid)
        cp = completion_map.get(eid)
        # Skip if missing from either prediction file
        if ns is None or cp is None:
            continue

        partial_steps = [s for s in v.get("PARTIAL_SEQUENCE", "").split("|") if s]
        predicted_steps = [s for s in cp.get("PREDICTED_SEQUENCE", "").split("|") if s]

        item = {
            "example_id": eid,
            "family": v.get("FAMILY", ""),
            "completion_fraction": float(v.get("COMPLETION_FRACTION", 0)),
            "partial_steps": partial_steps,
            "partial_step_count": len(partial_steps),
            "rank1": ns.get("RANK_1", ""),
            "rank2": ns.get("RANK_2", ""),
            "rank3": ns.get("RANK_3", ""),
            "rank4": ns.get("RANK_4", ""),
            "rank5": ns.get("RANK_5", ""),
            "predicted_sequence_steps": predicted_steps,
            "predicted_step_count": len(predicted_steps),
        }
        items.append(item)

    _COMPLETIONS = items
    _COMPLETIONS_BY_ID = {item["example_id"]: item for item in items}


def _load_anomalies() -> None:
    global _ANOMALIES, _ANOMALIES_BY_ID

    anomaly_rows = _read_csv(_RESULTS_DIR / "eval_input_anomaly.csv")
    anomaly_map: dict[str, dict] = {r["EXAMPLE_ID"]: r for r in anomaly_rows}

    submission_rows = _read_csv(_RESULTS_DIR / "submission_task3_hybrid.csv")
    submission_map: dict[str, dict] = {r["EXAMPLE_ID"]: r for r in submission_rows}

    items: list[dict] = []
    for eid, a in anomaly_map.items():
        sub = submission_map.get(eid)
        if sub is None:
            continue

        full_sequence = [s for s in a.get("SEQUENCE", "").split("|") if s]
        score_raw = sub.get("SCORE", "")
        score = float(score_raw) if score_raw.strip() else 0.5

        item = {
            "example_id": eid,
            "family": a.get("FAMILY", ""),
            "full_sequence": full_sequence,
            "sequence_step_count": len(full_sequence),
            "is_valid": sub.get("IS_VALID", "0") == "1",
            "score": score,
            "predicted_rule": sub.get("PREDICTED_RULE", "") or "",
        }
        items.append(item)

    _ANOMALIES = items
    _ANOMALIES_BY_ID = {item["example_id"]: item for item in items}


def _load() -> None:
    """Load all data at module import time."""
    try:
        _load_metrics()
    except Exception as exc:
        logger.error("Failed to load metrics: %s", exc)

    try:
        _load_completions()
    except Exception as exc:
        logger.error("Failed to load completions: %s", exc)

    try:
        _load_anomalies()
    except Exception as exc:
        logger.error("Failed to load anomalies: %s", exc)


_load()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/eval/metrics")
async def get_metrics():
    """Return all evaluation metrics."""
    return JSONResponse(content=_METRICS)


@router.get("/eval/completions")
async def list_completions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    family: str = Query(""),
    sort_by: str = Query("completion_fraction"),
    order: Literal["asc", "desc"] = Query("desc"),
):
    valid_sort = {"example_id", "family", "completion_fraction", "partial_step_count", "predicted_step_count"}
    if sort_by not in valid_sort:
        sort_by = "completion_fraction"

    items = _COMPLETIONS
    if family:
        items = [x for x in items if x["family"] == family]

    reverse = order == "desc"
    items = sorted(items, key=lambda x: x[sort_by], reverse=reverse)

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    # List view: omit partial_steps and predicted_sequence_steps
    row_keys = ("example_id", "family", "completion_fraction", "partial_step_count",
                "predicted_step_count", "rank1", "rank2", "rank3", "rank4", "rank5")
    rows = [{k: item[k] for k in row_keys} for item in page_items]

    return {"total": total, "page": page, "page_size": page_size, "items": rows}


@router.get("/eval/completions/{example_id}")
async def get_completion(example_id: str):
    """Return full completion detail including partial_steps and predicted_sequence_steps."""
    item = _COMPLETIONS_BY_ID.get(example_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Completion not found: {example_id}")
    return item


@router.get("/eval/anomalies")
async def list_anomalies(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    family: str = Query(""),
    is_valid: str = Query(""),
    sort_by: str = Query("is_valid"),
    order: Literal["asc", "desc"] = Query("asc"),
):
    valid_sort = {"example_id", "family", "is_valid", "score", "sequence_step_count"}
    if sort_by not in valid_sort:
        sort_by = "is_valid"

    items = _ANOMALIES
    if family:
        items = [x for x in items if x["family"] == family]
    if is_valid == "1":
        items = [x for x in items if x["is_valid"]]
    elif is_valid == "0":
        items = [x for x in items if not x["is_valid"]]

    reverse = order == "desc"
    items = sorted(items, key=lambda x: x[sort_by], reverse=reverse)

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    # List view: omit full_sequence
    row_keys = ("example_id", "family", "is_valid", "score", "predicted_rule", "sequence_step_count")
    rows = [{k: item[k] for k in row_keys} for item in page_items]

    return {"total": total, "page": page, "page_size": page_size, "items": rows}


@router.get("/eval/anomalies/{example_id}")
async def get_anomaly(example_id: str):
    """Return full anomaly detail including full_sequence."""
    item = _ANOMALIES_BY_ID.get(example_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Anomaly not found: {example_id}")
    return item
