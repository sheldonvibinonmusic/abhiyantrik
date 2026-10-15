"""Local HTTP inference API for model_artefact_verified.joblib."""
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from final_inference import clean_text, engineer_features  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "model_artefact_verified.joblib"))
MODEL_VERSION = "EXP_C_HYBRID_BLEND-threshold-0.50"
app = FastAPI(title="Rogue Protocol local inference", version="1.0.0")
bundle = None


class LogRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    log_id: str = Field(min_length=1, max_length=128)
    timestamp: str = Field(min_length=1, max_length=64)
    module: str
    mode: str
    node: str
    status: str
    reading: float | None = None
    delay_ms: float = Field(ge=0)
    retry_count: int = Field(ge=0, le=100)
    message: str = Field(max_length=8000)


@app.on_event("startup")
def load_model():
    global bundle
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"available": bundle is not None, "model_version": MODEL_VERSION,
            "runtime_status": "ready" if bundle is not None else "loading", "device": "cpu"}


def predict_record(record: LogRecord):
    started = time.perf_counter()
    row = record.model_dump()
    df = pd.DataFrame([row])
    features = engineer_features(df, bundle["nominal_centers"], bundle["nominal_scales"], bundle["cat_cols"])
    p1 = bundle["struct_m1"].predict_proba(features)[:, 1]
    p2 = bundle["struct_m2"].predict_proba(features)[:, 1]
    p_struct = 0.60 * p1 + 0.40 * p2
    if bundle.get("selected_exp") == "EXP_C_HYBRID_BLEND" and bundle.get("text_pipeline") is not None:
        text_features = bundle["text_pipeline"].transform(df["message"].apply(clean_text))
        p_text = bundle["text_clf"].predict_proba(text_features)[:, 1]
        weight = bundle.get("hybrid_weight_struct", 0.5)
        score = float(weight * p_struct[0] + (1.0 - weight) * p_text[0])
    else:
        score = float(p_struct[0])
    threshold = float(bundle["selected_th"])
    return {"request_id": str(uuid.uuid4()), "log_id": row["log_id"],
            "is_true_fault": int(score >= threshold), "score": score,
            "threshold": threshold, "model_version": MODEL_VERSION,
            "inference_ms": round((time.perf_counter() - started) * 1000, 3)}


@app.post("/predict")
def predict(record: LogRecord):
    try:
        return predict_record(record)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc


@app.post("/predict/batch")
def predict_batch(records: list[LogRecord]):
    if not records or len(records) > 100:
        raise HTTPException(status_code=400, detail="batch must contain 1 to 100 records")
    try:
        return {"predictions": [predict_record(record) for record in records]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Batch inference failed") from exc
