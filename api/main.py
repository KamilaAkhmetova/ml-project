import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api import monitoring  # noqa: E402
from src.feature_engineering import RAW_FEATURES, engineer_features  # noqa: E402

app = FastAPI(
    title="MAGIC Telescope Classifier API",
    description="Classifies high-energy particles as gamma rays (signal) or hadrons (noise) using a stacking ensemble (RF + XGBoost + LightGBM → LR) on physics-engineered features.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/stacking_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
SELECTED_FEATURES_PATH = os.getenv("SELECTED_FEATURES_PATH", "models/selected_features.json")
BASELINE_STATS_PATH = os.getenv("BASELINE_STATS_PATH", "models/baseline_stats.json")


def _load_artifact(path: str, loader, label: str):
    try:
        obj = loader(path)
        print(f"Loaded {label}: {path}")
        return obj
    except FileNotFoundError as e:
        print(f"Could not load {label} ({path}): {e}")
        return None


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _file_sha(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except FileNotFoundError:
        return None


model = _load_artifact(MODEL_PATH, joblib.load, "model")
scaler = _load_artifact(SCALER_PATH, joblib.load, "scaler")
selected_features = _load_artifact(SELECTED_FEATURES_PATH, _load_json, "selected_features")
baseline_stats = _load_artifact(BASELINE_STATS_PATH, _load_json, "baseline_stats")


class ParticleEvent(BaseModel):
    fLength: float = Field(..., description="Major axis of ellipse [mm]", examples=[28.7967])
    fWidth: float = Field(..., description="Minor axis of ellipse [mm]", examples=[16.0021])
    fSize: float = Field(..., description="log10(sum of pixel contents)", examples=[2.6449])
    fConc: float = Field(..., description="Ratio: 2 highest pixels / fSize", examples=[0.3918])
    fConc1: float = Field(..., description="Ratio: highest pixel / fSize", examples=[0.1982])
    fAsym: float = Field(..., description="Distance from highest pixel to center [mm]", examples=[27.7004])
    fM3Long: float = Field(..., description="3rd root of 3rd moment along major axis [mm]", examples=[22.011])
    fM3Trans: float = Field(..., description="3rd root of 3rd moment along minor axis [mm]", examples=[-8.2027])
    fAlpha: float = Field(..., description="Angle of major axis with vector to origin [deg]", examples=[40.092])
    fDist: float = Field(..., description="Distance from origin to ellipse center [mm]", examples=[81.8828])


class PredictionResponse(BaseModel):
    prediction: str
    is_gamma: bool
    confidence: float
    label: int


class BatchRequest(BaseModel):
    events: list[ParticleEvent]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    gamma_count: int
    hadron_count: int


def predict_events(events: list[ParticleEvent]) -> list[PredictionResponse]:
    if model is None or scaler is None or selected_features is None:
        raise HTTPException(
            status_code=503,
            detail="Model artefacts not loaded. Run `python src/train_stacking.py` first.",
        )

    raw_df = pd.DataFrame([{f: getattr(e, f) for f in RAW_FEATURES} for e in events])
    eng_df = engineer_features(raw_df)
    scaled = scaler.transform(eng_df)
    scaled_df = pd.DataFrame(scaled, columns=eng_df.columns, index=eng_df.index)
    X = scaled_df[selected_features]

    labels = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    results: list[PredictionResponse] = []
    for i, (lab, p) in enumerate(zip(labels, proba)):
        feat_dict = {col: float(scaled_df.iloc[i][col]) for col in selected_features}
        monitoring.log_prediction(feat_dict, int(lab), float(p))
        results.append(PredictionResponse(
            prediction="gamma" if lab == 1 else "hadron",
            is_gamma=bool(lab == 1),
            confidence=round(float(p), 4),
            label=int(lab),
        ))
    return results


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "model_loaded": model is not None and scaler is not None and selected_features is not None,
        "description": "MAGIC Gamma Telescope Classifier API (stacking ensemble)",
        "endpoints": {
            "single": "POST /predict",
            "batch": "POST /predict/batch",
            "metrics": "GET /metrics",
            "drift": "GET /monitoring/psi",
            "model": "GET /model/version",
            "docs": "GET /docs",
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(event: ParticleEvent):
    return predict_events([event])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    if not request.events:
        raise HTTPException(status_code=400, detail="No events provided.")
    if len(request.events) > 10_000:
        raise HTTPException(status_code=400, detail="Max 10,000 events per batch.")
    preds = predict_events(request.events)
    gamma = sum(1 for p in preds if p.is_gamma)
    return BatchResponse(
        predictions=preds,
        total=len(preds),
        gamma_count=gamma,
        hadron_count=len(preds) - gamma,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return monitoring.get_summary()


@app.get("/monitoring/psi", tags=["Monitoring"])
def psi():
    if baseline_stats is None:
        raise HTTPException(status_code=503, detail="Baseline stats not loaded.")
    return monitoring.compute_psi_all(baseline_stats)


@app.get("/model/version", tags=["Monitoring"])
def model_version():
    return {
        "model_path": MODEL_PATH,
        "model_sha256_prefix": _file_sha(MODEL_PATH),
        "scaler_sha256_prefix": _file_sha(SCALER_PATH),
        "n_selected_features": len(selected_features) if selected_features else 0,
        "selected_features": selected_features,
    }
