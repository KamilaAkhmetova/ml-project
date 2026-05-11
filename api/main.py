from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os

app = FastAPI(
    title="MAGIC Telescope Classifier API",
    description="Classifies high-energy particles as gamma rays (signal) or hadrons (noise).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/xgboost.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded from  {MODEL_PATH}")
    print(f"✅ Scaler loaded from {SCALER_PATH}")
except FileNotFoundError as e:
    print(f"⚠️  Could not load model/scaler: {e}")
    print("   Run notebooks/03_Modeling.ipynb first to generate model files.")
    model  = None
    scaler = None

FEATURE_NAMES = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"
]

class ParticleEvent(BaseModel):
    fLength:  float = Field(..., description="Major axis of ellipse [mm]", example=28.7967)
    fWidth:   float = Field(..., description="Minor axis of ellipse [mm]", example=16.0021)
    fSize:    float = Field(..., description="log10(sum of pixel contents)", example=2.6449)
    fConc:    float = Field(..., description="Ratio: 2 highest pixels / fSize", example=0.3918)
    fConc1:   float = Field(..., description="Ratio: highest pixel / fSize", example=0.1982)
    fAsym:    float = Field(..., description="Distance from highest pixel to center [mm]", example=27.7004)
    fM3Long:  float = Field(..., description="3rd root of 3rd moment along major axis [mm]", example=22.0110)
    fM3Trans: float = Field(..., description="3rd root of 3rd moment along minor axis [mm]", example=-8.2027)
    fAlpha:   float = Field(..., description="Angle of major axis with vector to origin [deg]", example=40.0920)
    fDist:    float = Field(..., description="Distance from origin to ellipse center [mm]", example=81.8828)


class PredictionResponse(BaseModel):
    prediction:  str   
    is_gamma:    bool
    confidence:  float  
    label:       int    


class BatchRequest(BaseModel):
    events: list[ParticleEvent]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total:       int
    gamma_count: int
    hadron_count: int


def predict_events(events: list[ParticleEvent]) -> list[PredictionResponse]:
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run notebooks/03_Modeling.ipynb first."
        )

    X = np.array([[
        e.fLength, e.fWidth, e.fSize, e.fConc, e.fConc1,
        e.fAsym, e.fM3Long, e.fM3Trans, e.fAlpha, e.fDist
    ] for e in events])

    X_scaled = scaler.transform(X)
    labels    = model.predict(X_scaled)
    proba     = model.predict_proba(X_scaled)[:, 1]  

    results = []
    for label, prob in zip(labels, proba):
        results.append(PredictionResponse(
            prediction = "gamma" if label == 1 else "hadron",
            is_gamma   = bool(label == 1),
            confidence = round(float(prob), 4),
            label      = int(label)
        ))
    return results


@app.get("/", tags=["Health"])
def root():
    """Health check — returns API status and model load state."""
    return {
        "status":       "running",
        "model_loaded": model is not None,
        "description":  "MAGIC Gamma Telescope Classifier API",
        "endpoints": {
            "single":  "POST /predict",
            "batch":   "POST /predict/batch",
            "docs":    "GET  /docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(event: ParticleEvent):
    results = predict_events([event])
    return results[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    if len(request.events) == 0:
        raise HTTPException(status_code=400, detail="No events provided.")
    if len(request.events) > 10000:
        raise HTTPException(status_code=400, detail="Max 10,000 events per batch.")

    predictions = predict_events(request.events)
    gamma_count  = sum(1 for p in predictions if p.is_gamma)

    return BatchResponse(
        predictions  = predictions,
        total        = len(predictions),
        gamma_count  = gamma_count,
        hadron_count = len(predictions) - gamma_count
    )
