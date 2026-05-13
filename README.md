# MAGIC Gamma Telescope — Particle Classification

Automatically classify high-energy particles captured by the MAGIC telescope as **gamma rays (signal)** or **hadrons (noise)** using a **stacking ensemble** built on **physics-informed engineered features**.

Reference article: https://amrskk.github.io/mlfinal/

---

## Architecture

```
                        ┌─────────────────────────┐
  10 raw Hillas params  │  FastAPI service        │  prediction + confidence
  (from frontend)  ───► │  (api/main.py)          │  ──────────────────────►
                        │  1. engineer_features() │
                        │  2. scaler.transform()  │
                        │  3. stack.predict()     │
                        │  4. log_prediction()    │
                        └────────┬────────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │   StackingClassifier        │
                  │ ┌────┐ ┌────┐ ┌──────────┐ │
                  │ │ RF │ │XGB │ │ LightGBM │ │  → LogisticRegression (meta)
                  │ └────┘ └────┘ └──────────┘ │
                  └─────────────────────────────┘
```

## Project layout

```
MLFinalProject/
├── api/
│   ├── main.py             # FastAPI: /predict, /predict/batch, /metrics, /monitoring/psi, /model/version
│   └── monitoring.py       # rolling prediction log + PSI computation
├── src/
│   ├── feature_engineering.py  # 7 physics-engineered features + constraint validation
│   ├── preprocessing.py        # load + engineer + split + Borderline-SMOTE + scale
│   ├── train_stacking.py       # Optuna + MLflow + SHAP + StackingClassifier
│   ├── train_model.py          # (legacy) baseline 4-model trainer
│   └── predict.py              # CLI batch predictor
├── frontend/
│   └── index.html          # static client, sends 10 raw fields to /predict
├── tests/                  # pytest: feature engineering, monitoring, API
├── models/                 # stacking_model.pkl, scaler.pkl, selected_features.json, baseline_stats.json
├── mlruns/                 # MLflow file store (created on first train)
├── data/
│   └── magic04.data        # UCI raw dataset
├── notebooks/              # EDA / Preprocessing / Modeling notebooks
├── Dockerfile              # python:3.12-slim, libgomp1 for LightGBM
├── docker-compose.yml      # api (8000) + nginx frontend (3000)
└── requirements.txt
```

## Dataset

**MAGIC Gamma Telescope** — UCI ([archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope))

- 19,020 events (12,332 gamma + 6,688 hadron)
- 10 Hillas parameters (geometric descriptors of the Cherenkov ellipse)
- Binary target: `g` → 1, `h` → 0

### Engineered features (added on top of the 10 raw ones)

| Feature | Formula | Physics rationale |
|---|---|---|
| `ellipticity` | `fLength / fWidth` | Gamma showers are elongated |
| `shower_density` | `fSize / (fLength * fWidth)` | Light concentration in ellipse |
| `miss_parameter` | `fDist * sin(fAlpha)` | Distance from source axis |
| `conc_ratio` | `fConc / fConc1` | Distinguishes EM vs hadronic cascade |
| `m3_magnitude` | `√(fM3Long² + fM3Trans²)` | Total skewness |
| `size_conc` | `fSize * fConc` | Energy-weighted concentration |
| `long_asymmetry` | `fAsym / fLength` | Asymmetry normalized by length |

## Model

**Stacking ensemble:**
- Base learners: `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`
- Meta-learner: `LogisticRegression`
- 5-fold internal CV, `passthrough=False`
- Optuna TPESampler, 30 trials per base learner, objective = mean ROC-AUC over 5-fold StratifiedKFold
- Borderline-SMOTE applied only on the training fold, synthetic rows violating physical constraints are discarded
- Feature selection: `mutual_info_classif` (drop bottom 10 %) + correlation filter (`|r| > 0.95`)
- SHAP summary plot logged as MLflow artifact

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | health check + load state |
| POST | `/predict` | single prediction (10 raw fields) |
| POST | `/predict/batch` | up to 10,000 events |
| GET | `/metrics` | rolling counts + gamma ratio + avg confidence |
| GET | `/monitoring/psi` | per-feature PSI vs training baseline (drift detection) |
| GET | `/model/version` | model sha256 prefix + selected features |
| GET | `/docs` | Swagger UI |

## How to run

### Train the stacking model (≈ 30–60 min on CPU)
```bash
pip install -r requirements.txt
python src/train_stacking.py
```
Produces:
- `models/stacking_model.pkl`
- `models/scaler.pkl`
- `models/selected_features.json`
- `models/baseline_stats.json`
- `mlruns/` (browse with `mlflow ui --backend-store-uri ./mlruns`)

### Launch the service
```bash
docker compose up --build
```
- API → http://localhost:8000 (Swagger at `/docs`)
- Frontend → http://localhost:3000

### Run tests
```bash
pytest tests/ -q
ruff check api src tests
```

## CI

`.github/workflows/ci.yml` runs lint (ruff) + pytest + docker build on every push/PR to `main`.

## Monitoring

After ≥ 100 predictions, `GET /monitoring/psi` returns PSI per feature:
- `< 0.1` — no drift
- `0.1 – 0.2` — moderate drift
- `> 0.2` — drift detected (retraining recommended)

Baseline distribution (10-bucket quantiles per feature) is snapshotted during training into `models/baseline_stats.json`.
