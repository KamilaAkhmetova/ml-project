"""In-memory prediction logging + Population Stability Index (PSI) computation."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

import numpy as np

PREDICTION_LOG: deque[dict[str, Any]] = deque(maxlen=10_000)
_LOCK = Lock()

_EPS = 1e-6


def log_prediction(features: dict[str, float], label: int, proba: float) -> None:
    with _LOCK:
        PREDICTION_LOG.append({"features": features, "label": int(label), "proba": float(proba)})


def get_summary() -> dict[str, Any]:
    with _LOCK:
        n = len(PREDICTION_LOG)
        if n == 0:
            return {"count": 0, "gamma_ratio": None, "hadron_ratio": None, "avg_gamma_proba": None}
        gamma = sum(1 for e in PREDICTION_LOG if e["label"] == 1)
        avg_proba = float(np.mean([e["proba"] for e in PREDICTION_LOG]))
        return {
            "count": n,
            "gamma_ratio": gamma / n,
            "hadron_ratio": 1 - gamma / n,
            "avg_gamma_proba": round(avg_proba, 4),
        }


def compute_psi(baseline_quantiles: list[float], actual: np.ndarray) -> float:
    """PSI between a baseline distribution (given as quantile cutpoints) and actual values.

    PSI = Σ (a - e) * ln(a / e), where e/a are bucket proportions.
    """
    bins = np.array(baseline_quantiles, dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    n_buckets = len(bins) - 1

    expected = np.full(n_buckets, 1.0 / n_buckets)
    counts, _ = np.histogram(actual, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    actual_prop = counts / total

    expected = np.clip(expected, _EPS, None)
    actual_prop = np.clip(actual_prop, _EPS, None)
    psi = float(np.sum((actual_prop - expected) * np.log(actual_prop / expected)))
    return psi


def compute_psi_all(baseline_stats: dict[str, dict], min_samples: int = 100) -> dict[str, Any]:
    with _LOCK:
        n = len(PREDICTION_LOG)
        if n < min_samples:
            return {
                "count": n,
                "ready": False,
                "message": f"Need >= {min_samples} predictions, have {n}",
            }
        records = list(PREDICTION_LOG)

    by_feature: dict[str, list[float]] = {f: [] for f in baseline_stats}
    for rec in records:
        for f in by_feature:
            v = rec["features"].get(f)
            if v is not None:
                by_feature[f].append(float(v))

    psi_per_feature: dict[str, float] = {}
    for f, values in by_feature.items():
        if not values:
            continue
        psi_per_feature[f] = round(compute_psi(baseline_stats[f]["quantiles"], np.array(values)), 4)

    max_psi = max(psi_per_feature.values()) if psi_per_feature else 0.0
    return {
        "count": n,
        "ready": True,
        "psi": psi_per_feature,
        "max_psi": round(max_psi, 4),
        "drift_detected": max_psi > 0.2,
        "thresholds": {"no_drift": "< 0.1", "moderate": "0.1 - 0.2", "drift": "> 0.2"},
    }
