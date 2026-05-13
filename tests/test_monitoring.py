import numpy as np
import pytest

from api import monitoring


@pytest.fixture(autouse=True)
def reset_log():
    monitoring.PREDICTION_LOG.clear()
    yield
    monitoring.PREDICTION_LOG.clear()


def test_log_and_summary():
    assert monitoring.get_summary()["count"] == 0
    monitoring.log_prediction({"x": 1.0}, 1, 0.8)
    monitoring.log_prediction({"x": 2.0}, 0, 0.3)
    s = monitoring.get_summary()
    assert s["count"] == 2
    assert s["gamma_ratio"] == 0.5
    assert s["hadron_ratio"] == 0.5
    assert s["avg_gamma_proba"] == pytest.approx((0.8 + 0.3) / 2, abs=1e-4)


def test_psi_zero_on_identical_distribution():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 5000)
    quantiles = np.quantile(baseline, np.linspace(0, 1, 11)).tolist()
    actual = rng.normal(0, 1, 5000)
    psi = monitoring.compute_psi(quantiles, actual)
    assert psi < 0.1


def test_psi_large_on_shifted_distribution():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 5000)
    quantiles = np.quantile(baseline, np.linspace(0, 1, 11)).tolist()
    shifted = rng.normal(3, 1, 5000)
    psi = monitoring.compute_psi(quantiles, shifted)
    assert psi > 0.25


def test_compute_psi_all_below_min_samples():
    baseline_stats = {"x": {"quantiles": list(range(11))}}
    monitoring.log_prediction({"x": 1.0}, 1, 0.5)
    result = monitoring.compute_psi_all(baseline_stats, min_samples=100)
    assert result["ready"] is False
