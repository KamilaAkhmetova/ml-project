from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)

GAMMA_PAYLOAD = {
    "fLength": 28.80, "fWidth": 16.00, "fSize": 2.64, "fConc": 0.39, "fConc1": 0.20,
    "fAsym": 27.70, "fM3Long": 22.01, "fM3Trans": -8.20, "fAlpha": 4.09, "fDist": 81.88,
}


def test_root_health():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_or_503_when_no_model():
    r = client.post("/predict", json=GAMMA_PAYLOAD)
    if main.model is None:
        assert r.status_code == 503
        return
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"gamma", "hadron"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["is_gamma"], bool)


def test_batch_or_503_when_no_model():
    r = client.post("/predict/batch", json={"events": [GAMMA_PAYLOAD, GAMMA_PAYLOAD]})
    if main.model is None:
        assert r.status_code == 503
        return
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2


def test_batch_empty_400():
    r = client.post("/predict/batch", json={"events": []})
    assert r.status_code in (400, 503)


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "count" in r.json()


def test_model_version_endpoint():
    r = client.get("/model/version")
    assert r.status_code == 200
    body = r.json()
    assert "model_path" in body
