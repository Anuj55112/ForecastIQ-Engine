import pytest
from fastapi.testclient import TestClient
from app.api import app, startup_event

client = TestClient(app)
# Trigger dataset scaling initialization
startup_event()

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_api_predict():
    payload = {
        "history": [float(i) for i in range(96)],
        "model_type": "patchtst"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 24

def test_api_backtest():
    response = client.get("/backtest", params={"model_type": "patchtst"})
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert "actuals" in data
    assert "forecast" in data
    assert "metrics" in data
    assert "MAE" in data["metrics"]
