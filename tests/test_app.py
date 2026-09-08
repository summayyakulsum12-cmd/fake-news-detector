"""
Basic smoke tests. Run with: pytest
Trains the demo model first if it hasn't been trained yet.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src import ml_model
from src.train_ml_model import main as train_main


@pytest.fixture(scope="module", autouse=True)
def ensure_model_trained():
    if not ml_model.is_ready():
        train_main(os.path.join("data", "sample_news.csv"))


def test_ml_model_predicts_real():
    result = ml_model.predict(
        "Central bank raises interest rates citing inflation data reviewed by economists."
    )
    assert result["label"] in {"REAL", "FAKE"}
    assert 0 <= result["confidence"] <= 1


def test_ml_model_predicts_fake_like():
    result = ml_model.predict(
        "Shocking secret cure banned by big pharma, insiders reveal miracle overnight fix."
    )
    assert result["label"] in {"REAL", "FAKE"}


def test_app_health(monkeypatch):
    from app import app

    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_api_predict_endpoint():
    from app import app

    client = app.test_client()
    resp = client.post("/api/predict", json={"text": "This is a normal test headline about weather."})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["result"]["final_verdict"] in {"REAL", "FAKE"}
