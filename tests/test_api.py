import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_predict_single_text():
    test_text = "Хороший текст, добрый"
    response = client.post("/predict", json={"text": test_text})
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "toxicity_score" in data
    assert "is_toxic" in data
    assert isinstance(data["toxicity_score"], float)
    assert isinstance(data["is_toxic"], bool)


def test_predict_batch():
    test_texts = [
        "Как же я ненавижу понедельники",
        "Примите пожалуйста лабораторную работу",
        "Что за ужасный код ты написал"
    ]
    
    response = client.post("/predict_batch", json={"texts": test_texts})
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(test_texts)
    
    for prediction in data["predictions"]:
        assert "text" in prediction
        assert "toxicity_score" in prediction
        assert "is_toxic" in prediction


def test_model_info():
    response = client.get("/model_info")
    
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_type" in data
    assert "max_length" in data


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["is_toxic"] is False
