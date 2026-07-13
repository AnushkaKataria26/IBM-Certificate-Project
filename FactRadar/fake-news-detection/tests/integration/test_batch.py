import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_batch_predict_valid_articles():
    articles = [
        {"article_id": f"id_{i}", "text": f"This is valid article number {i} with sufficient words to pass."}
        for i in range(5)
    ]
    response = client.post("/predict/batch", json={"articles": articles})
    assert response.status_code == 200
    data = response.json()
    
    assert "results" in data
    assert len(data["results"]) == 5
    
    for i in range(5):
        assert data["results"][i]["article_id"] == f"id_{i}"
        assert data["results"][i]["predicted_label"] in ["fake", "real"]
        assert data["results"][i]["confidence"] is not None
        
    assert data["summary"]["total_processed"] == 5
    assert data["summary"]["total_failed"] == 0

def test_batch_predict_empty_list():
    response = client.post("/predict/batch", json={"articles": []})
    assert response.status_code == 422
    assert "articles list cannot be empty" in response.text

def test_batch_predict_exceeds_max_limit():
    articles = [{"article_id": f"id_{i}", "text": f"Valid text {i}"} for i in range(501)]
    response = client.post("/predict/batch", json={"articles": articles})
    assert response.status_code == 422
    assert "batch size exceeds maximum limit" in response.text

def test_batch_predict_partial_invalid():
    articles = [
        {"article_id": "id_0", "text": "This is a valid article with enough words."},
        {"article_id": "id_1", "text": "   "}, # Invalid, empty after cleaning
        {"article_id": "id_2", "text": "Valid text another one"},
        {"article_id": "id_3", "text": "Valid text here too"},
        {"article_id": "id_4", "text": "And the final valid text."}
    ]
    response = client.post("/predict/batch", json={"articles": articles})
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["results"]) == 5
    assert data["results"][1]["predicted_label"] is None
    assert data["results"][1]["warning"] == "invalid_input"
    
    # Check the other four are valid
    valid_count = sum(1 for r in data["results"] if r["predicted_label"] is not None)
    assert valid_count == 4
    
    assert data["summary"]["total_processed"] == 4
    assert data["summary"]["total_failed"] == 1

def test_batch_predict_duplicate_ids():
    articles = [
        {"article_id": "duplicate_id", "text": "First article with duplicate ID."},
        {"article_id": "duplicate_id", "text": "Second article with duplicate ID."}
    ]
    response = client.post("/predict/batch", json={"articles": articles})
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["results"]) == 2
    assert data["results"][0]["article_id"] == "duplicate_id"
    assert data["results"][1]["article_id"] == "duplicate_id"
    
    assert data["summary"]["total_processed"] == 2
    assert data["summary"]["total_failed"] == 0
    assert data["summary"]["warnings"] is not None
    assert "Duplicate article_id detected." in data["summary"]["warnings"]

def test_batch_summary_arithmetic():
    articles = [
        {"article_id": "id_0", "text": "Valid text 1"},
        {"article_id": "id_1", "text": "http://only-url.com"}, # Invalid after cleaning
        {"article_id": "id_2", "text": "Valid text 2"},
    ]
    response = client.post("/predict/batch", json={"articles": articles})
    assert response.status_code == 200
    data = response.json()
    
    total_processed = data["summary"]["total_processed"]
    total_failed = data["summary"]["total_failed"]
    
    actual_processed = sum(1 for r in data["results"] if r["predicted_label"] is not None)
    actual_failed = sum(1 for r in data["results"] if r["predicted_label"] is None)
    
    assert total_processed == actual_processed
    assert total_failed == actual_failed
    assert total_processed + total_failed == len(articles)
