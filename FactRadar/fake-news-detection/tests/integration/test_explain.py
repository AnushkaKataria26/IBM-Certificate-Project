import time
from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

def test_explain_valid_request():
    payload = {
        "text": "This is a completely fabricated news article about aliens landing in New York and taking over the stock exchange."
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert "confidence" in data
    assert "top_contributing_tokens" in data
    
    tokens = data["top_contributing_tokens"]
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # The default num_features is 10, so it should be <= 10
    assert len(tokens) <= 10

def test_explain_empty_text():
    payload = {"text": "   "}
    response = client.post("/explain", json=payload)
    assert response.status_code == 422
    # Pydantic validation fails first
    assert "must contain at least one non-whitespace character" in response.text

def test_explain_text_cleans_to_empty():
    payload = {"text": "http://example.com/only-urls ?!"}
    response = client.post("/explain", json=payload)
    assert response.status_code == 422
    assert "no usable text content after preprocessing" in response.text

def test_explain_short_text_warning():
    # Less than 3 tokens
    payload = {"text": "Aliens landed."}
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["warning"] == "low_confidence_ood"
    
    tokens = data["top_contributing_tokens"]
    assert isinstance(tokens, list)
    # The length could be smaller than 10 because there are only 2 tokens
    assert len(tokens) <= 10

def test_predict_latency_unaffected():
    payload = {
        "text": "The government announced a new economic policy today aimed at reducing inflation over the next quarter."
    }
    
    # Warm up
    client.post("/predict", json=payload)
    
    start_time = time.time()
    n_requests = 100
    for _ in range(n_requests):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / n_requests
    # Let's just assert it's reasonably fast (e.g., < 50ms per request on average)
    assert avg_latency < 0.05, f"Predict latency too high: {avg_latency*1000:.2f}ms"
