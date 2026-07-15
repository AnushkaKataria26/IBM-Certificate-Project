"""
test_api.py — Integration tests for the fake news detection API.

Uses FastAPI's TestClient (backed by httpx) to exercise all endpoints
and edge cases without starting a real server process.
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that persists across all tests in this module."""
    with TestClient(app) as c:
        yield c


# ===================================================================
# POST /predict — Happy path
# ===================================================================
class TestPredictValid:
    """Valid prediction requests should return 200 with correct schema."""

    def test_valid_prediction(self, client: TestClient):
        """A normal article text should produce a well-formed response."""
        response = client.post(
            "/predict",
            json={
                "text": "The president announced new economic policies today "
                "in a press conference at the White House."
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_label"] in ("fake", "real")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["model_version"] == "sourcetrace-classifier-v3"
        # Normal-length text should NOT have a warning
        assert data["warning"] is None

    def test_valid_prediction_with_title(self, client: TestClient):
        """Title is accepted but should not break the prediction."""
        response = client.post(
            "/predict",
            json={
                "title": "Breaking News",
                "text": "Scientists discover new species in the Amazon rainforest "
                "during a large-scale biodiversity survey.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_label"] in ("fake", "real")


# ===================================================================
# POST /predict — Edge case: empty / whitespace text
# ===================================================================
class TestPredictEmptyText:
    """Empty or whitespace-only text must be caught by Pydantic validator."""

    def test_empty_text(self, client: TestClient):
        """Empty string text should return 422."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_whitespace_only_text(self, client: TestClient):
        """Whitespace-only text should return 422."""
        response = client.post("/predict", json={"text": "   \t\n  "})
        assert response.status_code == 422


# ===================================================================
# POST /predict — Edge case: text that cleans to empty string
# ===================================================================
class TestPredictCleansToEmpty:
    """Input that is non-empty but purely URLs/punctuation should return 422
    with the specific message from the endpoint, not a generic error."""

    def test_urls_and_punctuation(self, client: TestClient):
        """Text of only URLs and punctuation cleans to empty → 422."""
        response = client.post(
            "/predict",
            json={"text": "!!!! http://example.com ????"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "no usable text content after preprocessing" in data["detail"].lower()

    def test_only_urls(self, client: TestClient):
        """Text of only URLs cleans to empty → 422."""
        response = client.post(
            "/predict",
            json={"text": "https://foo.com https://bar.org"},
        )
        assert response.status_code == 422


# ===================================================================
# POST /predict — Edge case: very short text (OOD warning)
# ===================================================================
class TestPredictShortText:
    """Very short text should return 200 but with a warning flag."""

    def test_short_text_ood_warning(self, client: TestClient):
        """Input with fewer than 3 tokens after cleaning → warning populated."""
        response = client.post("/predict", json={"text": "ok"})
        assert response.status_code == 200
        data = response.json()
        assert data["warning"] == "low_confidence_ood"
        # Should still have a valid prediction
        assert data["predicted_label"] in ("fake", "real")
        assert 0.0 <= data["confidence"] <= 1.0


# ===================================================================
# POST /predict — Malformed request
# ===================================================================
class TestPredictMalformed:
    """Requests that don't match PredictRequest schema → 422."""

    def test_missing_text_field(self, client: TestClient):
        """Missing required 'text' field should return 422."""
        response = client.post("/predict", json={"title": "just a title"})
        assert response.status_code == 422

    def test_completely_wrong_body(self, client: TestClient):
        """Completely wrong JSON body should return 422."""
        response = client.post("/predict", json={"foo": "bar"})
        assert response.status_code == 422

    def test_no_json_body(self, client: TestClient):
        """No JSON body at all should return 422."""
        response = client.post("/predict")
        assert response.status_code == 422


# ===================================================================
# GET /health
# ===================================================================
class TestHealth:
    """Health endpoint should reflect actual model state."""

    def test_health_ok(self, client: TestClient):
        """Under normal conditions, model should be loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


# ===================================================================
# GET /model/version
# ===================================================================
class TestModelVersion:
    """Model version endpoint should return populated fields."""

    def test_model_version(self, client: TestClient):
        """Should return model version, timestamp, and metrics."""
        response = client.get("/model/version")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "sourcetrace-classifier-v3"
        assert data["trained_at"] is not None
        assert data["trained_at"] != ""
        assert isinstance(data["metrics"], dict)
        assert "accuracy" in data["metrics"]
        assert "f1_macro" in data["metrics"]
