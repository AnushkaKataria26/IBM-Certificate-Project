"""
schemas.py — Pydantic request/response models for the fake news detection API.

Uses Pydantic v2 field validators to enforce input constraints that
go beyond simple type annotations (e.g., rejecting whitespace-only strings).
"""

from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    """Schema for POST /predict request body.

    Attributes
    ----------
    title : str
        Optional article title. Default empty string. Not used by the
        current v0.1 baseline model (trained on text column only), but
        included for forward compatibility with future models.
    text : str
        Required article body text. Must contain at least one non-whitespace
        character — enforced via a Pydantic validator, not just a type
        annotation, since a whitespace-only string passes a naive str check.
    """

    title: str = ""
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        """Reject empty or whitespace-only text."""
        if not v.strip():
            raise ValueError(
                "text field must contain at least one non-whitespace character"
            )
        return v


class PredictResponse(BaseModel):
    """Schema for POST /predict response body."""

    predicted_label: Literal["fake", "real"]
    confidence: float
    model_version: str
    warning: Optional[str] = None


class BatchPredictArticle(BaseModel):
    """Individual article within a batch request."""
    article_id: str
    title: Optional[str] = None
    text: str


class BatchPredictRequest(BaseModel):
    """Schema for POST /predict/batch request body."""
    articles: list[BatchPredictArticle]

    @field_validator("articles")
    @classmethod
    def validate_articles_length(cls, v: list[BatchPredictArticle]) -> list[BatchPredictArticle]:
        if not v:
            raise ValueError("articles list cannot be empty")
        if len(v) > 500:
            raise ValueError("batch size exceeds maximum limit of 500 articles")
        return v


class BatchPredictResult(BaseModel):
    """Prediction result for a single article in a batch."""
    article_id: str
    predicted_label: Optional[Literal["fake", "real"]] = None
    confidence: Optional[float] = None
    warning: Optional[str] = None


class BatchPredictSummary(BaseModel):
    """Summary statistics for a batch prediction."""
    total_processed: int
    total_failed: int
    processing_time_ms: float
    warnings: Optional[list[str]] = None


class BatchPredictResponse(BaseModel):
    """Schema for POST /predict/batch response body."""
    results: list[BatchPredictResult]
    summary: BatchPredictSummary


class HealthResponse(BaseModel):
    """Schema for GET /health response body."""

    status: str
    model_loaded: bool


class ModelVersionResponse(BaseModel):
    """Schema for GET /model/version response body."""

    model_version: str
    trained_at: str
    metrics: dict


class TokenWeight(BaseModel):
    token: str
    weight: float


class ExplainResponse(BaseModel):
    """Schema for POST /explain response body."""

    predicted_label: Literal["fake", "real"]
    confidence: float
    top_contributing_tokens: list[TokenWeight]
    model_version: str
    warning: Optional[str] = None


class ExplanationDict(BaseModel):
    """Schema for LLM explanation block."""
    explanation: str
    model_used: str
    fallback_used: bool
    truncated: bool
    generation_time_ms: float


class VerificationResponse(BaseModel):
    """Schema for RAG verification block."""
    activated: bool
    reason: Optional[str] = None
    verdict: Optional[str] = None
    justification: Optional[str] = None
    evidence_count: Optional[int] = None
    evidence_ids: Optional[list[str]] = None
    parse_successful: Optional[bool] = None
    recommend_review: Optional[bool] = None


class AnalyzeRequest(BaseModel):
    """Schema for POST /analyze request body."""
    article_id: str
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        """Reject empty or whitespace-only text."""
        if not v.strip():
            raise ValueError(
                "text field must contain at least one non-whitespace character"
            )
        return v


class AnalyzeResponse(BaseModel):
    """Schema for POST /analyze response body."""
    article_id: str
    predicted_label: Literal["fake", "real"]
    confidence: float
    top_contributing_tokens: list[TokenWeight]
    explanation: ExplanationDict
    verification: Optional[VerificationResponse] = None
    model_version: str
    warning: Optional[str] = None

