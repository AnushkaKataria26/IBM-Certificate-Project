"""
app.py — FastAPI serving layer for the fake news detection model.

Loads the trained sklearn pipeline (TF-IDF + LogisticRegression) at module
level and exposes /predict, /health, and /model/version endpoints.

CRITICAL DESIGN DECISION — No train/serve skew:
    Training (train_baseline.py) fitted the pipeline on df_train["clean_text"],
    which is the text column after applying clean_text(). This serving layer
    reuses the exact same clean_text function from src.preprocessing.clean_text.
    The title field is accepted for forward compatibility but NOT concatenated
    with text — the v0.1 baseline was trained on text-only.
"""

import json
import logging
import traceback
from pathlib import Path

import joblib
import asyncio
from fastapi import FastAPI, HTTPException

from src.preprocessing.clean_text import clean_text
from src.serving.schemas import (
    HealthResponse,
    ModelVersionResponse,
    PredictRequest,
    PredictResponse,
    ExplainResponse,
    TokenWeight,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictResult,
    BatchPredictSummary,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths (relative to project root, where uvicorn is launched)
# ---------------------------------------------------------------------------
_MODEL_PATH = Path("models/v0.1_baseline.joblib")
_METRICS_PATH = Path("models/v0.1_baseline_metrics.json")
_MODEL_VERSION = "v0.1_baseline"

# ---------------------------------------------------------------------------
# Minimum token count for OOD warning (heuristic, not a hard block)
# ---------------------------------------------------------------------------
_MIN_TOKEN_COUNT = 3

# ---------------------------------------------------------------------------
# Load model and metrics at module level (once, not per request)
# ---------------------------------------------------------------------------
import mlflow
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")

# ---------------------------------------------------------------------------
# Load model from MLflow Registry
# ---------------------------------------------------------------------------
_pipeline = None
_metrics = None
model_loaded = False
_mtype = "sklearn"

def load_model():
    global _pipeline, _metrics, model_loaded, _mtype, _MODEL_VERSION
    client = mlflow.tracking.MlflowClient()
    try:
        model_name = "sourcetrace-classifier"
        mv = client.get_model_version_by_alias(model_name, "Production")
        model_uri = f"models:/{model_name}@Production"
        
        # Determine model flavor
        run = client.get_run(mv.run_id)
        if "transformer" in run.data.tags.get("mlflow.runName", "") or "transformer" in mv.source:
            _pipeline = mlflow.transformers.load_model(model_uri)
            _mtype = "transformers"
        else:
            _pipeline = mlflow.sklearn.load_model(model_uri)
            _mtype = "sklearn"
            
        _MODEL_VERSION = f"{model_name}-v{mv.version}"
        model_loaded = True
        logger.info(f"Loaded Production model {_MODEL_VERSION} ({_mtype}) from MLflow registry.")
        
        # Load metrics dynamically from run if possible
        _metrics = run.data.metrics
        _metrics["timestamp"] = run.info.start_time
    except Exception as exc:
        logger.warning(f"Failed to load from MLflow registry: {exc}. Falling back to local joblib.")
        try:
            _pipeline = joblib.load(_MODEL_PATH)
            _mtype = "sklearn"
            model_loaded = True
            with open(_METRICS_PATH, "r") as f:
                _metrics = json.load(f)
            logger.info("Fallback loaded successfully from %s", _MODEL_PATH)
        except Exception as e:
            logger.error(f"Fallback loading failed: {e}")
            model_loaded = False

load_model()

def unified_predict_proba(texts):
    """Wrapper to provide predict_proba interface for LIME, regardless of underlying model type."""
    if _mtype == "sklearn":
        return _pipeline.predict_proba(texts)
    else:
        # Transformers pipeline
        import numpy as np
        res = _pipeline(texts, truncation=True, max_length=512, top_k=None)
        probs = []
        for r in res:
            p0, p1 = 0.0, 0.0
            for score_dict in r:
                lbl = str(score_dict['label'])
                if lbl in ['LABEL_0', '0', 'real']:
                    p0 = score_dict['score']
                else:
                    p1 = score_dict['score']
            probs.append([p0, p1])
        return np.array(probs)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FactRadar — Fake News Detection API",
    version=_MODEL_VERSION,
    description="Serves the baseline TF-IDF + Logistic Regression model.",
)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single article as fake or real.

    Edge cases handled:
    - Model not loaded → 503
    - Cleaned text is empty (input was purely URLs/punctuation) → 422
    - Cleaned text has < 3 tokens → 200 with warning="low_confidence_ood"
    - Unexpected exception → 500 (logged server-side, generic message returned)
    """
    # 1. Model availability check
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is degraded.",
        )

    try:
        # 2. Apply the SAME clean_text used at training time.
        #    Use text field only — v0.1 baseline was trained on text-only.
        cleaned = clean_text(request.text)

        # 3. Edge case: input cleaned to empty string
        if not cleaned:
            raise HTTPException(
                status_code=422,
                detail="Input contains no usable text content after preprocessing.",
            )

        # 4. OOD warning for very short inputs
        warning = None
        token_count = len(cleaned.split())
        if token_count < _MIN_TOKEN_COUNT:
            warning = "low_confidence_ood"

        # 5. Run prediction
        from src.serving.log_inference import log_prediction
        import time
        start_time = time.time()
        
        probabilities = unified_predict_proba([cleaned])[0]
        predicted_class_idx = probabilities.argmax()
        confidence = float(probabilities[predicted_class_idx])
        predicted_class = int(predicted_class_idx)

        # Map numeric label to string: 0 → "real", 1 → "fake"
        predicted_label = "fake" if predicted_class == 1 else "real"
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log to DB
        log_prediction(_MODEL_VERSION, request.text, predicted_label, confidence, latency_ms)

        return PredictResponse(
            predicted_label=predicted_label,
            confidence=round(confidence, 6),
            model_version=_MODEL_VERSION,
            warning=warning,
        )

    except HTTPException:
        # Re-raise HTTP exceptions (422 from empty-after-clean check)
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error during prediction: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction.",
        )


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------
@app.post("/explain", response_model=ExplainResponse)
async def explain(request: PredictRequest) -> ExplainResponse:
    """Explain a single article's prediction using LIME.
    
    Returns HTTP 504 if the explanation takes longer than 10 seconds.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is degraded.",
        )
        
    try:
        from src.preprocessing.clean_text import clean_text
        cleaned = clean_text(request.text)
        if not cleaned:
            raise HTTPException(
                status_code=422,
                detail="Input contains no usable text content after preprocessing.",
            )
            
        warning = None
        token_count = len(cleaned.split())
        if token_count < _MIN_TOKEN_COUNT:
            warning = "low_confidence_ood"
            
        probabilities = unified_predict_proba([cleaned])[0]
        predicted_class_idx = probabilities.argmax()
        confidence = float(probabilities[predicted_class_idx])
        predicted_class = int(predicted_class_idx)
        predicted_label = "fake" if predicted_class == 1 else "real"
        
        from src.serving.explain_lime import explain_instance
        
        try:
            explanation_list = await asyncio.wait_for(
                asyncio.to_thread(explain_instance, request.text, unified_predict_proba, 10),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Explanation computation exceeded the 10 second timeout."
            )
            
        top_tokens = [TokenWeight(token=k, weight=v) for k, v in explanation_list]
        
        return ExplainResponse(
            predicted_label=predicted_label,
            confidence=round(confidence, 6),
            top_contributing_tokens=top_tokens,
            model_version=_MODEL_VERSION,
            warning=warning,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error during explanation: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during explanation computation.",
        )


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------
@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify a batch of articles as fake or real.

    Edge cases handled:
    - Model not loaded → 503
    - Catastrophic failure → 503
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is degraded.",
        )

    try:
        import time
        from src.serving.log_inference import log_prediction
        start_time = time.time()
        
        warnings_summary = set()
        seen_ids = set()
        
        # Prepare structures
        valid_indices = []
        texts_to_predict = []
        results = [None] * len(request.articles)
        
        for i, article in enumerate(request.articles):
            if article.article_id in seen_ids:
                warnings_summary.add("Duplicate article_id detected.")
            seen_ids.add(article.article_id)
            
            cleaned = clean_text(article.text)
            if not cleaned:
                results[i] = BatchPredictResult(
                    article_id=article.article_id,
                    predicted_label=None,
                    confidence=None,
                    warning="invalid_input"
                )
            else:
                warning = None
                if len(cleaned.split()) < _MIN_TOKEN_COUNT:
                    warning = "low_confidence_ood"
                    
                valid_indices.append(i)
                texts_to_predict.append(cleaned)
                results[i] = warning  # temporarily store warning
                
        if texts_to_predict:
            probabilities = unified_predict_proba(texts_to_predict)
            
            for idx, batch_idx in enumerate(valid_indices):
                probs = probabilities[idx]
                predicted_class_idx = probs.argmax()
                confidence = float(probs[predicted_class_idx])
                predicted_label = "fake" if int(predicted_class_idx) == 1 else "real"
                
                article = request.articles[batch_idx]
                warning = results[batch_idx]  # retrieve stored warning
                
                results[batch_idx] = BatchPredictResult(
                    article_id=article.article_id,
                    predicted_label=predicted_label,
                    confidence=round(confidence, 6),
                    warning=warning
                )
                
                # Log to DB
                log_prediction(_MODEL_VERSION, article.text, predicted_label, confidence, 0.0)
                
        total_processed = len(valid_indices)
        total_failed = len(request.articles) - total_processed
        latency_ms = (time.time() - start_time) * 1000
        
        summary = BatchPredictSummary(
            total_processed=total_processed,
            total_failed=total_failed,
            processing_time_ms=round(latency_ms, 2),
            warnings=list(warnings_summary) if warnings_summary else None
        )
        
        return BatchPredictResponse(results=results, summary=summary)
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error during batch prediction: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=503,
            detail="Internal server error during batch prediction.",
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check reflecting whether the model loaded successfully."""
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
    )


# ---------------------------------------------------------------------------
# GET /model/version
# ---------------------------------------------------------------------------
@app.get("/model/version", response_model=ModelVersionResponse)
async def model_version() -> ModelVersionResponse:
    """Return model version and training metrics."""
    if _metrics is None:
        raise HTTPException(
            status_code=503,
            detail="Model metrics are not available. Service is degraded.",
        )

    return ModelVersionResponse(
        model_version=_MODEL_VERSION,
        trained_at=_metrics.get("timestamp", "unknown"),
        metrics={
            "accuracy": _metrics.get("accuracy"),
            "precision_macro": _metrics.get("precision_macro"),
            "recall_macro": _metrics.get("recall_macro"),
            "f1_macro": _metrics.get("f1_macro"),
            "roc_auc": _metrics.get("roc_auc"),
        },
    )
