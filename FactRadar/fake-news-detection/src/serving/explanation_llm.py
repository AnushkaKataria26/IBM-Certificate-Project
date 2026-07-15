"""
MANDATORY FIRST STEP FINDINGS:
1. Exact function producing a prediction: `unified_predict_proba` inside `src/serving/app.py`, which is called by `/predict` and `/explain` endpoints. The `/explain` endpoint calculates `predicted_label` and `confidence` from these probabilities.
2. Exact function producing token attributions: `explain_instance` in `src.serving.explain_lime`, which returns a list of (word, weight) tuples.
3. MLflow run context at inference: There is NO active MLflow run context at inference time (app.py only loads the model and sets tracking URI, it does not call mlflow.start_run() during requests). Inferences are logged to an SQLite DB via `log_prediction` in `src.serving.log_inference`. Thus, MLflow logging will be skipped if no active run is detected.
4. Entry point for end-to-end classification: The `/explain` route in `src/serving/app.py` (FastAPI).

Expected `prediction_result` keys: "predicted_label" (str) and "confidence" (float).
"""

import time
import mlflow
from typing import Optional, Tuple, Dict, Any, List

class ExplanationModelLoadError(Exception):
    """Custom exception raised when the explanation model fails to load."""
    pass

# Module-level cache
_model_cache: Dict[str, Tuple[Any, Any]] = {}

def load_explanation_model(
    model_name: str = "google/flan-t5-base",
    device: str = "cpu"
) -> tuple:
    """
    Loads and caches tokenizer + model. Must only load once per process 
    (module-level cache keyed by model_name). Returns (tokenizer, model).
    Raises ExplanationModelLoadError (custom exception, define in this file) 
    if the model cannot be loaded or downloaded.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
        
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        _model_cache[model_name] = (tokenizer, model)
        return tokenizer, model
    except Exception as e:
        raise ExplanationModelLoadError(f"Failed to load explanation model {model_name}: {str(e)}")

def build_explanation_prompt(
    article_text: str,
    predicted_label: str,
    confidence: float,
    top_tokens: list[tuple[str, float]],
    max_article_chars: int = 1500
) -> str:
    """
    Constructs the text-to-text prompt fed to the model.
    """
    if len(article_text) > max_article_chars:
        truncated_text = article_text[:max_article_chars] + "...[truncated]"
    else:
        truncated_text = article_text

    confidence_pct = confidence * 100.0

    prompt = (
        "Instructions: Write a one-sentence explanation of why the article was classified based on the key words.\n"
        f"Article:\n{truncated_text}\n"
        f"Classification: {predicted_label} (Confidence: {confidence_pct:.1f}%)\n"
    )
    
    if top_tokens:
        sorted_tokens = sorted(top_tokens, key=lambda x: abs(x[1]), reverse=True)[:10]
        prompt += f"Key words: {', '.join([t[0] for t in sorted_tokens])}\n"
    else:
        prompt += "Key words: None (No token-level attribution was available.)\n"
        
    prompt += "Explanation:"

    return prompt

def generate_template_fallback(
    predicted_label: str,
    confidence: float,
    top_tokens: list[tuple[str, float]]
) -> str:
    """
    Deterministic, non-LLM template sentence generator used when the model 
    path fails.
    """
    confidence_pct = confidence * 100.0
    base_sentence = f"The article was classified as {predicted_label} with {confidence_pct:.1f}% confidence"
    
    if top_tokens:
        sorted_tokens = sorted(top_tokens, key=lambda x: abs(x[1]), reverse=True)[:3]
        token_strs = [f"'{t[0]}'" for t in sorted_tokens]
        if len(token_strs) == 1:
            token_list_str = token_strs[0]
        elif len(token_strs) == 2:
            token_list_str = f"{token_strs[0]} and {token_strs[1]}"
        else:
            token_list_str = ", ".join(token_strs[:-1]) + f", and {token_strs[-1]}"
        return f"{base_sentence}, primarily driven by the presence of terms such as {token_list_str}."
    else:
        return f"{base_sentence}, based on the overall linguistic patterns in the text."

def generate_explanation(
    article_text: str,
    predicted_label: str,
    confidence: float,
    top_tokens: list[tuple[str, float]],
    tokenizer=None,
    model=None,
    max_new_tokens: int = 128,
    timeout_seconds: float = 30.0
) -> dict:
    """
    Orchestrates prompt construction + generation + fallback.
    """
    import threading
    
    start_time = time.time()
    truncated = len(article_text) > 1500
    
    fallback_used = False
    model_used = "google/flan-t5-base"  # default
    explanation = ""
    
    try:
        if tokenizer is None or model is None:
            tokenizer, model = load_explanation_model()
        if hasattr(model, 'name_or_path'):
            model_used = model.name_or_path
    except ExplanationModelLoadError:
        fallback_used = True
        model_used = "template_fallback"
        explanation = generate_template_fallback(predicted_label, confidence, top_tokens)
        
    if not fallback_used:
        prompt = build_explanation_prompt(article_text, predicted_label, confidence, top_tokens)
        
        result_container = {}
        def _generate():
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.2,
                    do_sample=True,
                    temperature=0.7
                )
                raw_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
                result_container["text"] = raw_out
            except Exception as e:
                result_container["error"] = e

        thread = threading.Thread(target=_generate)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Timeout
            fallback_used = True
            model_used = "template_fallback"
            explanation = generate_template_fallback(predicted_label, confidence, top_tokens)
        elif "error" in result_container:
            # Re-raise runtime errors during generation
            raise result_container["error"]
        else:
            text = result_container.get("text", "")
            if not text.strip() or len(text) < 10:
                fallback_used = True
                model_used = "template_fallback"
                explanation = generate_template_fallback(predicted_label, confidence, top_tokens)
            # Prevent the LLM from simply regurgitating the article text (extractive summarization)
            elif text.strip().lower() in article_text.lower():
                fallback_used = True
                model_used = "template_fallback"
                explanation = generate_template_fallback(predicted_label, confidence, top_tokens)
            else:
                explanation = text

    generation_time_ms = (time.time() - start_time) * 1000.0

    return {
        "explanation": explanation,
        "model_used": model_used,
        "fallback_used": fallback_used,
        "truncated": truncated,
        "generation_time_ms": generation_time_ms
    }

def attach_explanation_to_prediction(
    prediction_result: dict,
    article_text: str,
    top_tokens: list[tuple[str, float]],
    tokenizer=None,
    model=None
) -> dict:
    """
    Takes the existing prediction_result dict and attaches an explanation.
    """
    if "predicted_label" not in prediction_result:
        raise KeyError("prediction_result is missing 'predicted_label' field")
    if "confidence" not in prediction_result:
        raise KeyError("prediction_result is missing 'confidence' field")
        
    predicted_label = prediction_result["predicted_label"]
    confidence = prediction_result["confidence"]
    
    explanation_dict = generate_explanation(
        article_text=article_text,
        predicted_label=predicted_label,
        confidence=confidence,
        top_tokens=top_tokens,
        tokenizer=tokenizer,
        model=model
    )
    
    if mlflow.active_run() is not None:
        try:
            mlflow.log_param("explanation_model_used", explanation_dict["model_used"])
            mlflow.log_param("explanation_fallback_used", explanation_dict["fallback_used"])
            mlflow.log_text(explanation_dict["explanation"], "explanation.txt")
        except Exception:
            pass
            
    new_result = prediction_result.copy()
    new_result["explanation"] = explanation_dict
    
    return new_result
