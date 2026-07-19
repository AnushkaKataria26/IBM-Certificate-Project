"""
MANDATORY FIRST STEP FINDINGS:
1. Location and loading mechanism of the labeled training corpus: 
   File path: `data/splits/train.csv`. 
   Loaded using `pandas.read_csv()`.
   Fields: `clean_text` for article text, `label` for label.
2. Exact schema of `prediction_result`: 
   A dictionary expected to have `predicted_label` (str, "fake" or "real") and `confidence` (float).
3. Reuse of `explanation_llm.py`: 
   `src/serving/explanation_llm.py` is present. `load_explanation_model` caches and returns `(tokenizer, model)`. This is reusable to avoid loading a second model into memory.
4. MLflow logging pattern: 
   We must check `if mlflow.active_run() is not None:` and then log using `mlflow.log_param` and `mlflow.log_text` inside a try-except block to gracefully handle failures.

NOTE ON RETRIEVAL:
`rank-bm25` was confirmed absent from the environment.
Falling back to TF-IDF + cosine similarity using `scikit-learn` as per constraints.
"""
import re
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _tokenize(text: str) -> list[str]:
    """
    Tokenization for retrieval: lowercases and splits on whitespace after basic 
    punctuation stripping.
    """
    if not text:
        return []
    # Strip basic punctuation and lowercase
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    return cleaned.split()

def build_reference_index(
    corpus_texts: list[str],
    corpus_labels: list[str],
    corpus_ids: list[str] = None
) -> dict:
    """
    Builds and returns an in-memory TF-IDF index plus parallel metadata (fallback for BM25).
    Returns dict with keys: "vectorizer", "tfidf_matrix", "texts" (list[str], 
    unmodified), "labels" (list[str]), "ids" (list[str] — generated as 
    str(index) if corpus_ids is None).
    Raises ValueError if len(corpus_texts) != len(corpus_labels), or if 
    corpus_texts is empty.
    Tokenization lowercases and splits on whitespace after basic punctuation stripping.
    """
    if not corpus_texts:
        raise ValueError("corpus_texts is empty")
    if len(corpus_texts) != len(corpus_labels):
        raise ValueError("len(corpus_texts) != len(corpus_labels)")
    
    if corpus_ids is None:
        corpus_ids = [str(i) for i in range(len(corpus_texts))]
    
    vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
    except ValueError:
        tfidf_matrix = None
        
    return {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "texts": corpus_texts,
        "labels": corpus_labels,
        "ids": corpus_ids
    }

def retrieve_evidence(
    index: dict,
    query_text: str,
    top_k: int = 5,
    exclude_id: str = None
) -> list[dict]:
    """
    Queries the TF-IDF index built by build_reference_index. Returns a list of up 
    to top_k dicts, each with keys: "id", "text", "label", "score" (float cosine 
    score), sorted descending by score. If exclude_id is provided, that id is 
    excluded from results. If query_text is empty or produces zero 
    matching tokens, returns an empty list (does not raise).
    """
    if not query_text:
        return []
        
    tokens = _tokenize(query_text)
    if not tokens:
        return []
        
    vectorizer = index.get("vectorizer")
    tfidf_matrix = index.get("tfidf_matrix")
    
    if vectorizer is None or tfidf_matrix is None:
        return []
        
    try:
        query_vec = vectorizer.transform([query_text])
    except ValueError:
        return []
        
    if query_vec.nnz == 0:
        return []
        
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sorted_indices = np.argsort(cosine_scores)[::-1]
    
    results = []
    for idx in sorted_indices:
        score = float(cosine_scores[idx])
        if score == 0.0:
            break
            
        doc_id = index["ids"][idx]
        if exclude_id is not None and str(doc_id) == str(exclude_id):
            continue
            
        results.append({
            "id": doc_id,
            "text": index["texts"][idx],
            "label": index["labels"][idx],
            "score": score
        })
        
        if len(results) >= top_k:
            break
            
    return results

def build_verification_prompt(
    article_text: str,
    predicted_label: str,
    confidence: float,
    evidence: list[dict],
    max_article_chars: int = 1200,
    max_evidence_chars_each: int = 500
) -> str:
    """
    Constructs the prompt asking the LLM to assess whether the retrieved 
    evidence supports or contradicts the flagged article.
    """
    if len(article_text) > max_article_chars:
        trunc_article = article_text[:max_article_chars] + "...[truncated]"
    else:
        trunc_article = article_text
        
    prompt = (
        "Instructions: Read the article and evidence. Decide if the evidence supports ('consistent'), contradicts ('contradictory'), or is not enough ('insufficient_evidence') for the article.\n"
        f"Article:\n{trunc_article}\n"
    )
    
    if not evidence:
        prompt += "Evidence: No reference evidence was found.\n"
    else:
        prompt += "Evidence:\n"
        for i, ev in enumerate(evidence):
            ev_text = ev["text"]
            if len(ev_text) > max_evidence_chars_each:
                ev_text = ev_text[:max_evidence_chars_each] + "...[truncated]"
            prompt += f"- {ev_text}\n"
        prompt += "\n"
        
    prompt += "Verdict (choose one: consistent, contradictory, insufficient_evidence):"
    
    return prompt

def parse_verification_response(raw_output: str) -> dict:
    """
    Parses the LLM's raw text output into a structured verdict. Returns dict 
    with keys: "verdict", "justification", "parse_successful".
    Matching is case-insensitive substring matching against the three 
    allowed verdict strings. If multiple are present, first by string index wins.
    """
    if not raw_output:
        return {
            "verdict": "unparseable",
            "justification": "",
            "parse_successful": False
        }
        
    lower_out = raw_output.lower()
    verdicts = ["consistent", "contradictory", "insufficient_evidence"]
    
    found_verdict = None
    found_idx = -1
    
    for v in verdicts:
        idx = lower_out.find(v)
        if idx != -1:
            if found_idx == -1 or idx < found_idx:
                found_idx = idx
                found_verdict = v
                
    if not found_verdict:
        return {
            "verdict": "unparseable",
            "justification": raw_output,
            "parse_successful": False
        }
        
    justification = re.sub(found_verdict, "", raw_output, count=1, flags=re.IGNORECASE).strip()
    justification = re.sub(r'^[:\-\.,\s]+', '', justification).strip()
    
    return {
        "verdict": found_verdict,
        "justification": justification,
        "parse_successful": True
    }

def run_verification_stage(
    prediction_result: dict,
    article_text: str,
    article_id: str,
    reference_index: dict,
    confidence_low: float = 0.50,
    confidence_high: float = 1.0,
    top_k_evidence: int = 5,
    tokenizer=None,
    model=None
) -> dict:
    """
    Orchestrates the full advisory stage.
    """
    if confidence_low > confidence_high:
        raise ValueError("confidence_low > confidence_high")
        
    pred_label = prediction_result.get("predicted_label")
    confidence = prediction_result.get("confidence", 0.0)
    
    if not (confidence_low <= confidence <= confidence_high):
        new_res = prediction_result.copy()
        new_res["verification"] = {
            "activated": False,
            "reason": "outside_trigger_band"
        }
        return new_res
        
    try:
        evidence = retrieve_evidence(reference_index, article_text, top_k=top_k_evidence, exclude_id=article_id)
        prompt = build_verification_prompt(article_text, pred_label, confidence, evidence)
        
        if tokenizer is None or model is None:
            from src.serving.explanation_llm import load_explanation_model
            tokenizer, model = load_explanation_model()
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=32,
            repetition_penalty=1.2,
            do_sample=False
        )
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        parsed = parse_verification_response(raw_output)
        verdict = parsed["verdict"]
        recommend_review = (verdict == "contradictory" or verdict == "unparseable")
        
        verification_block = {
            "activated": True,
            "verdict": verdict,
            "justification": parsed["justification"],
            "evidence_count": len(evidence),
            "evidence_ids": [str(e["id"]) for e in evidence],
            "parse_successful": parsed["parse_successful"],
            "recommend_review": recommend_review
        }
        
    except Exception:
        evidence_count = 0
        evidence_ids = []
        if 'evidence' in locals():
            evidence_count = len(evidence)
            evidence_ids = [str(e["id"]) for e in evidence]
            
        verification_block = {
            "activated": True,
            "verdict": "insufficient_evidence",
            "justification": "verification stage encountered an error and could not complete",
            "evidence_count": evidence_count,
            "evidence_ids": evidence_ids,
            "parse_successful": False,
            "recommend_review": True
        }
        
    new_res = prediction_result.copy()
    new_res["verification"] = verification_block
    
    if mlflow.active_run() is not None:
        try:
            mlflow.log_param("verification_verdict", verification_block["verdict"])
            mlflow.log_param("verification_recommend_review", verification_block["recommend_review"])
            mlflow.log_text(verification_block["justification"], "verification_justification.txt")
        except Exception:
            pass
            
    return new_res
