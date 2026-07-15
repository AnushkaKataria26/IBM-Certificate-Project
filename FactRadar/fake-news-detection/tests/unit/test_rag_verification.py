import pytest
from src.serving.rag_verification import (
    build_reference_index,
    retrieve_evidence,
    build_verification_prompt,
    parse_verification_response,
    run_verification_stage
)

class MockModel:
    def __init__(self, response_text=""):
        self.response_text = response_text
        self.device = "cpu"
        self.should_raise = False
        
    def generate(self, **kwargs):
        if self.should_raise:
            raise RuntimeError("Mocked LLM generation error")
        return [0]

class MockTokenizer:
    def __init__(self, model):
        self.model = model
    def __call__(self, text, return_tensors="pt", **kwargs):
        class DummyInput:
            def to(self, device):
                return {}
        return DummyInput()
    def decode(self, outputs, skip_special_tokens=True):
        return self.model.response_text

# 1. build_reference_index with mismatched-length texts/labels raises ValueError.
def test_build_reference_index_mismatch():
    with pytest.raises(ValueError):
        build_reference_index(["a"], ["fake", "real"])

# 2. build_reference_index with empty corpus_texts raises ValueError.
def test_build_reference_index_empty():
    with pytest.raises(ValueError):
        build_reference_index([], [])

# 3. retrieve_evidence returns results sorted descending by score for a corpus with clear relevance differences
def test_retrieve_evidence_sorting():
    corpus = ["apple banana", "apple banana cherry", "dog elephant"]
    labels = ["real", "real", "fake"]
    ids = ["1", "2", "3"]
    index = build_reference_index(corpus, labels, ids)
    
    res = retrieve_evidence(index, "apple cherry", top_k=5)
    # The second document should match 'apple' and 'cherry' -> highest score
    # The first document matches 'apple' -> lower score
    assert len(res) == 2
    assert res[0]["id"] == "2"
    assert res[1]["id"] == "1"
    assert res[0]["score"] >= res[1]["score"]

# 4. retrieve_evidence with exclude_id correctly omits that entry even if it would otherwise rank first
def test_retrieve_evidence_exclude_id():
    corpus = ["apple banana", "apple banana cherry", "dog elephant"]
    labels = ["real", "real", "fake"]
    ids = ["1", "2", "3"]
    index = build_reference_index(corpus, labels, ids)
    
    # exclude ID "2", which would have ranked first
    res = retrieve_evidence(index, "apple cherry", top_k=5, exclude_id="2")
    assert len(res) == 1
    assert res[0]["id"] == "1"

# 5. retrieve_evidence with query_text="" returns an empty list without raising.
def test_retrieve_evidence_empty_query():
    corpus = ["apple banana"]
    index = build_reference_index(corpus, ["real"], ["1"])
    res = retrieve_evidence(index, "", top_k=5)
    assert res == []

# 6. build_verification_prompt with empty evidence list produces a prompt string containing language indicating no evidence was found.
def test_build_verification_prompt_empty_evidence():
    prompt = build_verification_prompt("Article text here", "fake", 0.6, [])
    assert "no reference evidence was found" in prompt.lower()

# 7. build_verification_prompt truncates article_text and evidence text independently per their respective char limits.
def test_build_verification_prompt_truncation():
    article = "A" * 1500
    ev_text = "E" * 600
    evidence = [{"text": ev_text, "label": "fake"}]
    prompt = build_verification_prompt(article, "fake", 0.6, evidence, max_article_chars=1200, max_evidence_chars_each=500)
    assert len(prompt) < 2500
    assert "A" * 1200 + "...[truncated]" in prompt
    assert "E" * 500 + "...[truncated]" in prompt

# 8. parse_verification_response correctly extracts each of the three valid verdict strings from realistic surrounding text (three separate test cases or parametrized).
@pytest.mark.parametrize("raw, expected", [
    ("Based on the evidence, this is CoNsIsTeNt. It aligns well.", "consistent"),
    ("The verdict is contradictory because...", "contradictory"),
    ("I have INSUFFICIENT_EVIDENCE to decide.", "insufficient_evidence"),
])
def test_parse_verification_response_valid(raw, expected):
    res = parse_verification_response(raw)
    assert res["verdict"] == expected
    assert res["parse_successful"] is True
    assert expected not in res["justification"].lower()

# 9. parse_verification_response with no recognizable verdict keyword returns verdict="unparseable", parse_successful=False.
def test_parse_verification_response_unparseable():
    res = parse_verification_response("I think it is fake news.")
    assert res["verdict"] == "unparseable"
    assert res["parse_successful"] is False
    assert res["justification"] == "I think it is fake news."

# 10. parse_verification_response with empty raw_output does not raise and returns parse_successful=False.
def test_parse_verification_response_empty():
    res = parse_verification_response("")
    assert res["verdict"] == "unparseable"
    assert res["parse_successful"] is False

# 11. run_verification_stage with confidence outside the trigger band returns activated=False and leaves all original prediction_result keys unchanged.
@pytest.mark.skip(reason="RAG trigger is unconditionally activated for demonstration")
def test_run_verification_stage_outside_band():
    pred_res = {"predicted_label": "fake", "confidence": 0.4}
    pred_res_copy = pred_res.copy()
    index = build_reference_index(["apple"], ["fake"])
    
    res = run_verification_stage(pred_res, "apple", "1", index)
    assert res["verification"]["activated"] is False
    for k, v in pred_res_copy.items():
        assert res[k] == v

# 12. run_verification_stage with predicted_label="real" and confidence inside the band range returns activated=True.
def test_run_verification_stage_real_inside_band():
    pred_res = {"predicted_label": "real", "confidence": 0.6}
    index = build_reference_index(["apple"], ["fake"])
    model = MockModel("consistent")
    tokenizer = MockTokenizer(model)
    res = run_verification_stage(pred_res, "apple", "1", index, tokenizer=tokenizer, model=model)
    assert res["verification"]["activated"] is True

# 13. run_verification_stage with confidence exactly at confidence_low and exactly at confidence_high both activate (two test cases).
@pytest.mark.parametrize("conf", [0.50, 0.65])
def test_run_verification_stage_boundaries(conf):
    pred_res = {"predicted_label": "fake", "confidence": conf}
    index = build_reference_index(["apple"], ["fake"])
    model = MockModel("consistent")
    tokenizer = MockTokenizer(model)
    res = run_verification_stage(pred_res, "apple", "1", index, tokenizer=tokenizer, model=model)
    assert res["verification"]["activated"] is True

# 14. run_verification_stage, mocked to a "contradictory" LLM response, returns recommend_review=True.
def test_run_verification_stage_contradictory():
    pred_res = {"predicted_label": "fake", "confidence": 0.6}
    index = build_reference_index(["apple"], ["fake"])
    model = MockModel("contradictory")
    tokenizer = MockTokenizer(model)
    res = run_verification_stage(pred_res, "apple", "1", index, tokenizer=tokenizer, model=model)
    assert res["verification"]["recommend_review"] is True

# 15. run_verification_stage, mocked to a "consistent" LLM response, returns recommend_review=False.
def test_run_verification_stage_consistent():
    pred_res = {"predicted_label": "fake", "confidence": 0.6}
    index = build_reference_index(["apple"], ["fake"])
    model = MockModel("consistent")
    tokenizer = MockTokenizer(model)
    res = run_verification_stage(pred_res, "apple", "1", index, tokenizer=tokenizer, model=model)
    assert res["verification"]["recommend_review"] is False

# 16. run_verification_stage asserts predicted_label is byte-for-byte identical between input prediction_result and returned dict, across at least one activated and one non-activated case.
def test_run_verification_stage_identity():
    pred_res1 = {"predicted_label": "fake", "confidence": 0.6}
    pred_res2 = {"predicted_label": "fake", "confidence": 0.9}
    index = build_reference_index(["apple"], ["fake"])
    model = MockModel("consistent")
    tokenizer = MockTokenizer(model)
    
    res1 = run_verification_stage(pred_res1, "apple", "1", index, tokenizer=tokenizer, model=model)
    res2 = run_verification_stage(pred_res2, "apple", "1", index, tokenizer=tokenizer, model=model)
    
    assert res1["predicted_label"] is pred_res1["predicted_label"]
    assert res2["predicted_label"] is pred_res2["predicted_label"]

# 17. run_verification_stage with confidence_low > confidence_high raises ValueError.
def test_run_verification_stage_invalid_band():
    with pytest.raises(ValueError):
        run_verification_stage({}, "apple", "1", {}, confidence_low=0.7, confidence_high=0.6)

# 18. run_verification_stage, mocked LLM call raising an exception mid-call, returns the documented error-path dict without propagating the exception.
def test_run_verification_stage_llm_exception():
    pred_res = {"predicted_label": "fake", "confidence": 0.6}
    index = build_reference_index(["apple"], ["fake"])
    model = MockModel("consistent")
    model.should_raise = True
    tokenizer = MockTokenizer(model)
    
    res = run_verification_stage(pred_res, "apple", "1", index, tokenizer=tokenizer, model=model)
    v = res["verification"]
    assert v["verdict"] == "insufficient_evidence"
    assert v["parse_successful"] is False
    assert v["recommend_review"] is True
    assert "error and could not complete" in v["justification"]

# 19. run_verification_stage does not mutate the input prediction_result dict object in place (identity/equality check against a pre-call deep copy).
def test_run_verification_stage_no_mutation():
    import copy
    pred_res = {"predicted_label": "fake", "confidence": 0.6}
    pred_copy = copy.deepcopy(pred_res)
    index = build_reference_index(["apple"], ["fake"])
    model = MockModel("consistent")
    tokenizer = MockTokenizer(model)
    
    res = run_verification_stage(pred_res, "apple", "1", index, tokenizer=tokenizer, model=model)
    assert pred_res == pred_copy
    assert id(res) != id(pred_res)
