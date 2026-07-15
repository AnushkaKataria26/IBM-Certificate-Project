import pytest
import time
from unittest.mock import MagicMock, patch

from src.serving.explanation_llm import (
    build_explanation_prompt,
    generate_explanation,
    generate_template_fallback,
    attach_explanation_to_prediction,
    load_explanation_model,
    ExplanationModelLoadError
)

def test_1_build_explanation_prompt_with_tokens():
    tokens = [("token1", 0.5), ("token2", 0.4), ("token3", 0.3), ("token4", 0.2), ("token5", 0.1)]
    prompt = build_explanation_prompt("Some article text", "fake", 0.854, tokens)
    assert "fake" in prompt
    assert "85.4%" in prompt
    for t, _ in tokens:
        assert t in prompt

def test_2_build_explanation_prompt_empty_tokens():
    prompt = build_explanation_prompt("Some article text", "fake", 0.854, [])
    assert "fake" in prompt
    assert "85.4%" in prompt
    assert "No token-level attribution was available." in prompt

def test_3_build_explanation_prompt_truncated():
    long_text = "A" * 2000
    prompt = build_explanation_prompt(long_text, "fake", 0.854, [], max_article_chars=1500)
    assert "...[truncated]" in prompt
    # Check the article portion length by splitting on "Article:\n"
    article_portion = prompt.split("Article:\n")[1].split("\nClassification:")[0]
    assert len(article_portion) <= 1500 + 20

def test_4_generate_explanation_mocked_success():
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.name_or_path = "mocked-model"
    mock_outputs = [MagicMock()]
    mock_model.generate.return_value = mock_outputs

    mock_tokenizer = MagicMock()
    # Return a 50-character string
    mock_tokenizer.decode.return_value = "A" * 50
    mock_tokenizer.return_value.to.return_value = {"input_ids": [1]}

    res = generate_explanation("test", "fake", 0.9, [("test", 1.0)], tokenizer=mock_tokenizer, model=mock_model)
    assert res["fallback_used"] is False
    assert res["explanation"] == "A" * 50
    assert res["model_used"] == "mocked-model"

def test_5_generate_explanation_empty_output_fallback():
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = [MagicMock()]

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = ""
    mock_tokenizer.return_value.to.return_value = {"input_ids": [1]}

    res = generate_explanation("test", "fake", 0.9, [("test", 1.0)], tokenizer=mock_tokenizer, model=mock_model)
    assert res["fallback_used"] is True
    assert res["explanation"] == generate_template_fallback("fake", 0.9, [("test", 1.0)])

@patch('src.serving.explanation_llm.load_explanation_model')
def test_6_generate_explanation_load_error(mock_load):
    mock_load.side_effect = ExplanationModelLoadError("test error")
    res = generate_explanation("test", "fake", 0.9, [("test", 1.0)], tokenizer=None, model=None)
    assert res["fallback_used"] is True
    assert res["model_used"] == "template_fallback"
    assert "fake" in res["explanation"]

def test_7_generate_explanation_timeout():
    mock_model = MagicMock()
    mock_model.device = "cpu"
    def sleepy_generate(*args, **kwargs):
        time.sleep(0.5)
        return [MagicMock()]
    mock_model.generate.side_effect = sleepy_generate

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Success text"
    mock_tokenizer.return_value.to.return_value = {"input_ids": [1]}

    start = time.time()
    res = generate_explanation("test", "fake", 0.9, [("test", 1.0)], tokenizer=mock_tokenizer, model=mock_model, timeout_seconds=0.1)
    duration = time.time() - start
    
    assert duration < 0.5
    assert res["fallback_used"] is True

def test_8_generate_template_fallback_empty_tokens():
    res = generate_template_fallback("fake", 0.9, [])
    assert "fake" in res
    assert "90.0%" in res
    assert "linguistic patterns" in res
    assert len(res) > 10

def test_9_generate_template_fallback_negative_score():
    res = generate_template_fallback("fake", 0.9, [("negative_token", -10.0)])
    assert "negative_token" in res

def test_10_attach_explanation():
    prediction_result = {"predicted_label": "fake", "confidence": 0.9, "extra_key": "val"}
    original_copy = prediction_result.copy()
    
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Some explanation"
    
    new_result = attach_explanation_to_prediction(prediction_result, "text", [], tokenizer=mock_tokenizer, model=mock_model)
    
    assert "explanation" in new_result
    assert new_result["extra_key"] == "val"
    assert new_result["predicted_label"] == "fake"
    assert prediction_result == original_copy  # unmodified
    assert id(prediction_result) != id(new_result)

def test_11_attach_explanation_missing_field():
    with pytest.raises(KeyError, match="confidence"):
        attach_explanation_to_prediction({"predicted_label": "fake"}, "text", [])

def test_12_generate_explanation_empty_text():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Some long explanation text that passes length check"
    
    res = generate_explanation("", "fake", 0.9, [], tokenizer=mock_tokenizer, model=mock_model)
    
    assert "explanation" in res
    assert "model_used" in res
    assert "fallback_used" in res
    assert "truncated" in res
    assert "generation_time_ms" in res
    assert res["truncated"] is False
    assert res["fallback_used"] is False

@pytest.mark.integration
@pytest.mark.skip(reason="Loads real model from Hugging Face Hub")
def test_13_load_explanation_model_integration():
    tokenizer, model = load_explanation_model("google/flan-t5-small") # using small for faster test
    assert tokenizer is not None
    assert model is not None
