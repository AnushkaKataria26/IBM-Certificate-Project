import asyncio
from src.serving.rag_verification import build_reference_index, run_verification_stage
from src.serving.explanation_llm import load_explanation_model
import pandas as pd

df = pd.read_csv("data/splits/train.csv").head(100)
df['clean_text'] = df['clean_text'].fillna("")
df['label_str'] = df['label'].map({1: "fake", 0: "real"}).fillna("unknown")
reference_index = build_reference_index(
    corpus_texts=df['clean_text'].tolist(),
    corpus_labels=df['label_str'].tolist(),
    corpus_ids=df['article_id'].astype(str).tolist()
)

tokenizer, model = load_explanation_model()

prediction_result = {"predicted_label": "fake", "confidence": 0.55}

article_text = "The quick brown fox jumps over the lazy dog."
res = run_verification_stage(
    prediction_result=prediction_result,
    article_text=article_text,
    article_id="999",
    reference_index=reference_index,
    tokenizer=tokenizer,
    model=model
)
print("RAG Verification Result:", res.get("verification"))
