import pandas as pd
import json
from src.serving.app import load_model, unified_predict_proba, _pipeline
from src.preprocessing.clean_text import clean_text
from src.serving.explain_lime import explain_instance

df = pd.read_csv("../test_articles.csv")
print("Data:")
for i, row in df.iterrows():
    text = row['text']
    cleaned = clean_text(text)
    probs = unified_predict_proba([cleaned])[0]
    pred_class = probs.argmax()
    print(f"ID: {row['article_id']}, Probs: {probs}, Pred Class: {pred_class}")

print("\nTesting LIME on first article:")
cleaned = clean_text(df.iloc[0]['text'])
try:
    exp = explain_instance(df.iloc[0]['text'], unified_predict_proba, 10)
    print("LIME Success:", len(exp))
except Exception as e:
    print("LIME Error:", e)
