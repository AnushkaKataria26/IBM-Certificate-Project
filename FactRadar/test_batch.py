import pandas as pd
import requests
import time

df = pd.read_csv("fake-news-detection/test_data/test_articles.csv")
df['article_id'] = df['article_id'].astype(str)
df['title'] = df['title'].fillna("").astype(str)
df['text'] = df['text'].fillna("").astype(str)
valid_df = df[df['text'].str.strip() != ""]
articles = valid_df[['article_id', 'title', 'text']].to_dict(orient="records")

start = time.time()
try:
    res = requests.post("http://localhost:8000/predict/batch", json={"articles": articles}, timeout=150)
    print("Success in", time.time() - start)
    print(res.status_code)
except Exception as e:
    import traceback
    traceback.print_exc()
