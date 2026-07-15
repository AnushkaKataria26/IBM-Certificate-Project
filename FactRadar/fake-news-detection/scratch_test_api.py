import requests
import json

article_text = "The quick brown fox jumps over the lazy dog."

payload = {
    "article_id": "test_1",
    "text": article_text
}

try:
    resp = requests.post("http://localhost:8000/analyze", json=payload, timeout=60)
    print("Status Code:", resp.status_code)
    print("Response:", json.dumps(resp.json(), indent=2))
except Exception as e:
    print("Error:", e)
