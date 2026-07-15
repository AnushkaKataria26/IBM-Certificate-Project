import requests
import json

article_text = "Coffee can help improve productivity and productivity in the workplace."

payload = {
    "article_id": "test_2",
    "text": article_text
}

try:
    resp = requests.post("http://localhost:8000/analyze", json=payload, timeout=60)
    print("Status Code:", resp.status_code)
    print("Response:", json.dumps(resp.json(), indent=2))
except Exception as e:
    print("Error:", e)
