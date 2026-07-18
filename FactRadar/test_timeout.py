import requests
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import time

class SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(3)
        self.send_response(200)
        self.end_headers()

server = HTTPServer(('127.0.0.1', 8001), SlowHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()

try:
    requests.post("http://127.0.0.1:8001/foo", timeout=1)
except Exception as e:
    print("Exception type:", type(e))
    print("Exception msg:", str(e))
