import socket
import time
import threading
import requests

def listen_only():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 8002))
    s.listen(5)
    print("Listening, but not accepting...")
    time.sleep(10)

threading.Thread(target=listen_only, daemon=True).start()
time.sleep(1)

try:
    requests.post("http://127.0.0.1:8002/foo", timeout=2)
except Exception as e:
    print("Exception type:", type(e))
    print("Exception msg:", str(e))
