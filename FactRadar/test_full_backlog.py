import socket
import time
import threading
import requests

def listen_only():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 8006))
    s.listen(1) # very small backlog
    print("Listening, but not accepting...")
    time.sleep(10)

threading.Thread(target=listen_only, daemon=True).start()
time.sleep(1)

# Fill the backlog
fill_socks = []
for i in range(10):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.connect_ex(('127.0.0.1', 8006))
        fill_socks.append(sock)
    except Exception:
        pass

time.sleep(1)

try:
    print("Testing connection with requests...")
    requests.post("http://127.0.0.1:8006/foo", timeout=1)
except Exception as e:
    print("Exception type:", type(e))
    print("Exception msg:", str(e))
