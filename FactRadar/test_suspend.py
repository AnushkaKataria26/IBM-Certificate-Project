import socket
import time
import threading
import requests
import os
import signal

def server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', 8003))
    s.listen(5)
    print("Server listening, suspending itself...")
    os.kill(os.getpid(), signal.SIGSTOP) # suspend process
    print("Resumed!")

pid = os.fork()
if pid == 0:
    server()
    os._exit(0)

time.sleep(1) # wait for child to bind and suspend

try:
    print("Testing connection...")
    requests.post("http://127.0.0.1:8003/foo", timeout=2)
except Exception as e:
    print("Exception type:", type(e))
    print("Exception msg:", str(e))
finally:
    os.kill(pid, signal.SIGKILL)
