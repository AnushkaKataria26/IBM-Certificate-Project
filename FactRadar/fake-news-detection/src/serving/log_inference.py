import sqlite3
import logging
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/monitoring.db")

def init_db():
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS inference_log (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    model_version TEXT,
                    input_text_hash TEXT,
                    predicted_label TEXT,
                    confidence REAL,
                    latency_ms REAL
                )
            ''')
    except Exception as e:
        logger.error(f"Failed to initialize monitoring DB: {e}")

# Initialize DB on import
init_db()

def log_prediction(model_version: str, input_text: str, predicted_label: str, confidence: float, latency_ms: float):
    """
    Log prediction details to SQLite DB for drift monitoring.
    Never blocks or throws exceptions to the caller.
    """
    try:
        req_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        
        # Hash the raw text to avoid PII / storage bloat
        text_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
        
        with sqlite3.connect(DB_PATH, timeout=2.0) as conn:
            conn.execute('''
                INSERT INTO inference_log (request_id, timestamp, model_version, input_text_hash, predicted_label, confidence, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (req_id, ts, model_version, text_hash, predicted_label, confidence, latency_ms))
    except Exception as e:
        logger.error(f"Failed to log inference to DB: {e}")
