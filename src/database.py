import sqlite3
from datetime import datetime
from .config import DB_PATH

def init_db():
    """Ініціалізація БД: створення таблиці та індексів."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            prediction VARCHAR(10),
            confidence FLOAT,
            processing_time_ms FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_logs(timestamp);")
    
    conn.commit()
    conn.close()

def log_request(text: str, prediction: str, confidence: float, processing_time: float):
    """Запис результату перевірки в лог."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO usage_logs (input_text, prediction, confidence, processing_time_ms)
            VALUES (?, ?, ?, ?)
        """, (text[:500], prediction, confidence, processing_time))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Logging Error: {e}")

def get_stats():
    """Отримання статистики для відображення в UI."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT prediction, count(*) FROM usage_logs GROUP BY prediction")
        data = cursor.fetchall()
        return data
    except Exception:
        return []
    finally:
        conn.close()