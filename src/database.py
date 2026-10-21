import os
import sqlite3
import datetime

DB_PATH = os.getenv("DATABASE_PATH", "data/visionguard_audit.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            model_name TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            raw_score REAL NOT NULL,
            threshold REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialized database audit table at {DB_PATH}")

def log_prediction(filename: str, model_name: str, verdict: str, confidence: float, raw_score: float, threshold: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO prediction_audit_trail 
        (timestamp, filename, model_name, verdict, confidence, raw_score, threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now_iso, filename, model_name, verdict, confidence, raw_score, threshold))
    conn.commit()
    conn.close()

def get_prediction_history(limit: int = 50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, filename, model_name, verdict, confidence, raw_score, threshold
        FROM prediction_audit_trail
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
