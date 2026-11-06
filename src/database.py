import os
import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    import psycopg2
    from psycopg2.extras import RealDictCursor
    IS_POSTGRES = True
else:
    import sqlite3
    IS_POSTGRES = False
    DB_PATH = os.getenv("DATABASE_PATH", "data/visionguard_audit.db")

def get_db_connection():
    if IS_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    if IS_POSTGRES:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_audit_trail (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(64) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                model_name VARCHAR(128) NOT NULL,
                verdict VARCHAR(64) NOT NULL,
                confidence REAL NOT NULL,
                raw_score REAL NOT NULL,
                threshold REAL NOT NULL
            );
        """)
    else:
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
            );
        """)
    conn.commit()
    cursor.close()
    conn.close()

def log_prediction(filename: str, model_name: str, verdict: str, confidence: float, raw_score: float, threshold: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    placeholder = "%s, %s, %s, %s, %s, %s, %s" if IS_POSTGRES else "?, ?, ?, ?, ?, ?, ?"
    cursor.execute(f"""
        INSERT INTO prediction_audit_trail 
        (timestamp, filename, model_name, verdict, confidence, raw_score, threshold)
        VALUES ({placeholder})
    """, (now_iso, filename, model_name, verdict, confidence, raw_score, threshold))
    conn.commit()
    cursor.close()
    conn.close()

def get_prediction_history(limit: int = 50):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor) if IS_POSTGRES else conn.cursor()
    placeholder = "%s" if IS_POSTGRES else "?"
    cursor.execute(f"""
        SELECT timestamp, filename, model_name, verdict, confidence, raw_score, threshold
        FROM prediction_audit_trail
        ORDER BY id DESC
        LIMIT {placeholder}
    """, (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
