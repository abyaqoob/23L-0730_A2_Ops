import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

def test_database_module():
    """Verify database import and initialization fallback logic."""
    from src import database
    assert hasattr(database, "init_db")
    assert hasattr(database, "log_prediction")
    assert hasattr(database, "get_prediction_history")

def test_sqlite_fallback_table_creation(tmp_path, monkeypatch):
    """Test SQLite initialization in isolated temp directory."""
    temp_db = tmp_path / "test_audit.db"
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("DATABASE_PATH", str(temp_db))
    
    from src import database
    database.DB_PATH = str(temp_db)
    database.IS_POSTGRES = False
    
    database.init_db()
    assert os.path.exists(temp_db)
    
    # Test logging and retrieval
    database.log_prediction("test.png", "test_model", "Real", 0.95, 0.95, 0.5)
    history = database.get_prediction_history(limit=5)
    assert len(history) == 1
    assert history[0]["filename"] == "test.png"
    assert history[0]["verdict"] == "Real"
