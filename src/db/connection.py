import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "planograma_retail.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Devuelve una conexión SQLite validada con llaves foráneas activas."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"No existe la base de datos en {db_path}. Ejecuta primero src/db/init_db.py."
        )
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn