import sqlite3

DB_PATH = 'quiz.db'

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn

def init_db():
    conn = get_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            topic       TEXT NOT NULL,
            difficulty  TEXT NOT NULL,
            started_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS answers (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   INTEGER NOT NULL REFERENCES sessions(id),
            question_id  TEXT    NOT NULL,
            topic        TEXT    NOT NULL,
            selected     INTEGER NOT NULL,
            correct      INTEGER NOT NULL,
            answered_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()
