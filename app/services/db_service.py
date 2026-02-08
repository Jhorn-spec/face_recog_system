import sqlite3
from config import DB_PATH

class DBService:
    def __init__(self):
        self.create_table()

    def create_table(self):
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    position TEXT,
                    phone_number TEXT
                )
            ''')
            conn.commit()

    def write_user(self, id, name, position, phone):
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
                        (id, name, position, phone))
            conn.commit()

    def get_user(self, id):
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE id = ?", (id,))
            return cur.fetchone()
