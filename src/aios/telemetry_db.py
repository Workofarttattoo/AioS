import sqlite3
import json
import time

class TelemetryDB:
    """
    A simple SQLite database for storing and retrieving system telemetry.
    """
    def __init__(self, db_path="telemetry.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_table()

    def connect(self):
        """Establish a connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)

    def create_table(self):
        """Create the telemetry table if it doesn't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL
                )
            """)

    def store(self, telemetry_data: dict):
        """Store a new telemetry snapshot."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO telemetry (timestamp, data) VALUES (?, ?)",
                (time.time(), json.dumps(telemetry_data))
            )

    def retrieve_recent(self, limit: int = 1000) -> list[dict]:
        """Retrieve the most recent telemetry snapshots."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT data FROM telemetry ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


