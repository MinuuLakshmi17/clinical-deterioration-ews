from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

class AuditStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS prediction_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, patient_id TEXT NOT NULL, probability REAL NOT NULL, tier TEXT NOT NULL, drivers TEXT NOT NULL)")
            con.commit()
    def record(self, patient_id: str, probability: float, tier: str, drivers: list[dict]):
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT INTO prediction_audit(timestamp, patient_id, probability, tier, drivers) VALUES (?, ?, ?, ?, ?)",
                        (datetime.now(timezone.utc).isoformat(), patient_id, probability, tier, json.dumps(drivers)))
            con.commit()
    def recent(self, limit: int = 20):
        with sqlite3.connect(self.path) as con:
            rows = con.execute("SELECT id, timestamp, patient_id, probability, tier, drivers FROM prediction_audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "timestamp": r[1], "patient_id": r[2], "probability": r[3], "tier": r[4], "drivers": json.loads(r[5])} for r in rows]
