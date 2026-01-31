from __future__ import annotations

from datetime import datetime, timezone

from app.core.db import db_conn


class WatchlistService:
    def list(self) -> list[dict]:
        with db_conn() as conn:
            cur = conn.execute("SELECT instrument_key, label, created_ts FROM watchlist ORDER BY created_ts DESC")
            return [
                {
                    "instrument_key": r["instrument_key"],
                    "label": r["label"],
                    "created_ts": int(r["created_ts"]),
                }
                for r in cur.fetchall()
            ]

    def upsert(self, instrument_key: str, label: str | None = None) -> dict:
        ts = int(datetime.now(timezone.utc).timestamp())
        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO watchlist (instrument_key, label, created_ts)
                VALUES (?, ?, ?)
                ON CONFLICT(instrument_key) DO UPDATE SET
                    label=excluded.label
                """,
                (instrument_key, label, ts),
            )
        return {"instrument_key": instrument_key, "label": label}

    def remove(self, instrument_key: str) -> dict:
        with db_conn() as conn:
            conn.execute("DELETE FROM watchlist WHERE instrument_key=?", (instrument_key,))
        return {"removed": instrument_key}
