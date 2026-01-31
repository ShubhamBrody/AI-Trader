from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.core.db import db_conn


def log_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    actor: str | None = None,
    request_id: str | None = None,
) -> None:
    ts = int(datetime.now(timezone.utc).timestamp())
    payload_json = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO audit_events (ts, event_type, actor, request_id, payload_json) VALUES (?, ?, ?, ?, ?)",
            (ts, event_type, actor, request_id, payload_json),
        )


def recent_events(limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    with db_conn() as conn:
        if event_type:
            cur = conn.execute(
                "SELECT id, ts, event_type, actor, request_id, payload_json FROM audit_events WHERE event_type=? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            )
        else:
            cur = conn.execute(
                "SELECT id, ts, event_type, actor, request_id, payload_json FROM audit_events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        out: list[dict[str, Any]] = []
        for r in cur.fetchall():
            try:
                payload = json.loads(r["payload_json"] or "{}")
            except Exception:
                payload = {"_raw": r["payload_json"]}
            out.append(
                {
                    "id": int(r["id"]),
                    "ts": int(r["ts"]),
                    "event_type": r["event_type"],
                    "actor": r["actor"],
                    "request_id": r["request_id"],
                    "payload": payload,
                }
            )
        return out
