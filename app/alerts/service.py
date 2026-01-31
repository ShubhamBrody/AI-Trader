from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.audit import log_event
from app.core.db import db_conn
from app.core.settings import settings


@dataclass(frozen=True)
class Alert:
    alert_type: str
    ts: int
    payload: dict[str, Any]


class AlertService:
    def _throttle_ok(self, key: str, cooldown_seconds: int) -> bool:
        cooldown = max(0, int(cooldown_seconds))
        if cooldown == 0:
            return True

        now_ts = int(datetime.now(timezone.utc).timestamp())
        with db_conn() as conn:
            row = conn.execute("SELECT last_ts FROM alert_state WHERE key=?", (key,)).fetchone()
            last_ts = int(row["last_ts"]) if row else 0
            if last_ts and (now_ts - last_ts) < cooldown:
                return False

            conn.execute(
                "INSERT INTO alert_state (key, last_ts) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET last_ts=excluded.last_ts",
                (key, now_ts),
            )
        return True

    def maybe_emit_low_balance(
        self,
        *,
        source: str,
        balance: float | None,
        segment: str | None = None,
        currency: str = "INR",
    ) -> Alert | None:
        threshold = float(getattr(settings, "LOW_BALANCE_THRESHOLD_INR", 0.0) or 0.0)
        cooldown = int(getattr(settings, "LOW_BALANCE_COOLDOWN_SECONDS", 600) or 600)

        if threshold <= 0.0:
            return None
        if balance is None:
            return None
        if balance >= threshold:
            return None

        now_ts = int(datetime.now(timezone.utc).timestamp())
        seg = (segment or "").upper() or "NA"
        key = f"low_balance:{source}:{seg}"

        with db_conn() as conn:
            row = conn.execute("SELECT last_ts FROM alert_state WHERE key=?", (key,)).fetchone()
            last_ts = int(row["last_ts"]) if row else 0
            if last_ts and (now_ts - last_ts) < cooldown:
                return None

            conn.execute(
                "INSERT INTO alert_state (key, last_ts) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET last_ts=excluded.last_ts",
                (key, now_ts),
            )

        payload = {
            "source": source,
            "segment": seg,
            "currency": currency,
            "balance": float(balance),
            "threshold": float(threshold),
        }
        log_event("alert.low_balance", payload)
        return Alert(alert_type="low_balance", ts=now_ts, payload=payload)

    def notify(
        self,
        *,
        alert_type: str,
        payload: dict[str, Any],
        cooldown_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Best-effort out-of-band notification.

        Always writes to audit log via log_event in caller; this method only handles
        delivery (currently Telegram) with cooldown.
        """

        cd = int(cooldown_seconds if cooldown_seconds is not None else getattr(settings, "ALERT_TELEGRAM_COOLDOWN_SECONDS", 60))
        key = f"notify:{alert_type}"
        if not self._throttle_ok(key, cd):
            return {"ok": False, "detail": "cooldown"}

        # Telegram (optional)
        try:
            from app.alerts.telegram import send_telegram_message

            text = f"ALERT: {alert_type}\n{json.dumps(payload, ensure_ascii=False)}"
            return send_telegram_message(text)
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    def recent_alert_events(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with db_conn() as conn:
            cur = conn.execute(
                "SELECT id, ts, event_type, actor, request_id, payload_json "
                "FROM audit_events WHERE event_type LIKE 'alert.%' ORDER BY id DESC LIMIT ?",
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
