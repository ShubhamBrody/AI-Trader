from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.core.db import db_conn


def slot_key(*, instrument_key: str, interval: str, horizon_steps: int, model_family: str | None, cap_tier: str | None, kind: str) -> str:
    fam = (model_family or "generic").lower()
    cap = (cap_tier or "unknown").lower()
    return f"{kind.lower()}::{fam}::{cap}::{instrument_key}::{interval}::h{int(horizon_steps)}"


def promote(*, slot_key: str, model_key: str, kind: str, stage: str, metrics: dict[str, Any]) -> None:
    now = int(datetime.now(timezone.utc).timestamp())
    metrics_json = json.dumps(metrics or {}, ensure_ascii=False, separators=(",", ":"))
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO model_registry (slot_key, model_key, kind, stage, updated_ts, metrics_json) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(slot_key) DO UPDATE SET model_key=excluded.model_key, kind=excluded.kind, stage=excluded.stage, updated_ts=excluded.updated_ts, metrics_json=excluded.metrics_json",
            (str(slot_key), str(model_key), str(kind), str(stage), int(now), metrics_json),
        )


def get_registry(slot_key: str) -> dict[str, Any] | None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT slot_key, model_key, kind, stage, updated_ts, metrics_json FROM model_registry WHERE slot_key=?",
            (str(slot_key),),
        ).fetchone()
        if not row:
            return None
        try:
            metrics = json.loads(row["metrics_json"] or "{}")
        except Exception:
            metrics = {}
        return {
            "slot_key": row["slot_key"],
            "model_key": row["model_key"],
            "kind": row["kind"],
            "stage": row["stage"],
            "updated_ts": int(row["updated_ts"]),
            "metrics": metrics,
        }
