from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from app.core.db import db_conn


@dataclass(frozen=True)
class TradeControls:
    freeze_new_orders: bool
    agent_disabled: bool
    ts_updated: int


def _now_ts() -> int:
    return int(time.time())


def _get_raw(key: str) -> str | None:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM trade_controls WHERE key=?", (str(key),)).fetchone()
        return None if row is None else str(row["value"])


def _set_raw(key: str, value: str) -> None:
    ts = _now_ts()
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO trade_controls(key, value, ts_updated) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts_updated=excluded.ts_updated",
            (str(key), str(value), int(ts)),
        )


def get_controls() -> TradeControls:
    freeze = _get_raw("freeze_new_orders")
    disabled = _get_raw("agent_disabled")
    ts = _get_raw("ts_updated")

    # defaults
    freeze_b = str(freeze or "false").strip().lower() in {"1", "true", "yes", "on"}
    disabled_b = str(disabled or "false").strip().lower() in {"1", "true", "yes", "on"}

    # store a last-updated timestamp key (optional)
    try:
        ts_i = int(float(ts)) if ts else 0
    except Exception:
        ts_i = 0

    # If per-key timestamps exist, use the latest.
    with db_conn() as conn:
        row = conn.execute("SELECT MAX(ts_updated) AS m FROM trade_controls").fetchone()
        if row and row["m"] is not None:
            ts_i = int(row["m"])

    return TradeControls(freeze_new_orders=freeze_b, agent_disabled=disabled_b, ts_updated=ts_i)


def set_freeze_new_orders(*, enabled: bool, actor: str = "api") -> TradeControls:
    _set_raw("freeze_new_orders", "true" if enabled else "false")
    _set_raw("last_change", json.dumps({"actor": actor, "key": "freeze_new_orders", "enabled": bool(enabled)}, separators=(",", ":")))
    return get_controls()


def set_agent_disabled(*, disabled: bool, actor: str = "api") -> TradeControls:
    _set_raw("agent_disabled", "true" if disabled else "false")
    _set_raw("last_change", json.dumps({"actor": actor, "key": "agent_disabled", "disabled": bool(disabled)}, separators=(",", ":")))
    return get_controls()


def assert_new_orders_allowed() -> None:
    c = get_controls()
    if c.freeze_new_orders:
        raise PermissionError("freeze_new_orders=true: new orders are currently blocked")
