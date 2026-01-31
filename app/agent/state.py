from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from zoneinfo import ZoneInfo

from app.core.db import db_conn


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def local_day_bounds_utc(*, day_local: date, tz_name: str) -> tuple[int, int]:
    tz = ZoneInfo(tz_name)
    start_local = datetime(day_local.year, day_local.month, day_local.day, 0, 0, 0, tzinfo=tz)
    # Use [start, next-day-start) bounds for correctness.
    next_start_local = start_local + timedelta(days=1)
    return int(start_local.astimezone(timezone.utc).timestamp()), int(next_start_local.astimezone(timezone.utc).timestamp())


def record_trade_open(
    *,
    instrument_key: str,
    side: str,
    qty: float,
    entry: float,
    stop: float,
    target: float,
    entry_order_id: str | None,
    sl_order_id: str | None,
    monitor_app_stop: bool,
    ts_open: int | None = None,
    meta: dict[str, Any] | None = None,
) -> int:
    ts = int(ts_open) if ts_open is not None else _now_ts()
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO agent_trades (ts_open, ts_close, instrument_key, side, qty, entry, stop, target, status, entry_order_id, sl_order_id, monitor_app_stop, meta_json) "
            "VALUES (?, NULL, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)",
            (
                ts,
                instrument_key,
                side,
                float(qty),
                float(entry),
                float(stop),
                float(target),
                entry_order_id,
                sl_order_id,
                1 if monitor_app_stop else 0,
                json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        return int(cur.lastrowid)


def mark_trade_closed(
    trade_id: int,
    *,
    reason: str,
    exit_order_id: str | None = None,
    ts_close: int | None = None,
    close_price: float | None = None,
    pnl: float | None = None,
) -> None:
    ts = int(ts_close) if ts_close is not None else _now_ts()

    learned_payload: dict[str, Any] | None = None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT instrument_key, side, qty, entry, stop, meta_json FROM agent_trades WHERE id=?",
            (int(trade_id),),
        ).fetchone()
        try:
            meta = json.loads((row["meta_json"] if row else "{}") or "{}")
        except Exception:
            meta = {}
        if close_price is not None:
            meta["close_price"] = float(close_price)
        if pnl is not None:
            meta["pnl"] = float(pnl)

        # Prepare learning payload outside the DB transaction.
        try:
            instrument_key = (row["instrument_key"] if row else None)
            side = (row["side"] if row else None)
            qty = float(row["qty"] if row else 0.0)
            entry = float(row["entry"] if row else 0.0)
            # If pnl is missing but we have a close price, compute a best-effort PnL.
            pnl_eff = pnl
            if pnl_eff is None and close_price is not None and entry > 0 and qty > 0 and side:
                if str(side).upper() == "BUY":
                    pnl_eff = (float(close_price) - entry) * qty
                elif str(side).upper() == "SELL":
                    pnl_eff = (entry - float(close_price)) * qty

            learned_payload = {
                "instrument_key": (str(instrument_key) if instrument_key else None),
                "side": (str(side) if side else ""),
                "pnl": (None if pnl_eff is None else float(pnl_eff)),
                "meta": (meta if isinstance(meta, dict) else {}),
                "close_reason": str(reason or ""),
            }
        except Exception:
            learned_payload = None

        conn.execute(
            "UPDATE agent_trades SET ts_close=?, status='CLOSED', close_reason=?, exit_order_id=?, meta_json=? WHERE id=?",
            (
                ts,
                reason,
                exit_order_id,
                json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":")),
                int(trade_id),
            ),
        )

    # Best-effort: update pattern memory based on trade outcome.
    try:
        if learned_payload and learned_payload.get("pnl") is not None:
            from app.ai.pattern_memory import learn_from_trade_outcome

            learn_from_trade_outcome(
                instrument_key=learned_payload.get("instrument_key"),
                side=str(learned_payload.get("side") or ""),
                pnl=float(learned_payload.get("pnl")),
                meta=learned_payload.get("meta") if isinstance(learned_payload.get("meta"), dict) else {},
                close_reason=str(learned_payload.get("close_reason") or ""),
            )
    except Exception:
        # Never fail closing a trade due to learning.
        pass


def count_open_trades() -> int:
    with db_conn() as conn:
        row = conn.execute("SELECT COUNT(1) AS c FROM agent_trades WHERE status='OPEN'").fetchone()
        return int(row["c"] if row else 0)


def has_open_trade(*, instrument_key: str) -> bool:
    if not instrument_key:
        return False
    with db_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM agent_trades WHERE status='OPEN' AND instrument_key=? LIMIT 1",
            (instrument_key,),
        ).fetchone()
        return bool(row)


def count_trades_today(*, tz_name: str, now_utc: datetime | None = None) -> int:
    now_utc = now_utc or datetime.now(timezone.utc)
    tz = ZoneInfo(tz_name)
    day_local = now_utc.astimezone(tz).date()
    start_ts, end_ts = local_day_bounds_utc(day_local=day_local, tz_name=tz_name)
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(1) AS c FROM agent_trades WHERE ts_open >= ? AND ts_open < ?",
            (int(start_ts), int(end_ts)),
        ).fetchone()
        return int(row["c"] if row else 0)


def realized_pnl_today(*, tz_name: str, now_utc: datetime | None = None, limit: int = 2000) -> float:
    now_utc = now_utc or datetime.now(timezone.utc)
    tz = ZoneInfo(tz_name)
    day_local = now_utc.astimezone(tz).date()
    start_ts, end_ts = local_day_bounds_utc(day_local=day_local, tz_name=tz_name)
    limit = max(1, min(int(limit), 5000))

    total = 0.0
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT meta_json FROM agent_trades WHERE status='CLOSED' AND ts_close IS NOT NULL AND ts_close >= ? AND ts_close < ? ORDER BY ts_close DESC LIMIT ?",
            (int(start_ts), int(end_ts), int(limit)),
        )
        for r in cur.fetchall():
            try:
                meta = json.loads(r["meta_json"] or "{}")
            except Exception:
                meta = {}
            try:
                total += float(meta.get("pnl") or 0.0)
            except Exception:
                pass
    return float(total)


def has_traded_today(*, instrument_key: str, tz_name: str, now_utc: datetime | None = None) -> bool:
    if not instrument_key:
        return False
    now_utc = now_utc or datetime.now(timezone.utc)
    tz = ZoneInfo(tz_name)
    day_local = now_utc.astimezone(tz).date()
    start_ts, end_ts = local_day_bounds_utc(day_local=day_local, tz_name=tz_name)
    with db_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM agent_trades WHERE instrument_key=? AND ts_open >= ? AND ts_open < ? LIMIT 1",
            (instrument_key, int(start_ts), int(end_ts)),
        ).fetchone()
        return bool(row)


def list_open_trades(limit: int = 200) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT id, ts_open, instrument_key, side, qty, entry, stop, target, entry_order_id, sl_order_id, monitor_app_stop, meta_json "
            "FROM agent_trades WHERE status='OPEN' ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        out: list[dict[str, Any]] = []
        for r in cur.fetchall():
            try:
                meta = json.loads(r["meta_json"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "id": int(r["id"]),
                    "ts_open": int(r["ts_open"]),
                    "instrument_key": r["instrument_key"],
                    "side": r["side"],
                    "qty": float(r["qty"]),
                    "entry": float(r["entry"]),
                    "stop": float(r["stop"]),
                    "target": float(r["target"]),
                    "entry_order_id": r["entry_order_id"],
                    "sl_order_id": r["sl_order_id"],
                    "monitor_app_stop": bool(int(r["monitor_app_stop"])),
                    "meta": meta,
                }
            )
        return out


def get_trade(trade_id: int) -> dict[str, Any] | None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT id, ts_open, ts_close, instrument_key, side, qty, entry, stop, target, status, entry_order_id, sl_order_id, exit_order_id, close_reason, monitor_app_stop, meta_json "
            "FROM agent_trades WHERE id=?",
            (int(trade_id),),
        ).fetchone()
        if not row:
            return None
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}
        return {
            "id": int(row["id"]),
            "ts_open": int(row["ts_open"]),
            "ts_close": (None if row["ts_close"] is None else int(row["ts_close"])),
            "instrument_key": row["instrument_key"],
            "side": row["side"],
            "qty": float(row["qty"]),
            "entry": float(row["entry"]),
            "stop": float(row["stop"]),
            "target": float(row["target"]),
            "status": row["status"],
            "entry_order_id": row["entry_order_id"],
            "sl_order_id": row["sl_order_id"],
            "exit_order_id": row["exit_order_id"],
            "close_reason": row["close_reason"],
            "monitor_app_stop": bool(int(row["monitor_app_stop"])),
            "meta": meta,
        }


def list_recent_trades(limit: int = 200, *, include_open: bool = True) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    where = "" if include_open else "WHERE status='CLOSED'"
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT id FROM agent_trades " + where + " ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        out: list[dict[str, Any]] = []
        for r in cur.fetchall():
            t = get_trade(int(r["id"]))
            if t:
                out.append(t)
        return out


def update_trade(
    trade_id: int,
    *,
    entry_order_id: str | None = None,
    sl_order_id: str | None = None,
    monitor_app_stop: bool | None = None,
    stop: float | None = None,
    target: float | None = None,
    meta_patch: dict[str, Any] | None = None,
) -> None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT entry_order_id, sl_order_id, monitor_app_stop, meta_json FROM agent_trades WHERE id=?",
            (int(trade_id),),
        ).fetchone()
        if row is None:
            return

        current_entry = row["entry_order_id"]
        current_sl = row["sl_order_id"]
        current_monitor = bool(int(row["monitor_app_stop"]))
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}

        if meta_patch:
            meta.update(meta_patch)

        new_entry = current_entry if entry_order_id is None else entry_order_id
        new_sl = current_sl if sl_order_id is None else sl_order_id
        new_monitor = current_monitor if monitor_app_stop is None else bool(monitor_app_stop)

        new_stop = None if stop is None else float(stop)
        new_target = None if target is None else float(target)

        if new_stop is None and new_target is None:
            conn.execute(
                "UPDATE agent_trades SET entry_order_id=?, sl_order_id=?, monitor_app_stop=?, meta_json=? WHERE id=?",
                (
                    new_entry,
                    new_sl,
                    1 if new_monitor else 0,
                    json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":")),
                    int(trade_id),
                ),
            )
        else:
            # Keep existing stop/target if not provided.
            row2 = conn.execute(
                "SELECT stop, target FROM agent_trades WHERE id=?",
                (int(trade_id),),
            ).fetchone()
            cur_stop = float(row2["stop"]) if row2 else 0.0
            cur_target = float(row2["target"]) if row2 else 0.0
            conn.execute(
                "UPDATE agent_trades SET entry_order_id=?, sl_order_id=?, monitor_app_stop=?, stop=?, target=?, meta_json=? WHERE id=?",
                (
                    new_entry,
                    new_sl,
                    1 if new_monitor else 0,
                    cur_stop if new_stop is None else new_stop,
                    cur_target if new_target is None else new_target,
                    json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":")),
                    int(trade_id),
                ),
            )
