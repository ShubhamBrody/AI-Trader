from __future__ import annotations

import json
import time
from typing import Any

from app.core.db import db_conn
from app.core.settings import settings
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError


def _now_ts() -> int:
    return int(time.time())


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _extract_positions_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("positions", "net", "items"):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _extract_token(item: dict[str, Any]) -> str | None:
    for k in ("instrument_token", "instrumentToken", "instrument", "token"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _extract_net_qty(item: dict[str, Any]) -> float:
    for k in ("net_qty", "netQty", "quantity", "qty", "net_quantity"):
        v = _safe_float(item.get(k))
        if v is not None:
            return float(v)
    # If buy/sell qty available, net = buy - sell
    buy = _safe_float(item.get("buy_qty") or item.get("buyQty") or item.get("buy_quantity"))
    sell = _safe_float(item.get("sell_qty") or item.get("sellQty") or item.get("sell_quantity"))
    if buy is not None or sell is not None:
        return float(buy or 0.0) - float(sell or 0.0)
    return 0.0


def _extract_avg(item: dict[str, Any]) -> float | None:
    for k in ("avg_price", "avgPrice", "average_price", "buy_avg_price"):
        v = _safe_float(item.get(k))
        if v is not None:
            return float(v)
    return None


def _extract_ltp(item: dict[str, Any]) -> float | None:
    for k in ("last_price", "ltp", "lastPrice"):
        v = _safe_float(item.get(k))
        if v is not None:
            return float(v)
    return None


def _extract_pnl(item: dict[str, Any]) -> float | None:
    for k in ("pnl", "day_pnl", "unrealised_pnl", "realised_pnl", "mtm"):
        v = _safe_float(item.get(k))
        if v is not None:
            return float(v)
    return None


def _instrument_key_for_token(token: str | None) -> str | None:
    if not token:
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT instrument_key FROM instrument_meta WHERE upstox_token=? LIMIT 1",
            (str(token),),
        ).fetchone()
        return None if row is None else str(row["instrument_key"])


def reconcile_upstox_positions() -> dict[str, Any]:
    if not settings.UPSTOX_ACCESS_TOKEN:
        return {"ok": False, "detail": "UPSTOX_ACCESS_TOKEN is not configured"}

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client: UpstoxClient | None = None

    try:
        client = UpstoxClient(cfg)
        payload = client.positions_v2()
    except UpstoxError as e:
        return {"ok": False, "detail": str(e)}
    finally:
        if client is not None:
            client.close()

    ts = _now_ts()
    items = _extract_positions_list(payload)

    with db_conn() as conn:
        conn.execute(
            "INSERT INTO positions_snapshots(ts, broker, payload_json) VALUES(?,?,?)",
            (int(ts), "upstox", _json_dumps(payload)),
        )

        upserted = 0
        for it in items:
            token = _extract_token(it)
            if not token:
                continue
            net_qty = float(_extract_net_qty(it))
            ik = _instrument_key_for_token(token)
            avg = _extract_avg(it)
            ltp = _extract_ltp(it)
            pnl = _extract_pnl(it)

            conn.execute(
                """
                INSERT INTO positions_state(
                    broker, instrument_key, instrument_token, net_qty, avg_price, last_price, pnl, ts_updated, raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(broker, instrument_token)
                DO UPDATE SET
                    instrument_key=excluded.instrument_key,
                    net_qty=excluded.net_qty,
                    avg_price=excluded.avg_price,
                    last_price=excluded.last_price,
                    pnl=excluded.pnl,
                    ts_updated=excluded.ts_updated,
                    raw_json=excluded.raw_json
                """,
                (
                    "upstox",
                    ik,
                    str(token),
                    float(net_qty),
                    (None if avg is None else float(avg)),
                    (None if ltp is None else float(ltp)),
                    (None if pnl is None else float(pnl)),
                    int(ts),
                    _json_dumps(it),
                ),
            )
            upserted += 1

    return {"ok": True, "ts": int(ts), "count": len(items), "upserted": upserted}


def list_positions_state(*, broker: str = "upstox", limit: int = 500) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 5000))
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT broker, instrument_key, instrument_token, net_qty, avg_price, last_price, pnl, ts_updated FROM positions_state WHERE broker=? ORDER BY ABS(net_qty) DESC, ts_updated DESC LIMIT ?",
            (str(broker), int(limit)),
        )
        out: list[dict[str, Any]] = []
        for r in cur.fetchall():
            out.append(
                {
                    "broker": r["broker"],
                    "instrument_key": r["instrument_key"],
                    "instrument_token": r["instrument_token"],
                    "net_qty": float(r["net_qty"]),
                    "avg_price": (None if r["avg_price"] is None else float(r["avg_price"])),
                    "last_price": (None if r["last_price"] is None else float(r["last_price"])),
                    "pnl": (None if r["pnl"] is None else float(r["pnl"])),
                    "ts_updated": int(r["ts_updated"]),
                }
            )
        return out


def get_position_by_instrument_key(instrument_key: str, *, broker: str = "upstox") -> dict[str, Any] | None:
    ik = str(instrument_key or "").strip()
    if not ik:
        return None
    with db_conn() as conn:
        r = conn.execute(
            "SELECT broker, instrument_key, instrument_token, net_qty, avg_price, last_price, pnl, ts_updated FROM positions_state WHERE broker=? AND instrument_key=? LIMIT 1",
            (str(broker), ik),
        ).fetchone()
        if r is None:
            return None
        return {
            "broker": r["broker"],
            "instrument_key": r["instrument_key"],
            "instrument_token": r["instrument_token"],
            "net_qty": float(r["net_qty"]),
            "avg_price": (None if r["avg_price"] is None else float(r["avg_price"])),
            "last_price": (None if r["last_price"] is None else float(r["last_price"])),
            "pnl": (None if r["pnl"] is None else float(r["pnl"])),
            "ts_updated": int(r["ts_updated"]),
        }


def broker_pnl_estimate(*, broker: str = "upstox") -> float | None:
    # Conservative: sum per-position pnl if available.
    with db_conn() as conn:
        cur = conn.execute("SELECT pnl FROM positions_state WHERE broker=?", (str(broker),))
        total = 0.0
        found = False
        for r in cur.fetchall():
            if r["pnl"] is None:
                continue
            found = True
            try:
                total += float(r["pnl"])
            except Exception:
                pass
        return total if found else None
