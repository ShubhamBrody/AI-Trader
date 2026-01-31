from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from app.core.db import db_conn
from app.core.settings import settings
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError

BrokerName = Literal["paper", "upstox"]


@dataclass(frozen=True)
class OrderState:
    id: str
    ts_created: int
    ts_updated: int
    broker: str
    instrument_key: str | None
    symbol: str | None
    side: str
    qty: float
    order_kind: str
    order_type: str
    price: float | None
    trigger_price: float | None
    client_order_id: str | None
    broker_order_id: str | None
    status: str
    last_error: str | None
    meta: dict[str, Any]


def _now_ts() -> int:
    return int(time.time())


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))


def _json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _row_to_state(row: Any) -> OrderState:
    meta = _json_loads(row["meta_json"])
    return OrderState(
        id=str(row["id"]),
        ts_created=int(row["ts_created"]),
        ts_updated=int(row["ts_updated"]),
        broker=str(row["broker"]),
        instrument_key=(None if row["instrument_key"] is None else str(row["instrument_key"])),
        symbol=(None if row["symbol"] is None else str(row["symbol"])),
        side=str(row["side"]),
        qty=float(row["qty"]),
        order_kind=str(row["order_kind"]),
        order_type=str(row["order_type"]),
        price=(None if row["price"] is None else float(row["price"])),
        trigger_price=(None if row["trigger_price"] is None else float(row["trigger_price"])),
        client_order_id=(None if row["client_order_id"] is None else str(row["client_order_id"])),
        broker_order_id=(None if row["broker_order_id"] is None else str(row["broker_order_id"])),
        status=str(row["status"]),
        last_error=(None if row["last_error"] is None else str(row["last_error"])),
        meta=meta,
    )


def add_event(order_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
    ts = _now_ts()
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO order_events(ts, order_id, event_type, payload_json) VALUES (?, ?, ?, ?)",
            (ts, str(order_id), str(event_type), _json_dumps(payload or {})),
        )


def create_order(
    *,
    broker: BrokerName,
    side: str,
    qty: float,
    order_kind: str,
    order_type: str,
    instrument_key: str | None = None,
    symbol: str | None = None,
    price: float | None = None,
    trigger_price: float | None = None,
    client_order_id: str | None = None,
    broker_order_id: str | None = None,
    status: str,
    last_error: str | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    ts = _now_ts()
    oid = str(uuid.uuid4())
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO order_states(
                id, ts_created, ts_updated, broker, instrument_key, symbol, side, qty,
                order_kind, order_type, price, trigger_price,
                client_order_id, broker_order_id, status, last_error, meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                oid,
                ts,
                ts,
                str(broker),
                (None if instrument_key is None else str(instrument_key)),
                (None if symbol is None else str(symbol)),
                str(side).upper(),
                float(qty),
                str(order_kind),
                str(order_type),
                (None if price is None else float(price)),
                (None if trigger_price is None else float(trigger_price)),
                (None if not client_order_id else str(client_order_id)),
                (None if not broker_order_id else str(broker_order_id)),
                str(status),
                (None if not last_error else str(last_error)[:500]),
                _json_dumps(meta or {}),
            ),
        )

    add_event(oid, "created", {"status": status, "broker": broker})
    return oid


def update_order(
    order_id: str,
    *,
    status: str | None = None,
    broker_order_id: str | None = None,
    last_error: str | None = None,
    meta_patch: dict[str, Any] | None = None,
) -> None:
    ts = _now_ts()
    with db_conn() as conn:
        row = conn.execute(
            "SELECT status, broker_order_id, last_error, meta_json FROM order_states WHERE id=?",
            (str(order_id),),
        ).fetchone()
        if row is None:
            return

        meta = _json_loads(row["meta_json"])
        if meta_patch:
            meta.update(meta_patch)

        new_status = str(row["status"]) if status is None else str(status)
        new_broker_order_id = row["broker_order_id"] if broker_order_id is None else broker_order_id
        new_last_error = row["last_error"] if last_error is None else last_error

        conn.execute(
            "UPDATE order_states SET ts_updated=?, status=?, broker_order_id=?, last_error=?, meta_json=? WHERE id=?",
            (
                ts,
                new_status,
                new_broker_order_id,
                new_last_error,
                _json_dumps(meta),
                str(order_id),
            ),
        )

    add_event(
        str(order_id),
        "updated",
        {
            "status": status,
            "broker_order_id": broker_order_id,
            "last_error": (None if last_error is None else str(last_error)[:200]),
            "meta_patch": (meta_patch or {}),
        },
    )


def get_order(order_id: str) -> OrderState | None:
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM order_states WHERE id=?", (str(order_id),)).fetchone()
        return None if row is None else _row_to_state(row)


def get_order_by_client_order_id(client_order_id: str) -> OrderState | None:
    key = str(client_order_id or "").strip()
    if not key:
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM order_states WHERE client_order_id=? LIMIT 1",
            (key,),
        ).fetchone()
        return None if row is None else _row_to_state(row)


def list_orders(
    *,
    limit: int = 100,
    broker: str | None = None,
    status: str | None = None,
    instrument_key: str | None = None,
) -> list[OrderState]:
    limit = max(1, min(int(limit), 1000))

    where: list[str] = []
    args: list[Any] = []

    if broker:
        where.append("broker=?")
        args.append(str(broker))
    if status:
        where.append("status=?")
        args.append(str(status))
    if instrument_key:
        where.append("instrument_key=?")
        args.append(str(instrument_key))

    sql = "SELECT * FROM order_states"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts_created DESC LIMIT ?"
    args.append(limit)

    with db_conn() as conn:
        cur = conn.execute(sql, tuple(args))
        return [_row_to_state(r) for r in cur.fetchall()]


def list_events(order_id: str, limit: int = 200) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 2000))
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT id, ts, event_type, payload_json FROM order_events WHERE order_id=? ORDER BY id DESC LIMIT ?",
            (str(order_id), limit),
        )
        out: list[dict[str, Any]] = []
        for r in cur.fetchall():
            out.append(
                {
                    "id": int(r["id"]),
                    "ts": int(r["ts"]),
                    "event_type": str(r["event_type"]),
                    "payload": _json_loads(r["payload_json"]),
                }
            )
        return out


def _normalize_upstox_status(raw: str) -> str:
    s = str(raw or "").strip().upper().replace(" ", "_")
    # Common terminal states
    if s in {"COMPLETE", "COMPLETED", "FILLED", "EXECUTED"}:
        return "FILLED"
    if s in {"REJECTED", "CANCELLED", "CANCELED", "ERROR", "FAILED"}:
        return "REJECTED" if s == "REJECTED" else "CANCELLED" if "CANCEL" in s else "ERROR"

    # Common live states
    if s in {"OPEN", "PLACED", "PENDING", "PUT_ORDER_REQ_RECEIVED", "TRIGGER_PENDING"}:
        return "OPEN"

    return "UNKNOWN"


def _book_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # Some APIs nest further.
        for k in ("orders", "items", "order_list"):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _pick_order_id_from_book_item(item: dict[str, Any]) -> str | None:
    for k in ("order_id", "orderId", "id"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _pick_tag_from_book_item(item: dict[str, Any]) -> str | None:
    for k in ("tag", "order_tag", "orderTag"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _pick_status_from_book_item(item: dict[str, Any]) -> str | None:
    for k in ("status", "order_status", "orderStatus"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _pick_token_from_book_item(item: dict[str, Any]) -> str | None:
    for k in ("instrument_token", "instrumentToken", "instrument"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _pick_qty_from_book_item(item: dict[str, Any]) -> int | None:
    for k in ("quantity", "qty"):
        v = item.get(k)
        if v is None:
            continue
        try:
            return int(float(v))
        except Exception:
            continue
    return None


def _pick_side_from_book_item(item: dict[str, Any]) -> str | None:
    for k in ("transaction_type", "transactionType", "side"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip().upper()
    return None


def _find_match_in_book(st: OrderState, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    tag = (st.meta.get("tag") or ((st.meta.get("body") or {}) if isinstance(st.meta.get("body"), dict) else {}).get("tag"))
    body = st.meta.get("body") if isinstance(st.meta.get("body"), dict) else {}
    token = body.get("instrument_token")
    try:
        qty_i = int(round(float(st.qty)))
    except Exception:
        qty_i = None
    side = str(st.side or "").upper()

    candidates = items
    if tag:
        candidates = [it for it in candidates if _pick_tag_from_book_item(it) == str(tag)]

    if token:
        candidates2 = [it for it in candidates if _pick_token_from_book_item(it) == str(token)]
        if candidates2:
            candidates = candidates2

    if side:
        candidates2 = [it for it in candidates if (_pick_side_from_book_item(it) or "") == side]
        if candidates2:
            candidates = candidates2

    if qty_i is not None:
        candidates2 = [it for it in candidates if _pick_qty_from_book_item(it) == qty_i]
        if candidates2:
            candidates = candidates2

    for it in candidates:
        if _pick_order_id_from_book_item(it):
            return it
    return None


def _is_terminal_upstox_status(raw: str | None) -> bool:
    s = str(raw or "").strip().upper().replace(" ", "_")
    return s in {
        "COMPLETE",
        "COMPLETED",
        "FILLED",
        "EXECUTED",
        "REJECTED",
        "CANCELLED",
        "CANCELED",
        "FAILED",
        "ERROR",
    }


def _is_ours_tag(tag: str | None) -> bool:
    t = str(tag or "").strip()
    if not t:
        return False
    return t.startswith("agent:") or t.startswith("manual:") or t.startswith("system:")


def _summarize_unknown_book_items(items: list[dict[str, Any]], known_broker_ids: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items:
        oid = _pick_order_id_from_book_item(it)
        if not oid:
            continue
        if oid in known_broker_ids:
            continue

        raw_status = _pick_status_from_book_item(it) or ""
        if _is_terminal_upstox_status(raw_status):
            continue

        tag = _pick_tag_from_book_item(it)
        out.append(
            {
                "order_id": oid,
                "status": str(raw_status),
                "tag": tag,
                "ours": _is_ours_tag(tag),
                "instrument_token": _pick_token_from_book_item(it),
                "side": _pick_side_from_book_item(it),
                "qty": _pick_qty_from_book_item(it),
            }
        )

    # Keep output bounded and put our-tagged items first.
    out.sort(key=lambda x: (not bool(x.get("ours")), str(x.get("order_id"))))
    return out[:50]


def reconcile_upstox_orders(*, order_ids: Iterable[str] | None = None, limit: int = 200) -> dict[str, Any]:
    if not settings.UPSTOX_ACCESS_TOKEN:
        return {"ok": False, "detail": "UPSTOX_ACCESS_TOKEN is not configured"}

    ids: list[str]
    if order_ids is not None:
        ids = [str(x) for x in order_ids if str(x).strip()]
    else:
        # Pull a slice of potentially-active orders.
        with db_conn() as conn:
            cur = conn.execute(
                "SELECT id FROM order_states WHERE broker='upstox' AND status IN ('NEW','SUBMITTED','OPEN','UNKNOWN') ORDER BY ts_created DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            )
            ids = [str(r["id"]) for r in cur.fetchall()]

    if not ids:
        return {"ok": True, "checked": 0, "updated": 0, "errors": []}

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client: UpstoxClient | None = None
    checked = 0
    updated = 0
    errors: list[str] = []

    unknown_open: list[dict[str, Any]] = []

    try:
        client = UpstoxClient(cfg)

        # For crash recovery, we may need to discover missing broker_order_id values.
        # Fetch the order book once and match on tag/body fields when possible.
        try:
            book = client.order_book_v2()
            book_items = _book_items(book)
        except Exception:
            book_items = []

        # Unknown open orders in broker book (not in our DB).
        if book_items:
            with db_conn() as conn:
                cur = conn.execute(
                    "SELECT broker_order_id FROM order_states WHERE broker='upstox' AND broker_order_id IS NOT NULL AND broker_order_id <> ''"
                )
                known_ids = {str(r["broker_order_id"]).strip() for r in cur.fetchall() if str(r["broker_order_id"]).strip()}
            unknown_open = _summarize_unknown_book_items(book_items, known_ids)

        for oid in ids:
            st = get_order(oid)
            if st is None:
                continue

            # If we don't have a broker_order_id, attempt discovery from order book.
            if not st.broker_order_id:
                match = _find_match_in_book(st, book_items)
                if match:
                    found_id = _pick_order_id_from_book_item(match)
                    raw_status = _pick_status_from_book_item(match) or ""
                    norm = _normalize_upstox_status(raw_status)
                    update_order(
                        oid,
                        broker_order_id=found_id,
                        status=(None if norm == "UNKNOWN" else norm),
                        meta_patch={"upstox": {"book_match": match, "raw_status": raw_status}},
                    )
                    updated += 1
                    st = get_order(oid) or st

            if not st.broker_order_id:
                continue

            checked += 1
            try:
                details = client.order_details_v2(str(st.broker_order_id))
                data = details.get("data") or {}
                raw_status = data.get("status") or data.get("order_status") or ""
                norm = _normalize_upstox_status(str(raw_status))

                update_order(
                    oid,
                    status=(None if norm == "UNKNOWN" else norm),
                    meta_patch={"upstox": {"raw_status": str(raw_status), "details": data}},
                )
                updated += 1
            except UpstoxError as e:
                update_order(oid, status="ERROR", last_error=str(e)[:500], meta_patch={"reconcile": {"error": str(e)[:200]}})
                errors.append(f"{oid}:{e}")
            except Exception as e:
                update_order(oid, status="ERROR", last_error=str(e)[:500], meta_patch={"reconcile": {"error": str(e)[:200]}})
                errors.append(f"{oid}:{e}")

    finally:
        if client is not None:
            client.close()

    return {
        "ok": True,
        "checked": checked,
        "updated": updated,
        "errors": errors[:20],
        "unknown_open_count": len(unknown_open),
        "unknown_open": unknown_open,
    }
