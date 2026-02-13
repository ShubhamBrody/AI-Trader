from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel

import httpx

from app.core.db import db_conn
from app.universe.service import UniverseService

router = APIRouter(prefix="/universe", tags=["universe"])
svc = UniverseService()


def _normalize_expiry(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    # Upstox exchange files often encode expiry as epoch millis.
    try:
        if s.isdigit() and len(s) >= 10:
            n = int(s)
            ts = (n / 1000.0) if n >= 1_000_000_000_000 else float(n)
            return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
    except Exception:
        pass
    return s


def _normalize_option_type(it: dict) -> str | None:
    raw = str(it.get("option_type") or "").strip().upper()
    if raw in {"CE", "PE"}:
        return raw
    if raw in {"CALL", "C"}:
        return "CE"
    if raw in {"PUT", "P"}:
        return "PE"

    inst = str(it.get("instrument_type") or "").strip().upper()
    if inst in {"CE", "PE"}:
        return inst
    return None


class UniverseUpsert(BaseModel):
    instrument_key: str
    tradingsymbol: str | None = None
    cap_tier: str | None = None  # large|mid|small|unknown
    upstox_token: str | None = None


@router.get("")
def list_items(cap_tier: str | None = None, limit: int = 500) -> dict:
    return {"items": svc.list(cap_tier=cap_tier, limit=limit)}


@router.post("")
def upsert(req: UniverseUpsert) -> dict:
    return svc.upsert(
        instrument_key=req.instrument_key,
        tradingsymbol=req.tradingsymbol,
        cap_tier=req.cap_tier,
        upstox_token=req.upstox_token,
    )


@router.post("/import")
def bulk_import(items: list[UniverseUpsert]) -> dict:
    return svc.bulk_import([i.model_dump() for i in items])


@router.post("/import-upstox-nse-eq")
def import_upstox_nse_eq(limit: int = 0) -> dict:
    """Import Upstox BOD instruments for NSE equities into instrument_meta.

    Source: https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz

    Filters to:
    - segment == "NSE_EQ"
    - instrument_type == "EQ"

    Stores:
    - instrument_key
    - tradingsymbol (trading_symbol)
    - upstox_token (exchange_token)

    Note: cap_tier remains "unknown" unless you set it separately.
    """

    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        r = httpx.get(url, timeout=60.0)
        r.raise_for_status()
        raw = gzip.decompress(r.content)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=502, detail={"ok": False, "reason": "failed to fetch/parse instruments", "error": str(e)})

    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail={"ok": False, "reason": "unexpected instruments format"})

    items: list[dict] = []
    extras: list[dict] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        if str(it.get("segment") or "") != "NSE_EQ":
            continue
        if str(it.get("instrument_type") or "") != "EQ":
            continue

        instrument_key = str(it.get("instrument_key") or "").strip()
        if not instrument_key.startswith("NSE_EQ|"):
            continue

        items.append(
            {
                "instrument_key": instrument_key,
                "tradingsymbol": (str(it.get("trading_symbol") or "").strip() or None),
                "cap_tier": "unknown",
                "upstox_token": (str(it.get("exchange_token") or "").strip() or None),
            }
        )

        extras.append(
            {
                "instrument_key": instrument_key,
                "exchange": str(it.get("exchange") or "NSE"),
                "segment": str(it.get("segment") or "NSE_EQ"),
                "instrument_type": str(it.get("instrument_type") or "EQ"),
                "name": (str(it.get("name") or "").strip() or None),
                "underlying_symbol": (str(it.get("underlying_symbol") or "").strip() or None),
                "expiry": _normalize_expiry(it.get("expiry")),
                "strike": (float(it.get("strike_price")) if it.get("strike_price") not in (None, "") else None),
                "option_type": _normalize_option_type(it),
                "raw_json": json.dumps(it, ensure_ascii=False),
            }
        )
        if int(limit) > 0 and len(items) >= int(limit):
            break

    res = svc.bulk_import(items)
    try:
        ts = int(datetime.now(timezone.utc).timestamp())
        with db_conn() as conn:
            for ex in extras:
                conn.execute(
                    "INSERT INTO instrument_extra (instrument_key,exchange,segment,instrument_type,name,underlying_symbol,expiry,strike,option_type,raw_json,updated_ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(instrument_key) DO UPDATE SET "
                    "exchange=excluded.exchange,segment=excluded.segment,instrument_type=excluded.instrument_type,name=excluded.name,"
                    "underlying_symbol=excluded.underlying_symbol,expiry=excluded.expiry,strike=excluded.strike,option_type=excluded.option_type,raw_json=excluded.raw_json,updated_ts=excluded.updated_ts",
                    (
                        ex["instrument_key"],
                        ex.get("exchange"),
                        ex.get("segment"),
                        ex.get("instrument_type"),
                        ex.get("name"),
                        ex.get("underlying_symbol"),
                        ex.get("expiry"),
                        ex.get("strike"),
                        ex.get("option_type"),
                        ex.get("raw_json"),
                        ts,
                    ),
                )
    except Exception:
        pass

    return {"ok": True, "source": url, "parsed": len(items), **res}


@router.post("/import-upstox-exchange")
def import_upstox_exchange(exchange: str, limit: int = 0) -> dict:
    """Import Upstox BOD instruments for a given exchange into instrument_meta + instrument_extra.

    Examples:
    - exchange=NSE (equities, derivatives)
    - exchange=MCX (commodities like SILVER)

    Source pattern:
    https://assets.upstox.com/market-quote/instruments/exchange/{EXCHANGE}.json.gz

    Note: we do not filter to EQ here; we import whatever is present for better search.
    """

    ex = str(exchange or "").strip().upper()
    if not ex:
        raise HTTPException(status_code=400, detail="exchange is required")

    url = f"https://assets.upstox.com/market-quote/instruments/exchange/{ex}.json.gz"
    try:
        r = httpx.get(url, timeout=60.0)
        r.raise_for_status()
        raw = gzip.decompress(r.content)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=502, detail={"ok": False, "reason": "failed to fetch/parse instruments", "error": str(e)})

    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail={"ok": False, "reason": "unexpected instruments format"})

    items: list[dict] = []
    extras: list[dict] = []
    for it in data:
        if not isinstance(it, dict):
            continue

        instrument_key = str(it.get("instrument_key") or "").strip()
        if not instrument_key:
            continue

        items.append(
            {
                "instrument_key": instrument_key,
                "tradingsymbol": (str(it.get("trading_symbol") or "").strip() or None),
                "cap_tier": "unknown",
                "upstox_token": (str(it.get("exchange_token") or "").strip() or None),
            }
        )

        extras.append(
            {
                "instrument_key": instrument_key,
                "exchange": str(it.get("exchange") or ex),
                "segment": (str(it.get("segment") or "").strip() or None),
                "instrument_type": (str(it.get("instrument_type") or "").strip() or None),
                "name": (str(it.get("name") or "").strip() or None),
                "underlying_symbol": (str(it.get("underlying_symbol") or "").strip() or None),
                "expiry": _normalize_expiry(it.get("expiry")),
                "strike": (float(it.get("strike_price")) if it.get("strike_price") not in (None, "") else None),
                "option_type": _normalize_option_type(it),
                "raw_json": json.dumps(it, ensure_ascii=False),
            }
        )

        if int(limit) > 0 and len(items) >= int(limit):
            break

    res = svc.bulk_import(items)
    try:
        ts = int(datetime.now(timezone.utc).timestamp())
        with db_conn() as conn:
            for exr in extras:
                conn.execute(
                    "INSERT INTO instrument_extra (instrument_key,exchange,segment,instrument_type,name,underlying_symbol,expiry,strike,option_type,raw_json,updated_ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(instrument_key) DO UPDATE SET "
                    "exchange=excluded.exchange,segment=excluded.segment,instrument_type=excluded.instrument_type,name=excluded.name,"
                    "underlying_symbol=excluded.underlying_symbol,expiry=excluded.expiry,strike=excluded.strike,option_type=excluded.option_type,raw_json=excluded.raw_json,updated_ts=excluded.updated_ts",
                    (
                        exr["instrument_key"],
                        exr.get("exchange"),
                        exr.get("segment"),
                        exr.get("instrument_type"),
                        exr.get("name"),
                        exr.get("underlying_symbol"),
                        exr.get("expiry"),
                        exr.get("strike"),
                        exr.get("option_type"),
                        exr.get("raw_json"),
                        ts,
                    ),
                )
    except Exception:
        pass

    return {"ok": True, "source": url, "parsed": len(items), **res}
