from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from app.core.settings import settings
from app.core.db import db_conn
from app.agent.state import list_open_trades
from app.hft.index_options.service import HFT_INDEX_OPTIONS
from app.hft.index_options.selector import list_option_chain, find_nearest_future
from app.candles.service import CandleService
from app.realtime.bus import publish_sync
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError

router = APIRouter(prefix="/hft/index-options", tags=["hft"])


def _last_close_from_db(instrument_key: str, interval: str = "1m") -> float | None:
    if not instrument_key:
        return None
    try:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT close FROM candles WHERE instrument_key=? AND interval=? ORDER BY ts DESC LIMIT 1",
                (str(instrument_key), str(interval)),
            ).fetchone()
            if row is None:
                return None
            return float(row["close"]) if row["close"] is not None else None
    except Exception:
        return None


def _upstox_last_price_map(instrument_keys: list[str]) -> dict[str, float]:
    """Best-effort live last_price for instrument_key -> ltp.

    Uses Upstox market quote endpoint. Returns {} on any error.
    """

    keys = [str(k).strip() for k in (instrument_keys or []) if str(k).strip()]
    if not keys:
        return {}

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client: UpstoxClient | None = None
    try:
        client = UpstoxClient(cfg)
        payload = client.market_quote_quotes_v2(keys)
    except Exception:
        return {}
    finally:
        if client is not None:
            client.close()

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return {}

    out: dict[str, float] = {}
    for v in data.values():
        if not isinstance(v, dict):
            continue
        ik = str(v.get("instrument_token") or "").strip()
        if not ik:
            continue
        lp = v.get("last_price")
        if lp is None:
            continue
        try:
            out[ik] = float(lp)
        except Exception:
            continue
    return out


class TradeRiskUpdate(BaseModel):
    stop: float | None = None
    target: float | None = None


class BrokerSwitch(BaseModel):
    broker: Literal["paper", "upstox"]


@router.get("/status")
def status() -> dict[str, Any]:
    return {"ok": True, **HFT_INDEX_OPTIONS.status()}


@router.post("/start")
async def start() -> dict[str, Any]:
    res = await HFT_INDEX_OPTIONS.start()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res


@router.post("/stop")
async def stop() -> dict[str, Any]:
    return await HFT_INDEX_OPTIONS.stop()


@router.post("/run-once")
async def run_once(force: bool = False) -> dict[str, Any]:
    if not bool(getattr(settings, "INDEX_OPTIONS_HFT_ENABLED", False)):
        raise HTTPException(status_code=400, detail="INDEX_OPTIONS_HFT_ENABLED=false")
    return await HFT_INDEX_OPTIONS.run_once(force_offmarket_paper=bool(force))


@router.post("/flatten")
async def flatten() -> dict[str, Any]:
    return await HFT_INDEX_OPTIONS.flatten()


@router.get("/broker")
def get_broker() -> dict[str, Any]:
    return {"ok": True, "broker": str(HFT_INDEX_OPTIONS.broker())}


@router.post("/broker")
def set_broker(body: BrokerSwitch) -> dict[str, Any]:
    res = HFT_INDEX_OPTIONS.set_broker(str(body.broker), actor="api")
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("detail") or res)
    return res


@router.get("/open-trades")
def open_trades(limit: int = 200, origin: str | None = "hft_index_options") -> dict[str, Any]:
    """Return open agent trades, optionally filtered by meta.origin."""

    rows = list_open_trades(limit=int(limit))
    if origin:
        want = str(origin)
        rows = [t for t in rows if str((t.get("meta") or {}).get("origin") or "") == want]
    return {"ok": True, "trades": rows}


@router.post("/trade/{trade_id}/risk")
def update_trade_risk(trade_id: int, body: TradeRiskUpdate) -> dict[str, Any]:
    """Update TP/SL for an open HFT trade.

    Note: this updates the app's trade record (agent_trades). If you are using a live broker,
    you still need broker-side order modification support.
    """

    stop = body.stop
    target = body.target
    if stop is None and target is None:
        raise HTTPException(status_code=400, detail={"detail": "provide stop and/or target"})

    with db_conn() as conn:
        row = conn.execute(
            "SELECT status, meta_json FROM agent_trades WHERE id=?",
            (int(trade_id),),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail={"detail": "trade not found"})
        if str(row["status"]) != "OPEN":
            raise HTTPException(status_code=400, detail={"detail": "trade not open"})

        # Best-effort origin check: only allow updating HFT trades.
        try:
            import json

            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}
        origin = str((meta or {}).get("origin") or "")
        if origin not in {"hft_index_options", "hft_manual"}:
            raise HTTPException(status_code=400, detail={"detail": "risk update allowed only for HFT trades", "origin": origin})

        # Keep existing values when not provided.
        cur2 = conn.execute("SELECT stop, target FROM agent_trades WHERE id=?", (int(trade_id),)).fetchone()
        cur_stop = float(cur2["stop"]) if cur2 is not None else 0.0
        cur_target = float(cur2["target"]) if cur2 is not None else 0.0
        new_stop = cur_stop if stop is None else float(stop)
        new_target = cur_target if target is None else float(target)
        if new_stop <= 0 or new_target <= 0:
            raise HTTPException(status_code=400, detail={"detail": "stop/target must be > 0"})

        conn.execute(
            "UPDATE agent_trades SET stop=?, target=? WHERE id=?",
            (float(new_stop), float(new_target), int(trade_id)),
        )

    publish_sync("hft", "hft.trade_risk", {"trade_id": int(trade_id), "stop": float(new_stop), "target": float(new_target)})
    return {"ok": True, "trade_id": int(trade_id), "stop": float(new_stop), "target": float(new_target)}


@router.get("/options-chain")
def options_chain(underlying: str = "NIFTY", count: int = 7) -> dict[str, Any]:
    """Compact chain around ATM for nearest expiry.

    Returns instrument identifiers and best-effort last close from local DB cache.
    """

    # Spot for ATM selection: use the spot query candle if possible.
    sym = str(underlying or "").strip().upper()
    if sym not in {"NIFTY", "SENSEX"}:
        raise HTTPException(status_code=400, detail={"detail": "unsupported underlying", "underlying": sym})

    spot_query = getattr(settings, f"INDEX_OPTIONS_HFT_{sym}_SPOT_QUERY", sym)
    spot_key = CandleService._resolve_instrument_key(str(spot_query))

    # Prefer DB cache (fast). If missing, try Upstox quote (best-effort).
    spot = _last_close_from_db(spot_key, "1m") or _last_close_from_db(spot_key, "5m") or 0.0
    if spot <= 0:
        spot_map = _upstox_last_price_map([spot_key])
        spot = float(spot_map.get(spot_key) or 0.0)
    if spot <= 0:
        raise HTTPException(status_code=400, detail={"detail": "spot unavailable", "spot_instrument_key": spot_key})

    expiry, rows = list_option_chain(underlying_symbol=sym, spot=float(spot), strikes_each_side=int(count), max_expiry_days=int(getattr(settings, "INDEX_OPTIONS_HFT_MAX_EXPIRY_DAYS", 14)))

    # Best-effort live LTP for the chain (independent of trading mode).
    # Falls back to DB if quote call fails.
    want_keys: list[str] = []
    for r in rows:
        if r.ce:
            want_keys.append(str(r.ce.instrument_key))
        if r.pe:
            want_keys.append(str(r.pe.instrument_key))
    live_prices = _upstox_last_price_map(want_keys)

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        ce = None
        if r.ce:
            ce_key = str(r.ce.instrument_key)
            ce = {
                "instrument_key": ce_key,
                "tradingsymbol": r.ce.tradingsymbol,
                "strike": r.ce.strike,
                "option_type": "CE",
                "ltp": (live_prices.get(ce_key) if live_prices else None) or _last_close_from_db(ce_key, "1m"),
            }
        pe = None
        if r.pe:
            pe_key = str(r.pe.instrument_key)
            pe = {
                "instrument_key": pe_key,
                "tradingsymbol": r.pe.tradingsymbol,
                "strike": r.pe.strike,
                "option_type": "PE",
                "ltp": (live_prices.get(pe_key) if live_prices else None) or _last_close_from_db(pe_key, "1m"),
            }
        out_rows.append({"strike": float(r.strike), "ce": ce, "pe": pe})

    return {
        "ok": True,
        "underlying": sym,
        "spot_instrument_key": spot_key,
        "spot": float(spot),
        "expiry": (None if expiry is None else str(expiry)),
        "rows": out_rows,
    }


@router.get("/futures")
def futures(underlying: str = "NIFTY") -> dict[str, Any]:
    sym = str(underlying or "").strip().upper()
    if sym not in {"NIFTY", "SENSEX"}:
        raise HTTPException(status_code=400, detail={"detail": "unsupported underlying", "underlying": sym})

    fut = find_nearest_future(underlying_symbol=sym)
    if fut is None:
        return {"ok": True, "underlying": sym, "contract": None}
    return {
        "ok": True,
        "underlying": sym,
        "contract": {
            "instrument_key": fut.instrument_key,
            "tradingsymbol": fut.tradingsymbol,
            "expiry": (None if fut.expiry is None else str(fut.expiry)),
            "ltp": _last_close_from_db(fut.instrument_key, "1m"),
        },
    }
