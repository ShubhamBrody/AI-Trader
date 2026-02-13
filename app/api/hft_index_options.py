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

    # IMPORTANT: keep this endpoint non-blocking and fast.
    # Do NOT call poll_intraday() here (it may hit broker/network and hang).
    spot = _last_close_from_db(spot_key, "1m") or _last_close_from_db(spot_key, "5m") or 0.0
    if spot <= 0:
        raise HTTPException(
            status_code=400,
            detail={"detail": "spot unavailable; warm candles cache first", "spot_instrument_key": spot_key},
        )

    expiry, rows = list_option_chain(underlying_symbol=sym, spot=float(spot), strikes_each_side=int(count), max_expiry_days=int(getattr(settings, "INDEX_OPTIONS_HFT_MAX_EXPIRY_DAYS", 14)))

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        ce = None
        if r.ce:
            ce = {
                "instrument_key": r.ce.instrument_key,
                "tradingsymbol": r.ce.tradingsymbol,
                "strike": r.ce.strike,
                "option_type": "CE",
                "ltp": _last_close_from_db(r.ce.instrument_key, "1m"),
            }
        pe = None
        if r.pe:
            pe = {
                "instrument_key": r.pe.instrument_key,
                "tradingsymbol": r.pe.tradingsymbol,
                "strike": r.pe.strike,
                "option_type": "PE",
                "ltp": _last_close_from_db(r.pe.instrument_key, "1m"),
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
