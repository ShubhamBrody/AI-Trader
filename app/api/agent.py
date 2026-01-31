from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.agent.service import AGENT_SINGLETON
from app.agent.state import get_trade, list_open_trades, list_recent_trades, realized_pnl_today
from app.core.settings import settings

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/status")
def status() -> dict:
    return AGENT_SINGLETON.status()


@router.post("/start")
async def start() -> dict:
    res = await AGENT_SINGLETON.start()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res


@router.post("/stop")
async def stop() -> dict:
    return await AGENT_SINGLETON.stop()


@router.post("/run-once")
async def run_once() -> dict:
    return await AGENT_SINGLETON.run_once()


@router.post("/flatten")
async def flatten() -> dict:
    res = await AGENT_SINGLETON.flatten()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res


@router.post("/cancel-open-orders")
async def cancel_open_orders() -> dict:
    res = await AGENT_SINGLETON.cancel_open_orders()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res


@router.get("/trades/open")
def trades_open(limit: int = 200) -> dict:
    return {"ok": True, "trades": list_open_trades(limit=limit)}


@router.get("/trades/recent")
def trades_recent(limit: int = 200, include_open: bool = True) -> dict:
    return {"ok": True, "trades": list_recent_trades(limit=limit, include_open=include_open)}


@router.get("/trades/{trade_id}")
def trade_by_id(trade_id: int) -> dict:
    t = get_trade(int(trade_id))
    if not t:
        raise HTTPException(status_code=404, detail="trade not found")
    return {"ok": True, "trade": t}


@router.get("/performance/today")
def performance_today() -> dict:
    pnl = float(realized_pnl_today(tz_name=settings.TIMEZONE))
    return {"ok": True, "pnl_realized": pnl, "currency": "INR"}


@router.get("/preview")
async def preview(instrument_key: str) -> dict:
    return await AGENT_SINGLETON.preview(instrument_key)
