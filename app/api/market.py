from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from app.core.guards import get_safety_status

router = APIRouter(prefix="/market")


@router.get("/session")
def session() -> dict:
    status = get_safety_status()
    return {
        "now_ist": status.now_ist,
        "trading_day": status.trading_day,
        "market_state": status.market_state,
        "open_for_trading": not status.trade_lock,
    }


@router.get("/status")
def status() -> dict:
    """Frontend compatibility endpoint.

    The UI expects: market, session, ist_time, trading_day, can_trade.
    """

    s = get_safety_status()
    return {
        "market": "NSE",
        "session": s.market_state,
        "ist_time": s.now_ist,
        "trading_day": s.trading_day,
        "can_trade": (not s.trade_lock),
        "reason": s.reason,
    }
