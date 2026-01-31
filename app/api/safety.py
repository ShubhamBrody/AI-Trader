from __future__ import annotations

from fastapi import APIRouter

from app.core.guards import get_safety_status
from app.core.settings import settings

router = APIRouter(prefix="/safety")


@router.get("/status")
def status() -> dict:
    s = get_safety_status()
    return {
        "safe_mode": settings.SAFE_MODE,
        **s.__dict__,
    }


@router.get("/context")
def context() -> dict:
    """Frontend compatibility endpoint.

    The UI expects: market, session, can_trade, read_only.
    """

    s = get_safety_status()
    return {
        "market": "NSE",
        "session": s.market_state,
        "ist_time": s.now_ist,
        "trading_day": s.trading_day,
        "read_only": bool(settings.SAFE_MODE),
        # For UI purposes, SAFE_MODE should behave like "can't trade".
        "can_trade": (not s.trade_lock) and (not settings.SAFE_MODE),
        "trade_lock": s.trade_lock,
        "reason": s.reason,
        "safe_mode": bool(settings.SAFE_MODE),
    }
