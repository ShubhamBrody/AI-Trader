from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from app.core.settings import settings
from app.markets.nse.calendar import (
    MarketSession,
    is_trading_day,
    load_holidays,
    load_market_session,
    market_state,
)


@dataclass(frozen=True)
class SafetyStatus:
    now_ist: str
    trading_day: bool
    market_state: str
    trade_lock: bool
    reason: str


def get_safety_status(now_utc: datetime | None = None) -> SafetyStatus:
    now_utc = now_utc or datetime.now(timezone.utc)

    tz = ZoneInfo(settings.TIMEZONE)
    now_local = now_utc.astimezone(tz)

    session: MarketSession = load_market_session("app/config/market_hours.yaml")
    holidays = load_holidays("app/config/nse_holidays.yaml")

    trading_day = is_trading_day(now_local.date(), holidays)
    state = market_state(now_local, session, trading_day)

    # Hard lock outside LIVE window
    lock = state != "LIVE"

    reason = "OK"
    if not trading_day:
        reason = "Not a trading day"
    elif state == "PRE_MARKET":
        reason = "Pre-market: trading locked"
    elif state == "POST_MARKET":
        reason = "Post-market: trading locked"
    elif state == "CLOSED":
        reason = "Market closed"

    # Global safe mode additionally locks live trading execution.
    if settings.SAFE_MODE:
        # We do not lock strategy computation; only live broker execution should respect SAFE_MODE.
        pass

    return SafetyStatus(
        now_ist=now_local.isoformat(),
        trading_day=trading_day,
        market_state=state,
        trade_lock=lock,
        reason=reason,
    )


def assert_trade_allowed() -> None:
    status = get_safety_status()
    if status.trade_lock:
        raise PermissionError(status.reason)
