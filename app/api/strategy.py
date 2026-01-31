from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException

from app.candles.service import CandleService
from app.core.settings import settings
from app.markets.nse.calendar import last_n_trading_days, load_holidays, load_market_session
from app.strategy.engine import StrategyEngine

router = APIRouter(prefix="/strategy", tags=["strategy"])


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


@router.get("")
def strategy_signal(instrument_key: str, interval: str = "1m", lookback: int = 220) -> dict:
    """Returns a deterministic strategy signal used by the frontend.

    Response keys are chosen to match the UI: action, confidence, entry, stop_loss, target, regime.
    """

    ik = str(instrument_key or "").strip()
    if not ik:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    lookback = max(60, min(int(lookback), 5000))

    # Pull data from DB; auto-warm from adapters when empty.
    svc = CandleService()

    now = datetime.now(timezone.utc)

    # Heuristic window: aim to cover the required lookback with slack for gaps.
    interval_s = str(interval).lower().strip()
    if interval_s.endswith("m"):
        try:
            minutes = int(interval_s[:-1])
        except Exception:
            minutes = 1
        span = timedelta(minutes=max(60, minutes * lookback * 2))
    elif interval_s.endswith("h"):
        try:
            hours = int(interval_s[:-1])
        except Exception:
            hours = 1
        span = timedelta(hours=max(12, hours * max(1, lookback // 10)))
    else:
        # daily/unknown
        span = timedelta(days=max(30, max(1, lookback // 5)))

    start = now - span

    candles: list = []
    # Prefer historical range (auto-warms). If it is still empty and market is live, try a short poll.
    try:
        series = svc.get_historical(ik, interval, start=start, end=now, limit=lookback)
        candles = list(series.candles or [])
    except Exception:
        candles = []

    if len(candles) < 60:
        try:
            series = svc.poll_intraday(ik, interval, lookback_minutes=int(span.total_seconds() // 60))
            candles = list(series.candles or [])
        except Exception:
            pass

    # Market-closed fallback: on weekends/holidays, "today" intraday endpoints can return 0.
    # In that case, fetch the last trading session (IST market hours) so the UI still works.
    if len(candles) < 60 and str(interval).lower().strip() != "1d":
        try:
            tz = ZoneInfo(settings.TIMEZONE)
            now_local = datetime.now(timezone.utc).astimezone(tz)

            holidays = load_holidays("app/config/nse_holidays.yaml")
            last_day = last_n_trading_days(now_local.date(), 1, holidays)[0]
            session = load_market_session("app/config/market_hours.yaml")
            session_tz = ZoneInfo(session.tz)

            start_local = datetime.combine(last_day, session.open_time, tzinfo=session_tz)
            end_local = datetime.combine(last_day, session.close_time, tzinfo=session_tz)

            series = svc.load_historical(ik, interval, start=start_local, end=end_local)
            candles = list(series.candles or [])
        except Exception:
            pass

    if len(candles) < 60:
        # Do not hard-fail the UI; return a deterministic HOLD with context.
        return {
            "instrument_key": ik,
            "interval": str(interval),
            "action": "HOLD",
            "confidence": 0.10,
            "entry": None,
            "stop_loss": None,
            "target": None,
            "rr": None,
            "regime": "unknown",
            "reason": "not_enough_candles",
            "detail": {"required": 60, "available": len(candles)},
        }

    candles = candles[-lookback:]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    closes = [float(c.close) for c in candles]

    engine = StrategyEngine()
    idea = engine.build_idea(symbol=ik, highs=highs, lows=lows, closes=closes)

    action = str(idea.side).upper()

    # Best-effort confidence heuristic.
    if action == "HOLD":
        confidence = 0.30
    else:
        # Use R/R as signal strength.
        rr = float(idea.rr or 0.0)
        confidence = 0.55 + _clamp((rr - 1.0) / 4.0, 0.0, 0.40)

    # Regime is a friendly label for the UI.
    regime = "trend" if action in {"BUY", "SELL"} else "neutral"

    return {
        "instrument_key": ik,
        "interval": str(interval),
        "action": action,
        "confidence": _clamp(float(confidence), 0.0, 0.99),
        "entry": float(idea.entry),
        "stop_loss": float(idea.stop_loss),
        "target": float(idea.target),
        "rr": float(idea.rr),
        "regime": regime,
        "reason": idea.reason,
    }


@router.get("/decision")
def trade_decision(
    instrument_key: str,
    interval: str = "1m",
    account_balance: float = 100000,
    lot_size: int = 1,
    risk_fraction: float = 0.01,
) -> dict:
    """Position sizing for the UI.

    Uses the strategy signal's entry/stop to size by a fixed risk fraction of account_balance.
    """

    signal = strategy_signal(instrument_key=instrument_key, interval=interval)

    # If we don't have actionable prices, return a 0-qty decision.
    if signal.get("entry") is None or signal.get("stop_loss") is None:
        return {
            **signal,
            "qty": 0,
            "lot_size": int(max(1, int(lot_size))),
            "risk_budget": 0.0,
            "risk_per_share": None,
            "notional": 0.0,
            "detail": signal.get("detail") or "missing entry/stop for sizing",
        }

    entry = float(signal["entry"])
    stop = float(signal["stop_loss"])

    risk_per_share = abs(entry - stop)
    if not math.isfinite(risk_per_share) or risk_per_share <= 0:
        return {
            **signal,
            "qty": 0,
            "lot_size": int(lot_size),
            "risk_budget": 0.0,
            "risk_per_share": risk_per_share,
            "notional": 0.0,
            "detail": "invalid stop/entry for sizing",
        }

    bal = float(account_balance)
    risk_budget = max(0.0, bal * float(risk_fraction))

    raw_qty = int(risk_budget // risk_per_share) if risk_per_share > 0 else 0
    ls = max(1, int(lot_size))
    qty = (raw_qty // ls) * ls

    notional = qty * entry

    return {
        **signal,
        "qty": int(qty),
        "lot_size": int(ls),
        "risk_budget": float(risk_budget),
        "risk_per_share": float(risk_per_share),
        "notional": float(notional),
    }
