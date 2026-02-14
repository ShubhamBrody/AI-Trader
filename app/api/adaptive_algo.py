from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from zoneinfo import ZoneInfo

from app.ai.adaptive_volume_algo import AlgoParams, compute_signal, optimize, trend_table
from app.candles.service import CandleService
from app.core.settings import settings
from app.core.guards import get_safety_status
from app.markets.nse.calendar import last_n_trading_days, load_holidays

router = APIRouter(prefix="/adaptive-algo")
svc = CandleService()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Accept ISO8601. If no tz, treat as UTC.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


@router.get("")
def adaptive_algo(
    instrument_key: str = Query(...),
    interval: str = Query("5m"),
    start: str | None = Query(None, description="Optional ISO start; if provided with end, uses historical window."),
    end: str | None = Query(None, description="Optional ISO end; if provided with start, uses historical window."),
    lookback: int = Query(500, ge=60, le=5000),
    period: int = Query(40, ge=10, le=200),
    multiplier: float = Query(2.0, ge=0.5, le=5.0),
    z_threshold: float = Query(0.8, ge=0.0, le=5.0),
    atr_period: int = Query(14, ge=5, le=60),
    rr: float = Query(2.0, ge=0.5, le=6.0),
    horizon_steps: int = Query(12, ge=1, le=240),
    non_repainting: bool = Query(True, description="If true, compute on last closed candle."),
    do_optimize: bool = Query(True, description="If true, run the 24-combo optimizer."),
    trend_intervals: str = Query("1m,5m,15m,1h,1d", description="Comma-separated intervals for trend table."),
) -> dict:
    instrument_key = str(instrument_key)
    interval = str(interval)

    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)

    now_utc = datetime.now(timezone.utc)
    status = get_safety_status(now_utc)

    # Candle window selection:
    if start_dt and end_dt:
        series = svc.get_historical(instrument_key, interval, start_dt, end_dt)
        candles = list(series.candles or [])
    else:
        # If live, use last 6h; else use last trading session.
        if status.market_state == "LIVE":
            end2 = now_utc
            start2 = end2 - timedelta(hours=6)
            series = svc.get_historical(instrument_key, interval, start2, end2)
            candles = list(series.candles or [])
        else:
            tz = ZoneInfo(settings.TIMEZONE)
            holidays = load_holidays("app/config/nse_holidays.yaml")
            last_day = last_n_trading_days(now_utc.astimezone(tz).date(), 1, holidays)[0]
            start2 = datetime(last_day.year, last_day.month, last_day.day, 9, 15, tzinfo=tz)
            end2 = datetime(last_day.year, last_day.month, last_day.day, 15, 29, tzinfo=tz)
            series = svc.get_historical(instrument_key, interval, start2, end2)
            candles = list(series.candles or [])

    # Limit lookback to keep compute bounded.
    if lookback and len(candles) > int(lookback):
        candles = candles[-int(lookback) :]

    params = AlgoParams(
        period=int(period),
        multiplier=float(multiplier),
        z_threshold=float(z_threshold),
        atr_period=int(atr_period),
        rr=float(rr),
        horizon_steps=int(horizon_steps),
        use_last_closed=bool(non_repainting),
    )

    sig = compute_signal(candles, params)

    opt = None
    if bool(do_optimize):
        opt = optimize(candles, params)

    # Trend table: pull multi-interval candles using same end timestamp.
    trend_map: dict[str, list] = {}
    intervals = [s.strip() for s in str(trend_intervals).split(",") if s.strip()]

    for iv in intervals:
        try:
            # Fetch enough history for EMA50.
            end_for_tf = end_dt or now_utc
            start_for_tf = end_for_tf - timedelta(days=12)
            s2 = svc.get_historical(instrument_key, iv, start_for_tf, end_for_tf)
            cs2 = list(s2.candles or [])
            if len(cs2) > 500:
                cs2 = cs2[-500:]
            trend_map[iv] = cs2
        except Exception:
            trend_map[iv] = []

    trends = trend_table(trend_map, fast=20, slow=50, use_last_closed=bool(non_repainting))

    # asof_ts
    asof_ts = None
    try:
        idx = int(sig.get("asof_index") or 0)
        if idx >= 0 and idx < len(candles):
            asof_ts = int(getattr(candles[idx], "ts"))
    except Exception:
        asof_ts = None

    return {
        "ok": True,
        "instrument_key": instrument_key,
        "interval": interval,
        "market_state": status.market_state,
        "asof_ts": asof_ts,
        "params": {
            "period": params.period,
            "multiplier": params.multiplier,
            "z_threshold": params.z_threshold,
            "atr_period": params.atr_period,
            "rr": params.rr,
            "horizon_steps": params.horizon_steps,
            "non_repainting": bool(non_repainting),
        },
        "signal": sig,
        "optimization": opt,
        "trends": trends,
    }
