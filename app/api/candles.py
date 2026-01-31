from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.candles.models import BulkLoadRequest, CandleSeries, LoadHistoricalRequest
from app.candles.service import CandleService
from app.core.guards import get_safety_status
from app.integrations.upstox.client import UpstoxError

router = APIRouter(prefix="/candles")
svc = CandleService()


def _as_candle_list(series: CandleSeries, *, limit: int) -> list[dict]:
    candles = series.candles or []
    if limit > 0 and len(candles) > limit:
        candles = candles[-limit:]

    out: list[dict] = []
    for c in candles:
        ts = int(getattr(c, "ts"))
        out.append(
            {
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            }
        )
    return out


def _maybe_auth_401(e: Exception) -> JSONResponse | None:
    msg = str(e)
    if "Open /api/auth/upstox/login" in msg or "Upstox is not authenticated" in msg:
        return JSONResponse(status_code=401, content={"detail": msg, "login_url": "/api/auth/upstox/login"})
    if "Upstox HTTP 401" in msg:
        return JSONResponse(status_code=401, content={"detail": msg, "login_url": "/api/auth/upstox/login"})
    return None


@router.post("/historical/load")
def load_historical(
    instrument_key: str = Query(...),
    interval: str = Query("1d"),
    start: datetime = Query(...),
    end: datetime = Query(...),
) -> dict:
    try:
        series = svc.load_historical(instrument_key, interval, start, end)
        return {"status": "ok", "cached": len(series.candles or [])}
    except (UpstoxError, Exception) as e:
        auth = _maybe_auth_401(e)
        if auth is not None:
            return auth
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/historical/load/series", response_model=CandleSeries)
def load_historical_series(req: LoadHistoricalRequest) -> CandleSeries:
    return svc.load_historical(req.instrument_key, req.interval, req.start, req.end)


@router.post("/historical/bulk-load")
def bulk_load(
    instrument_key: str = Query(...),
    interval: str = Query("1d"),
    num_trading_sessions: int = Query(100, ge=1, le=1000),
) -> dict:
    req = BulkLoadRequest(instrument_key=instrument_key, interval=interval, num_trading_sessions=num_trading_sessions)
    return svc.bulk_load_last_sessions(req.instrument_key, req.interval, req.num_trading_sessions)


@router.get("/historical")
def get_historical(
    instrument_key: str,
    interval: str = "1d",
    limit: int = Query(200, ge=1, le=10000),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
) -> list[dict]:
    try:
        if start is not None and end is not None:
            series = svc.get_historical(instrument_key, interval, start, end, limit=limit)
        else:
            # Best-effort: pull from cache/db; if empty, warm a small default window.
            now = datetime.now(timezone.utc)
            default_days = 30 if str(interval).lower() != "1d" else 365
            series = svc.get_historical(instrument_key, interval, now - timedelta(days=default_days), now, limit=limit)
        return _as_candle_list(series, limit=limit)
    except (UpstoxError, Exception) as e:
        auth = _maybe_auth_401(e)
        if auth is not None:
            return auth
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/historical/series", response_model=CandleSeries)
def get_historical_series(
    instrument_key: str,
    interval: str = "1d",
    start: datetime = Query(...),
    end: datetime = Query(...),
) -> CandleSeries:
    return svc.get_historical(instrument_key, interval, start, end)


@router.get("/auto", response_model=CandleSeries)
def auto(
    instrument_key: str,
    live_interval: str = Query("1m"),
    lookback_minutes: int = Query(30, ge=1, le=360),
    eod_days: int = Query(120, ge=10, le=2000),
) -> CandleSeries:
    """Return live candles during market hours, else EOD (1d) candles."""
    safety = get_safety_status()

    if safety.market_state == "LIVE":
        return svc.poll_intraday(instrument_key, live_interval, lookback_minutes)

    # Market not live: return daily candles
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(eod_days))
    return svc.load_historical(instrument_key, "1d", start, end)
