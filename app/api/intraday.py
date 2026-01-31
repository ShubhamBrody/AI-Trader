from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from zoneinfo import ZoneInfo

from app.candles.models import CandleSeries, IntradayPollRequest
from app.candles.service import CandleService
from app.core.guards import get_safety_status
from app.core.settings import settings
from app.markets.nse.calendar import last_n_trading_days, load_holidays
from app.ai.intraday_overlays import analyze_intraday
from app.ai.intraday_overlay_model import load_model, train_and_store

router = APIRouter(prefix="/intraday")
svc = CandleService()


@router.post("/poll")
def poll_intraday(
    instrument_key: str = Query(...),
    interval: str = Query(...),
    lookback_minutes: int = Query(60, ge=5, le=240),
) -> dict:
    # Match BackendComplete + frontend contract.
    status = get_safety_status()
    if status.trade_lock:
        return {"status": "blocked", "reason": status.reason}

    svc.poll_intraday(str(instrument_key), str(interval), int(lookback_minutes))
    return {"status": "polled"}


@router.post("/poll/series", response_model=CandleSeries)
def poll_series(req: IntradayPollRequest) -> CandleSeries:
    """Back-compat endpoint returning the raw CandleSeries."""
    return svc.poll_intraday(req.instrument_key, req.interval, req.lookback_minutes)


@router.get("")
def get_intraday(
    instrument_key: str = Query(...),
    interval: str = Query("1m"),
) -> list[dict]:
    # BackendComplete contract: returns Candle[] with ISO timestamps.
    # Read from DB cache by default; use POST /poll to refresh.
    instrument_key = str(instrument_key)
    interval = str(interval)

    now_utc = datetime.now(timezone.utc)
    status = get_safety_status(now_utc)

    # During LIVE market session, show recent intraday window.
    if status.market_state == "LIVE":
        end = now_utc
        start = end - timedelta(hours=6)
        series = svc.get_historical(instrument_key, interval, start, end)
    else:
        # After hours, /poll is blocked; still show the most recent trading session.
        tz = ZoneInfo(settings.TIMEZONE)
        holidays = load_holidays("app/config/nse_holidays.yaml")
        last_day = last_n_trading_days(now_utc.astimezone(tz).date(), 1, holidays)[0]
        start = datetime(last_day.year, last_day.month, last_day.day, 9, 15, tzinfo=tz)
        end = datetime(last_day.year, last_day.month, last_day.day, 15, 29, tzinfo=tz)
        series = svc.get_historical(instrument_key, interval, start, end)

    out: list[dict] = []
    for c in list(series.candles or []):
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


@router.get("/overlays")
def intraday_overlays(
    instrument_key: str = Query(...),
    interval: str = Query("1m"),
    lookback_minutes: int = Query(90, ge=10, le=360),
    use_model: bool = Query(True, description="If true, apply ML calibrator when available."),
) -> dict:
    """Compute realtime intraday overlays (support/resistance, trend, entry/target/stop).

    During LIVE hours this will poll the broker for the latest window so it updates in realtime.
    """

    instrument_key = str(instrument_key)
    interval = str(interval)

    now_utc = datetime.now(timezone.utc)
    status = get_safety_status(now_utc)

    if status.market_state == "LIVE":
        series = svc.poll_intraday(instrument_key, interval, int(lookback_minutes))
        candles = list(series.candles or [])
    else:
        # After hours: compute from most recent session window.
        tz = ZoneInfo(settings.TIMEZONE)
        holidays = load_holidays("app/config/nse_holidays.yaml")
        last_day = last_n_trading_days(now_utc.astimezone(tz).date(), 1, holidays)[0]
        start = datetime(last_day.year, last_day.month, last_day.day, 9, 15, tzinfo=tz)
        end = datetime(last_day.year, last_day.month, last_day.day, 15, 29, tzinfo=tz)
        series = svc.get_historical(instrument_key, interval, start, end)
        candles = list(series.candles or [])

    analysis = analyze_intraday(instrument_key=instrument_key, interval=interval, candles=candles)
    if not use_model and isinstance(analysis, dict):
        analysis.pop("ai", None)
        if isinstance(analysis.get("trade"), dict):
            analysis["trade"].pop("model_confidence", None)
            analysis["trade"].pop("risk_profile", None)
    return {"status": "ok", "market_state": status.market_state, "analysis": analysis}


@router.get("/overlays/model")
def overlays_model_status(instrument_key: str = Query(...), interval: str = Query("1m")) -> dict:
    m = load_model(str(instrument_key), str(interval))
    if m is None:
        return {"ok": True, "model": None}
    return {
        "ok": True,
        "model": {
            "instrument_key": m.instrument_key,
            "interval": m.interval,
            "created_ts": int(m.created_ts),
            "features": list(m.features),
            "metrics": dict(m.metrics or {}),
            "horizon_steps": int(m.horizon_steps),
            "k_atr": float(m.k_atr),
        },
    }


@router.post("/overlays/train")
def train_overlays_model(
    instrument_key: str = Query(...),
    interval: str = Query("1m"),
    lookback_minutes: int = Query(6 * 60, ge=60, le=24 * 60),
    feature_window: int = Query(60, ge=30, le=180),
    horizon_steps: int = Query(30, ge=5, le=180),
    k_atr: float = Query(1.2, ge=0.5, le=3.0),
) -> dict:
    """Train ML calibrator from recent candles.

    Note: For best results, warm the DB with /api/intraday/poll (or data pipeline) first.
    """

    instrument_key = str(instrument_key)
    interval = str(interval)

    series = svc.poll_intraday(instrument_key, interval, int(lookback_minutes))
    candles = list(series.candles or [])
    if len(candles) < (int(feature_window) + int(horizon_steps) + 20):
        return {"ok": False, "reason": "not_enough_candles", "n": len(candles)}

    return train_and_store(
        instrument_key=instrument_key,
        interval=interval,
        candles=candles,
        feature_window=int(feature_window),
        horizon_steps=int(horizon_steps),
        k_atr=float(k_atr),
    )
