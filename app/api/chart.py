from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query

from app.candles.service import CandleService
from app.ai.engine import AIEngine
from app.strategy.levels import detect_support_resistance

router = APIRouter(prefix="/chart")

_candles = CandleService()
_ai = AIEngine()


@router.get("/overlay")
def overlay(
    instrument_key: str = Query(...),
    interval: str = Query("1d"),
    lookback_days: int = Query(120, ge=10, le=2000),
) -> dict:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    series = _candles.load_historical(instrument_key, interval, start, end)
    highs = [c.high for c in series.candles]
    lows = [c.low for c in series.candles]
    closes = [c.close for c in series.candles]

    levels = detect_support_resistance(highs, lows)

    # AI-first trade idea (keeps response shape stable)
    last = float(closes[-1]) if closes else 0.0
    pred = _ai.predict(str(instrument_key), interval=str(interval), lookback_days=int(lookback_days), horizon_steps=1)
    p = pred.get("prediction") or {}
    ohlc = p.get("next_hour_ohlc") or {}
    predicted_close = float(ohlc.get("close") or last)
    signal = str(p.get("signal") or "HOLD").upper()
    confidence = float(p.get("confidence") or 0.0)
    uncertainty = float(p.get("uncertainty") or 1.0)

    # Vol proxy for stop/target bands when levels are not enough.
    band = max(last * 0.01, abs(predicted_close - last), last * 0.002)
    band = float(band * (1.0 + 0.8 * uncertainty))

    if signal == "BUY":
        entry = last
        stop = max(0.0, entry - band)
        target = max(entry + band * 2.0, predicted_close)
        rr = (target - entry) / max(entry - stop, 1e-9)
        reason = f"AI BUY (conf={confidence:.2f}, unc={uncertainty:.2f})"
    elif signal == "SELL":
        entry = last
        stop = entry + band
        target = min(entry - band * 2.0, predicted_close)
        rr = (entry - target) / max(stop - entry, 1e-9)
        reason = f"AI SELL (conf={confidence:.2f}, unc={uncertainty:.2f})"
    else:
        entry = last
        stop = last
        target = last
        rr = 0.0
        reason = f"AI HOLD (conf={confidence:.2f}, unc={uncertainty:.2f})"

    if levels.support or levels.resistance:
        reason += " | levels detected"

    return {
        "instrument_key": instrument_key,
        "interval": interval,
        "candles": [c.model_dump() for c in series.candles],
        "overlays": {
            "support": levels.support,
            "resistance": levels.resistance,
            "trade": {
                "side": signal,
                "entry": float(entry),
                "stop_loss": float(stop),
                "target": float(target),
                "rr": float(rr),
                "reason": reason,
            },
        },
    }
