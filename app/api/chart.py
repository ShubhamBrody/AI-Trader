from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query

from app.candles.service import CandleService
from app.strategy.engine import StrategyEngine
from app.strategy.levels import detect_support_resistance

router = APIRouter(prefix="/chart")

_candles = CandleService()
_strategy = StrategyEngine()


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

    idea = _strategy.build_idea(
        symbol=instrument_key,
        highs=highs,
        lows=lows,
        closes=closes,
    )

    return {
        "instrument_key": instrument_key,
        "interval": interval,
        "candles": [c.model_dump() for c in series.candles],
        "overlays": {
            "support": levels.support,
            "resistance": levels.resistance,
            "trade": {
                "side": idea.side,
                "entry": idea.entry,
                "stop_loss": idea.stop_loss,
                "target": idea.target,
                "rr": idea.rr,
                "reason": idea.reason,
            },
        },
    }
