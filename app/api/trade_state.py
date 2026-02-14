from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query

from app.ai.engine import AIEngine
from app.candles.service import CandleService
from app.core.guards import get_safety_status
from app.core.settings import settings
from app.paper_trading.service import PaperTradingService
from app.risk.engine import RiskEngine
from app.strategy.engine import StrategyEngine

router = APIRouter(prefix="/trade-state")

_ai = AIEngine()
_candles = CandleService()
_strategy = StrategyEngine()
_risk = RiskEngine()
_paper = PaperTradingService()


@router.get("")
def trade_state(
    instrument_key: str = Query(...),
    interval: str = Query("1d"),
    lookback_days: int = Query(180, ge=30, le=2000),
) -> dict:
    safety = get_safety_status()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    series = _candles.load_historical(instrument_key, interval, start, end)

    highs = [c.high for c in series.candles]
    lows = [c.low for c in series.candles]
    closes = [c.close for c in series.candles]
    volumes = [getattr(c, "volume", 0.0) for c in series.candles]

    idea = _strategy.build_idea(symbol=instrument_key, highs=highs, lows=lows, closes=closes, volumes=volumes)
    pred = _ai.predict(instrument_key=instrument_key, include_nifty=False)

    conf = float(pred.get("prediction", {}).get("confidence", 0.0))
    capital = float(_paper.account().get("balance", 0.0))

    sizing = _risk.position_size(capital=capital, entry=idea.entry, stop=idea.stop_loss, confidence=conf)

    will_trade_live = False
    reasons: list[str] = []

    if safety.trade_lock:
        reasons.append(f"Market locked: {safety.reason}")
    if settings.SAFE_MODE:
        reasons.append("SAFE_MODE=true (live execution disabled)")

    if not reasons and idea.side in {"BUY", "SELL"} and sizing["qty"] > 0:
        will_trade_live = True

    if idea.side == "HOLD":
        reasons.append("Strategy=HOLD")

    return {
        "instrument_key": instrument_key,
        "safety": safety.__dict__,
        "ai": pred,
        "strategy": {
            "side": idea.side,
            "entry": idea.entry,
            "stop_loss": idea.stop_loss,
            "target": idea.target,
            "rr": idea.rr,
            "reason": idea.reason,
        },
        "risk": {
            "capital": capital,
            "position_size": sizing,
        },
        "decision": {
            "will_trade_live": will_trade_live,
            "reasons": reasons,
        },
    }
