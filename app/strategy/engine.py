from __future__ import annotations

from dataclasses import dataclass

from app.strategy.indicators import atr, ema, rsi, sma
from app.strategy.levels import detect_support_resistance


@dataclass(frozen=True)
class TradeIdea:
    symbol: str
    side: str  # BUY/SELL/HOLD
    entry: float
    stop_loss: float
    target: float
    rr: float
    reason: str


class StrategyEngine:
    def build_idea(self, symbol: str, highs: list[float], lows: list[float], closes: list[float]) -> TradeIdea:
        if not closes:
            return TradeIdea(symbol=symbol, side="HOLD", entry=0, stop_loss=0, target=0, rr=0, reason="No data")

        last = float(closes[-1])
        rsi14 = rsi(closes, 14)
        ema20 = ema(closes, 20)
        sma50 = sma(closes, 50)
        a = atr(highs, lows, closes)
        levels = detect_support_resistance(highs, lows)

        # Simple regime: trend if EMA above SMA and RSI supportive
        if ema20 > sma50 and rsi14 < 70:
            side = "BUY"
            entry = last
            stop = max(0.0, last - max(a, last * 0.01))
            target = last + 2 * (last - stop)
            rr = (target - entry) / max(entry - stop, 1e-9)
            reason = f"Trend up (EMA20>SMA50), RSI={rsi14:.1f}"
        elif ema20 < sma50 and rsi14 > 30:
            side = "SELL"
            entry = last
            stop = last + max(a, last * 0.01)
            target = last - 2 * (stop - last)
            rr = (entry - target) / max(stop - entry, 1e-9)
            reason = f"Trend down (EMA20<SMA50), RSI={rsi14:.1f}"
        else:
            side = "HOLD"
            entry = last
            stop = last
            target = last
            rr = 0.0
            reason = f"No edge (RSI={rsi14:.1f})"

        # Attach nearby levels for transparency
        if levels.support or levels.resistance:
            reason += " | levels detected"

        return TradeIdea(symbol=symbol, side=side, entry=entry, stop_loss=stop, target=target, rr=rr, reason=reason)
