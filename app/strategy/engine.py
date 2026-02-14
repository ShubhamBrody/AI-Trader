from __future__ import annotations

from dataclasses import dataclass

from app.strategy.indicators import adx_last, atr, ema, macd_last, rsi, sma
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
    def build_idea(
        self,
        symbol: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        *,
        volumes: list[float] | None = None,
    ) -> TradeIdea:
        if not closes:
            return TradeIdea(symbol=symbol, side="HOLD", entry=0, stop_loss=0, target=0, rr=0, reason="No data")

        last = float(closes[-1])
        rsi14 = rsi(closes, 14)
        ema20 = ema(closes, 20)
        sma50 = sma(closes, 50)
        a = atr(highs, lows, closes)
        levels = detect_support_resistance(highs, lows)

        # --- Trend determination (intraday/HFT friendly) ---
        # Combine:
        # - Price action: HH/HL or LL/LH over recent windows
        # - Moving averages: EMA20 vs SMA50
        # - Momentum: MACD sign
        # - Trend strength: ADX (+DI/-DI)
        # - Optional volume confirmation
        pa_up = False
        pa_down = False
        if len(highs) >= 40 and len(lows) >= 40:
            hi_prev = max(float(x) for x in highs[-40:-20])
            hi_now = max(float(x) for x in highs[-20:])
            lo_prev = min(float(x) for x in lows[-40:-20])
            lo_now = min(float(x) for x in lows[-20:])
            pa_up = (hi_now > hi_prev) and (lo_now > lo_prev)
            pa_down = (hi_now < hi_prev) and (lo_now < lo_prev)

        macd = macd_last([float(x) for x in closes])
        macd_line = float(macd.get("macd") or 0.0)

        adx = adx_last([float(x) for x in highs], [float(x) for x in lows], [float(x) for x in closes])
        adx_v = float(adx.get("adx") or 0.0)
        pdi = float(adx.get("+di") or 0.0)
        mdi = float(adx.get("-di") or 0.0)

        vol_confirm = None
        if volumes is not None and len(volumes) >= 40:
            v_prev = sum(float(x) for x in volumes[-40:-20]) / 20.0
            v_now = sum(float(x) for x in volumes[-20:]) / 20.0
            # "Increasing volume" confirmation (simple, best-effort)
            vol_confirm = bool(v_prev > 0 and v_now >= v_prev * 1.05)

        # RSI strategy constraints (Wilder RSI14 + common thresholds).
        # - Avoid longs when overbought (>=70), avoid shorts when oversold (<=30).
        # - Trend confirmation bands (Cardwell-style): uptrends tend to hold RSI >=40, downtrends <=60.
        rsi_overbought = 70.0
        rsi_oversold = 30.0
        rsi_uptrend_floor = 40.0
        rsi_downtrend_ceiling = 60.0

        # Score-based trend decision so we can combine multiple techniques.
        up_score = 0
        down_score = 0

        # Price action
        if pa_up:
            up_score += 1
        if pa_down:
            down_score += 1

        # Moving averages
        if ema20 > sma50:
            up_score += 1
        elif ema20 < sma50:
            down_score += 1

        # MACD momentum
        if macd_line > 0:
            up_score += 1
        elif macd_line < 0:
            down_score += 1

        # ADX trend strength and direction
        if adx_v >= 25.0:
            if pdi > mdi:
                up_score += 1
            elif mdi > pdi:
                down_score += 1

        # Optional volume confirmation: if provided, treat as a small extra vote
        if vol_confirm is True:
            if pa_up:
                up_score += 1
            if pa_down:
                down_score += 1

        trend_up = up_score >= 3 and up_score >= down_score + 1
        trend_down = down_score >= 3 and down_score >= up_score + 1

        # Entry decisions: require BOTH a trend vote and RSI band compliance.
        if trend_up and (rsi_uptrend_floor <= rsi14 < rsi_overbought):
            side = "BUY"
            entry = last
            stop = max(0.0, last - max(a, last * 0.01))
            target = last + 2 * (last - stop)
            rr = (target - entry) / max(entry - stop, 1e-9)
            reason = f"Trend up (score={up_score}/{down_score}, RSI={rsi14:.1f}, MACD={macd_line:.3f}, ADX={adx_v:.1f})"
        elif trend_down and (rsi_oversold < rsi14 <= rsi_downtrend_ceiling):
            side = "SELL"
            entry = last
            stop = last + max(a, last * 0.01)
            target = last - 2 * (stop - last)
            rr = (entry - target) / max(stop - entry, 1e-9)
            reason = f"Trend down (score={down_score}/{up_score}, RSI={rsi14:.1f}, MACD={macd_line:.3f}, ADX={adx_v:.1f})"
        else:
            side = "HOLD"
            entry = last
            stop = last
            target = last
            rr = 0.0
            reason = f"No edge (up={up_score}, down={down_score}, RSI={rsi14:.1f}, MACD={macd_line:.3f}, ADX={adx_v:.1f})"

        # Attach nearby levels for transparency
        if levels.support or levels.resistance:
            reason += " | levels detected"

        return TradeIdea(symbol=symbol, side=side, entry=entry, stop_loss=stop, target=target, rr=rr, reason=reason)
