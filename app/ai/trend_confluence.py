from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

from app.strategy.indicators import adx_last, macd_last, rsi, sma, ema


TrendDir = Literal["up", "down", "range", "mixed", "unknown"]
TradeAction = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class TimeframeConfluence:
    timeframe: str
    close: float
    sma50: float
    sma200: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_cross: str  # bullish|bearish|none
    rsi14: float
    adx14: float
    plus_di: float
    minus_di: float
    adx_rising: bool
    vol: float
    vol_ema10: float

    core_up: bool
    core_down: bool
    avoid_range: bool


@dataclass(frozen=True)
class TrendConfluenceResult:
    dir: TrendDir
    confidence: float
    strategy: str | None
    score: float
    timeframes: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ConfluencePolicy:
    """Best-effort policy derived from confluence output.

    The intent is to let the system decide when to:
    - hard-gate an entry (force HOLD), vs
    - allow entry but scale risk (via risk_multiplier).

    This is deterministic and uses only the confluence's own confidence.
    """

    action: TradeAction
    risk_multiplier: float
    reason: str


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, float(v))))


def confluence_policy(
    *,
    intended_action: TradeAction,
    confluence: dict[str, Any] | None,
    gate_confidence: float = 0.65,
    unclear_confidence: float = 0.60,
) -> ConfluencePolicy:
    """Translate confluence output into an action + risk multiplier.

    - If confluence is strong and *conflicts*, gate (HOLD).
    - If confluence is strong but *unclear* (range/mixed/unknown), gate (HOLD).
    - If confluence aligns, allow and scale risk up modestly.
    - If confluence is weak/absent, don't gate; scale risk slightly down.

    The returned risk_multiplier is intended to scale risk budget (0..1.25).
    """

    intended: TradeAction = intended_action if intended_action in {"BUY", "SELL"} else "HOLD"
    if intended == "HOLD":
        return ConfluencePolicy(action="HOLD", risk_multiplier=1.0, reason="policy_hold")

    if not isinstance(confluence, dict):
        return ConfluencePolicy(action=intended, risk_multiplier=0.90, reason="confluence_unavailable")

    tdir = str(confluence.get("dir") or "unknown")
    tconf = float(confluence.get("confidence") or 0.0)
    tconf = float(_clamp01(tconf))

    # Gate if regime is unclear *and* the confluence is confident about that.
    if tdir in {"range", "mixed", "unknown"} and tconf >= float(unclear_confidence):
        return ConfluencePolicy(action="HOLD", risk_multiplier=0.0, reason=f"confluence_{tdir}")

    # Gate on strong directional conflicts.
    if intended == "BUY" and tdir == "down" and tconf >= float(gate_confidence):
        return ConfluencePolicy(action="HOLD", risk_multiplier=0.0, reason="confluence_conflict_down")
    if intended == "SELL" and tdir == "up" and tconf >= float(gate_confidence):
        return ConfluencePolicy(action="HOLD", risk_multiplier=0.0, reason="confluence_conflict_up")

    # If aligned, scale up; if weak or slightly conflicting, scale down.
    aligned = (intended == "BUY" and tdir == "up") or (intended == "SELL" and tdir == "down")
    if aligned:
        # 0.90 .. 1.25
        rm = 0.90 + 0.35 * tconf
        return ConfluencePolicy(action=intended, risk_multiplier=float(max(0.10, min(1.25, rm))), reason="confluence_aligned")

    # Mild disagreement but not confident enough to gate.
    rm = 0.75 - 0.35 * tconf  # 0.75 .. 0.40
    return ConfluencePolicy(action=intended, risk_multiplier=float(max(0.10, min(1.25, rm))), reason="confluence_soft_disagree")


def _macd_cross(closes: list[float]) -> str:
    if len(closes) < 4:
        return "none"
    prev = macd_last(closes[:-1], fast=12, slow=26, signal=9)
    last = macd_last(closes, fast=12, slow=26, signal=9)
    pm, ps = float(prev.get("macd") or 0.0), float(prev.get("signal") or 0.0)
    lm, ls = float(last.get("macd") or 0.0), float(last.get("signal") or 0.0)
    if pm <= ps and lm > ls:
        return "bullish"
    if pm >= ps and lm < ls:
        return "bearish"
    return "none"


def _tf_signals(
    *,
    timeframe: str,
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None,
) -> TimeframeConfluence:
    last_close = float(closes[-1]) if closes else 0.0

    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)

    macd_d = macd_last(closes, fast=12, slow=26, signal=9)
    macd_line = float(macd_d.get("macd") or 0.0)
    macd_sig = float(macd_d.get("signal") or 0.0)
    macd_hist = float(macd_d.get("hist") or 0.0)
    macd_cross = _macd_cross(closes)

    rsi14 = float(rsi(closes, 14))

    adx_d = adx_last(highs, lows, closes, period=14)
    adx_v = float(adx_d.get("adx") or 0.0)
    pdi = float(adx_d.get("+di") or 0.0)
    mdi = float(adx_d.get("-di") or 0.0)

    # Best-effort ADX rising check.
    # We re-run ADX on the series excluding the last candle to approximate previous ADX.
    if len(closes) >= 6:
        adx_prev = adx_last(highs[:-1], lows[:-1], closes[:-1], period=14).get("adx") or 0.0
        adx_rising = float(adx_v) > float(adx_prev)
    else:
        adx_rising = False

    vol = float(volumes[-1]) if volumes else 0.0
    vol_ema10 = float(ema(volumes, 10)) if volumes else 0.0

    # Core combined conditions (your spec).
    price_above_200 = last_close > sma200 and sma200 > 0
    price_below_200 = last_close < sma200 and sma200 > 0

    ma_up = sma50 > sma200 and sma200 > 0
    ma_down = sma50 < sma200 and sma200 > 0

    # MACD confirmation requires both cross and sign.
    macd_bull = macd_cross == "bullish" and macd_hist > 0 and macd_line > 0
    macd_bear = macd_cross == "bearish" and macd_hist < 0 and macd_line < 0

    # RSI confirmation bands.
    rsi_bull = (rsi14 > 50.0) and (rsi14 < 70.0)
    rsi_bear = (rsi14 < 50.0) and (rsi14 > 30.0)

    # Strong trend filter.
    adx_strong = adx_v > 25.0 and adx_rising
    adx_bull = adx_strong and (pdi > mdi)
    adx_bear = adx_strong and (mdi > pdi)

    # Volume participation.
    vol_ok = bool(vol_ema10 > 0 and vol > vol_ema10)

    avoid_range = adx_v < 20.0

    core_up = bool(price_above_200 and ma_up and macd_bull and rsi_bull and adx_bull and vol_ok and not avoid_range)
    core_down = bool(price_below_200 and ma_down and macd_bear and rsi_bear and adx_bear and vol_ok and not avoid_range)

    return TimeframeConfluence(
        timeframe=str(timeframe),
        close=float(last_close),
        sma50=float(sma50),
        sma200=float(sma200),
        macd=float(macd_line),
        macd_signal=float(macd_sig),
        macd_hist=float(macd_hist),
        macd_cross=str(macd_cross),
        rsi14=float(rsi14),
        adx14=float(adx_v),
        plus_di=float(pdi),
        minus_di=float(mdi),
        adx_rising=bool(adx_rising),
        vol=float(vol),
        vol_ema10=float(vol_ema10),
        core_up=bool(core_up),
        core_down=bool(core_down),
        avoid_range=bool(avoid_range),
    )


def analyze_multi_timeframe_confluence(
    *,
    daily: dict[str, list[float]] | None = None,
    h4: dict[str, list[float]] | None = None,
    h1: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Analyze trend direction using multi-indicator + multi-timeframe confluence.

    Inputs are per-timeframe OHLCV arrays: {highs,lows,closes,volumes}.

    Returns a JSON-serializable dict.
    """

    tfs: list[tuple[str, dict[str, list[float]] | None, float]] = [
        ("1d", daily, 3.0),
        ("4h", h4, 2.0),
        ("1h", h1, 1.0),
    ]

    tf_details: dict[str, dict[str, Any]] = {}
    up_w = 0.0
    down_w = 0.0
    range_w = 0.0

    for tf, data, w in tfs:
        if not data:
            continue
        highs = list(map(float, data.get("highs") or []))
        lows = list(map(float, data.get("lows") or []))
        closes = list(map(float, data.get("closes") or []))
        volumes_raw = data.get("volumes")
        volumes = list(map(float, volumes_raw)) if isinstance(volumes_raw, list) else None

        if not closes or not highs or not lows:
            continue

        sig = _tf_signals(timeframe=tf, highs=highs, lows=lows, closes=closes, volumes=volumes)
        tf_details[tf] = asdict(sig)

        if sig.avoid_range:
            range_w += w
        if sig.core_up:
            up_w += w
        if sig.core_down:
            down_w += w

    if not tf_details:
        return asdict(
            TrendConfluenceResult(
                dir="unknown",
                confidence=0.0,
                strategy=None,
                score=0.0,
                timeframes={},
            )
        )

    # Determine direction.
    # High confidence when all required timeframes align; otherwise "mixed".
    dirn: TrendDir
    if up_w > 0 and down_w == 0 and range_w == 0 and up_w >= 5.5:
        dirn = "up"
    elif down_w > 0 and up_w == 0 and range_w == 0 and down_w >= 5.5:
        dirn = "down"
    elif range_w >= 3.0 and up_w == 0 and down_w == 0:
        dirn = "range"
    elif up_w > 0 and down_w > 0:
        dirn = "mixed"
    else:
        dirn = "unknown"

    # Confidence heuristic.
    aligned = max(up_w, down_w)
    base = aligned / 6.0  # 0..1
    if dirn in {"up", "down"}:
        conf = 0.55 + 0.40 * base
    elif dirn == "range":
        conf = 0.50 + 0.25 * _clamp01(range_w / 6.0)
    elif dirn == "mixed":
        conf = 0.25
    else:
        conf = 0.10

    # Score: signed for convenience.
    score = float(up_w - down_w)

    return asdict(
        TrendConfluenceResult(
            dir=dirn,
            confidence=float(_clamp01(conf)),
            strategy="ma_macd_rsi_adx_volume_mtf_v1" if dirn in {"up", "down"} else None,
            score=float(score),
            timeframes=tf_details,
        )
    )
