from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class PatternMatch:
    name: str
    family: str  # bullish_reversal|bearish_reversal|bullish_continuation|bearish_continuation
    side: str  # buy|sell|neutral
    window: int
    confidence: float
    details: dict[str, Any]


@dataclass(frozen=True)
class PatternDef:
    name: str
    family: str
    side: str
    best_timeframe: str
    min_window: int
    base_reliability: float
    detector: Callable[[list[Candle]], tuple[float, dict[str, Any]]]


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sorted(cs: list[Candle]) -> list[Candle]:
    return sorted(list(cs or []), key=lambda c: int(c.ts))


def _range(c: Candle) -> float:
    return max(0.0, float(c.high) - float(c.low))


def _body(c: Candle) -> float:
    return abs(float(c.close) - float(c.open))


def _upper_wick(c: Candle) -> float:
    return max(0.0, float(c.high) - max(float(c.open), float(c.close)))


def _lower_wick(c: Candle) -> float:
    return max(0.0, min(float(c.open), float(c.close)) - float(c.low))


def _is_bull(c: Candle) -> bool:
    return float(c.close) > float(c.open)


def _is_bear(c: Candle) -> bool:
    return float(c.close) < float(c.open)


def _pct(x: float, denom: float) -> float:
    if denom <= 1e-12:
        return 0.0
    return float(x / denom)


def _trend_dir(closes: list[float], window: int = 20) -> str:
    # Very lightweight: compare start/end over window.
    if not closes:
        return "flat"
    w = max(5, int(window))
    xs = closes[-w:]
    if len(xs) < 5:
        return "flat"
    start = float(xs[0])
    end = float(xs[-1])
    if start <= 0 or end <= 0:
        return "flat"
    chg = (end - start) / start
    if abs(chg) < 0.002:
        return "flat"
    return "up" if chg > 0 else "down"


def _gap_up(prev: Candle, cur: Candle, *, min_pct: float = 0.001) -> bool:
    # In intraday, gaps are rare; treat as meaningful separation.
    return float(cur.open) >= float(prev.close) * (1.0 + float(min_pct))


def _gap_down(prev: Candle, cur: Candle, *, min_pct: float = 0.001) -> bool:
    return float(cur.open) <= float(prev.close) * (1.0 - float(min_pct))


def _score_clamp(score: float) -> float:
    if not math.isfinite(float(score)):
        return 0.0
    return float(max(0.0, min(1.0, score)))


# --- Single candle patterns ---

def _detect_doji(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    c = cs[-1]
    r = _range(c)
    if r <= 0:
        return 0.0, {}
    b = _body(c)
    body_pct = _pct(b, r)
    # Stronger doji when body very small.
    score = 1.0 - min(1.0, body_pct / 0.12)
    if body_pct <= 0.12:
        return _score_clamp(score), {"body_pct": body_pct}
    return 0.0, {}


def _detect_dragonfly_doji(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    c = cs[-1]
    r = _range(c)
    if r <= 0:
        return 0.0, {}
    b = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    body_pct = _pct(b, r)
    if body_pct > 0.12:
        return 0.0, {}
    if _pct(lw, r) < 0.6:
        return 0.0, {}
    if _pct(uw, r) > 0.15:
        return 0.0, {}
    score = 0.6 + 0.4 * min(1.0, _pct(lw, r))
    return _score_clamp(score), {"body_pct": body_pct, "lw_pct": _pct(lw, r)}


def _detect_gravestone_doji(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    c = cs[-1]
    r = _range(c)
    if r <= 0:
        return 0.0, {}
    b = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    body_pct = _pct(b, r)
    if body_pct > 0.12:
        return 0.0, {}
    if _pct(uw, r) < 0.6:
        return 0.0, {}
    if _pct(lw, r) > 0.15:
        return 0.0, {}
    score = 0.6 + 0.4 * min(1.0, _pct(uw, r))
    return _score_clamp(score), {"body_pct": body_pct, "uw_pct": _pct(uw, r)}


def _detect_spinning_top(cs: list[Candle], *, bearish: bool) -> tuple[float, dict[str, Any]]:
    c = cs[-1]
    r = _range(c)
    if r <= 0:
        return 0.0, {}
    b = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    body_pct = _pct(b, r)
    if not (body_pct <= 0.3 and _pct(uw, r) >= 0.2 and _pct(lw, r) >= 0.2):
        return 0.0, {}
    # Slight bias: bearish spinning top prefers bearish close.
    color_bonus = 0.1 if (bearish and _is_bear(c)) or ((not bearish) and _is_bull(c)) else 0.0
    score = 0.6 + color_bonus
    return _score_clamp(score), {"body_pct": body_pct}


def _detect_marubozu(cs: list[Candle], *, bearish: bool) -> tuple[float, dict[str, Any]]:
    c = cs[-1]
    r = _range(c)
    if r <= 0:
        return 0.0, {}
    b = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    body_pct = _pct(b, r)
    wick_pct = _pct(uw + lw, r)
    if body_pct < 0.85 or wick_pct > 0.15:
        return 0.0, {}
    if bearish and not _is_bear(c):
        return 0.0, {}
    if (not bearish) and not _is_bull(c):
        return 0.0, {}
    score = 0.75 + 0.25 * min(1.0, (body_pct - 0.85) / 0.15)
    return _score_clamp(score), {"body_pct": body_pct, "wick_pct": wick_pct}


def _detect_hammer_like(cs: list[Candle], *, inverted: bool) -> tuple[float, dict[str, Any]]:
    c = cs[-1]
    r = _range(c)
    if r <= 0:
        return 0.0, {}
    b = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    body_pct = _pct(b, r)
    if body_pct > 0.35:
        return 0.0, {}

    if inverted:
        if uw < 2.0 * max(1e-9, b):
            return 0.0, {}
        if lw > 0.35 * r:
            return 0.0, {}
        score = 0.6 + 0.4 * min(1.0, uw / max(1e-9, r))
        return _score_clamp(score), {"body_pct": body_pct, "uw": uw, "lw": lw}

    # normal hammer
    if lw < 2.0 * max(1e-9, b):
        return 0.0, {}
    if uw > 0.35 * r:
        return 0.0, {}
    score = 0.6 + 0.4 * min(1.0, lw / max(1e-9, r))
    return _score_clamp(score), {"body_pct": body_pct, "uw": uw, "lw": lw}


def _detect_belt_hold(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    c = cs[-1]
    r = _range(c)
    if r <= 0:
        return 0.0, {}
    b = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)
    body_pct = _pct(b, r)
    if body_pct < 0.6:
        return 0.0, {}
    if bullish:
        if not _is_bull(c):
            return 0.0, {}
        if _pct(lw, r) > 0.1:
            return 0.0, {}
        score = 0.7 + 0.3 * min(1.0, body_pct)
        return _score_clamp(score), {"body_pct": body_pct}
    else:
        if not _is_bear(c):
            return 0.0, {}
        if _pct(uw, r) > 0.1:
            return 0.0, {}
        score = 0.7 + 0.3 * min(1.0, body_pct)
        return _score_clamp(score), {"body_pct": body_pct}


# --- Two candle patterns ---

def _detect_engulfing(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    if bullish:
        if not (_is_bear(a) and _is_bull(b)):
            return 0.0, {}
        if not (min(b.open, b.close) <= min(a.open, a.close) and max(b.open, b.close) >= max(a.open, a.close)):
            return 0.0, {}
        score = 0.75
        return _score_clamp(score), {}
    else:
        if not (_is_bull(a) and _is_bear(b)):
            return 0.0, {}
        if not (min(b.open, b.close) <= min(a.open, a.close) and max(b.open, b.close) >= max(a.open, a.close)):
            return 0.0, {}
        score = 0.75
        return _score_clamp(score), {}


def _detect_harami(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    if bullish:
        if not _is_bear(a):
            return 0.0, {}
        if not (min(b.open, b.close) >= min(a.open, a.close) and max(b.open, b.close) <= max(a.open, a.close)):
            return 0.0, {}
        score = 0.65
        return _score_clamp(score), {}
    else:
        if not _is_bull(a):
            return 0.0, {}
        if not (min(b.open, b.close) >= min(a.open, a.close) and max(b.open, b.close) <= max(a.open, a.close)):
            return 0.0, {}
        score = 0.65
        return _score_clamp(score), {}


def _detect_piercing_line(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    if not (_is_bear(a) and _is_bull(b)):
        return 0.0, {}
    mid = (a.open + a.close) / 2.0
    if b.open > a.low:
        return 0.0, {}
    if not (b.close > mid and b.close < a.open):
        return 0.0, {}
    return 0.7, {}


def _detect_dark_cloud_cover(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    if not (_is_bull(a) and _is_bear(b)):
        return 0.0, {}
    mid = (a.open + a.close) / 2.0
    if b.open < a.high:
        return 0.0, {}
    if not (b.close < mid and b.close > a.open):
        return 0.0, {}
    return 0.7, {}


def _detect_tweezers(cs: list[Candle], *, top: bool) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    tol = 0.002  # 0.2%
    if top:
        if not (_is_bull(a) and _is_bear(b)):
            return 0.0, {}
        if abs(a.high - b.high) / max(1e-9, a.high) > tol:
            return 0.0, {}
        return 0.65, {"level": float((a.high + b.high) / 2.0)}
    else:
        if not (_is_bear(a) and _is_bull(b)):
            return 0.0, {}
        if abs(a.low - b.low) / max(1e-9, a.low) > tol:
            return 0.0, {}
        return 0.65, {"level": float((a.low + b.low) / 2.0)}


def _detect_counterattack(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    tol = 0.003
    if bullish:
        if not (_is_bear(a) and _is_bull(b)):
            return 0.0, {}
        if abs(b.close - a.close) / max(1e-9, a.close) > tol:
            return 0.0, {}
        return 0.6, {}
    else:
        if not (_is_bull(a) and _is_bear(b)):
            return 0.0, {}
        if abs(b.close - a.close) / max(1e-9, a.close) > tol:
            return 0.0, {}
        return 0.6, {}


def _detect_homing_pigeon(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    if not (_is_bear(a) and _is_bear(b)):
        return 0.0, {}
    if not (min(b.open, b.close) >= min(a.open, a.close) and max(b.open, b.close) <= max(a.open, a.close)):
        return 0.0, {}
    return 0.6, {}


def _detect_stick_sandwich(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    tol = 0.003
    if not (_is_bear(a) and _is_bull(b) and _is_bear(c)):
        return 0.0, {}
    if abs(c.close - a.close) / max(1e-9, a.close) > tol:
        return 0.0, {}
    return 0.6, {}


# --- Multi-candle patterns ---

def _detect_three_soldiers(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    if bullish:
        if not (_is_bull(a) and _is_bull(b) and _is_bull(c)):
            return 0.0, {}
        if not (b.close > a.close and c.close > b.close):
            return 0.0, {}
        return 0.75, {}
    else:
        if not (_is_bear(a) and _is_bear(b) and _is_bear(c)):
            return 0.0, {}
        if not (b.close < a.close and c.close < b.close):
            return 0.0, {}
        return 0.75, {}


def _detect_three_inside(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    if bullish:
        if not _is_bear(a):
            return 0.0, {}
        # b is inside a
        if not (min(b.open, b.close) >= min(a.open, a.close) and max(b.open, b.close) <= max(a.open, a.close)):
            return 0.0, {}
        if not _is_bull(c):
            return 0.0, {}
        if c.close <= a.open:
            return 0.0, {}
        return 0.7, {}
    else:
        if not _is_bull(a):
            return 0.0, {}
        if not (min(b.open, b.close) >= min(a.open, a.close) and max(b.open, b.close) <= max(a.open, a.close)):
            return 0.0, {}
        if not _is_bear(c):
            return 0.0, {}
        if c.close >= a.open:
            return 0.0, {}
        return 0.7, {}


def _detect_star_doji(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    doji_score, _ = _detect_doji([b])
    if doji_score <= 0:
        return 0.0, {}
    if bullish:
        if not (_is_bear(a) and _is_bull(c)):
            return 0.0, {}
        if not _gap_down(a, b, min_pct=0.0005):
            return 0.0, {}
        if c.close <= (a.open + a.close) / 2.0:
            return 0.0, {}
        return 0.7, {"doji": doji_score}
    else:
        if not (_is_bull(a) and _is_bear(c)):
            return 0.0, {}
        if not _gap_up(a, b, min_pct=0.0005):
            return 0.0, {}
        if c.close >= (a.open + a.close) / 2.0:
            return 0.0, {}
        return 0.7, {"doji": doji_score}


def _detect_abandoned_baby(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    doji_score, _ = _detect_doji([b])
    if doji_score <= 0:
        return 0.0, {}
    if bullish:
        if not (_is_bear(a) and _is_bull(c)):
            return 0.0, {}
        if not _gap_down(a, b, min_pct=0.001):
            return 0.0, {}
        if not _gap_up(b, c, min_pct=0.001):
            return 0.0, {}
        return 0.75, {"doji": doji_score}
    else:
        if not (_is_bull(a) and _is_bear(c)):
            return 0.0, {}
        if not _gap_up(a, b, min_pct=0.001):
            return 0.0, {}
        if not _gap_down(b, c, min_pct=0.001):
            return 0.0, {}
        return 0.75, {"doji": doji_score}


def _detect_tristar(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    s1, _ = _detect_doji([a])
    s2, _ = _detect_doji([b])
    s3, _ = _detect_doji([c])
    if min(s1, s2, s3) <= 0:
        return 0.0, {}
    # Simplified: middle doji is gapped away from the others.
    if bullish:
        if not _gap_down(a, b, min_pct=0.0005):
            return 0.0, {}
        if not _gap_up(b, c, min_pct=0.0005):
            return 0.0, {}
        return 0.65, {"doji": float((s1 + s2 + s3) / 3.0)}
    else:
        if not _gap_up(a, b, min_pct=0.0005):
            return 0.0, {}
        if not _gap_down(b, c, min_pct=0.0005):
            return 0.0, {}
        return 0.65, {"doji": float((s1 + s2 + s3) / 3.0)}


def _detect_rising_three_methods(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b, c, d, e = cs[-5], cs[-4], cs[-3], cs[-2], cs[-1]
    if not (_is_bull(a) and _is_bull(e)):
        return 0.0, {}
    # middle three are small and bearish-ish, inside first candle range
    mids = [b, c, d]
    if not all(_range(x) > 0 for x in mids):
        return 0.0, {}
    if not all(float(x.high) <= float(a.high) and float(x.low) >= float(a.low) for x in mids):
        return 0.0, {}
    if e.close <= a.close:
        return 0.0, {}
    return 0.7, {}


def _detect_mat_hold(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    # Similar to rising three methods but allows a gap up after first candle.
    a, b, c, d, e = cs[-5], cs[-4], cs[-3], cs[-2], cs[-1]
    if not (_is_bull(a) and _is_bull(e)):
        return 0.0, {}
    if not _gap_up(a, b, min_pct=0.0003):
        return 0.0, {}
    mids = [b, c, d]
    if not all(float(x.high) <= float(a.high) * 1.01 for x in mids):
        return 0.0, {}
    if e.close <= max(a.close, b.close):
        return 0.0, {}
    return 0.68, {}


def _detect_unique_three_rivers(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    if not _is_bear(a):
        return 0.0, {}
    # b: lower low, closes higher than open (hammer-ish)
    if not (b.low < a.low and b.close > b.open):
        return 0.0, {}
    if not _is_bull(c):
        return 0.0, {}
    if c.close <= b.close:
        return 0.0, {}
    return 0.6, {}


def _detect_concealing_baby_swallow(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b, c, d = cs[-4], cs[-3], cs[-2], cs[-1]
    # Approximate: first two are bearish marubozu, then third bearish with upper wick, fourth bearish engulfing third.
    s1, _ = _detect_marubozu([a], bearish=True)
    s2, _ = _detect_marubozu([b], bearish=True)
    if min(s1, s2) <= 0:
        return 0.0, {}
    if not _is_bear(c) or not _is_bear(d):
        return 0.0, {}
    if _upper_wick(c) <= 0:
        return 0.0, {}
    # engulfing c's body range
    if not (min(d.open, d.close) <= min(c.open, c.close) and max(d.open, d.close) >= max(c.open, c.close)):
        return 0.0, {}
    return 0.6, {}


def _detect_rounding_bottom(cs: list[Candle], window: int = 20) -> tuple[float, dict[str, Any]]:
    xs = cs[-max(10, int(window)) :]
    if len(xs) < 10:
        return 0.0, {}
    closes = [float(x.close) for x in xs]
    if any(x <= 0 for x in closes):
        return 0.0, {}
    n = len(closes)
    mid = n // 2
    left = closes[:mid]
    right = closes[mid:]
    m = min(closes)
    if m not in closes:
        return 0.0, {}
    min_idx = closes.index(m)
    if not (n * 0.25 <= min_idx <= n * 0.75):
        return 0.0, {}
    # Ends above the trough
    if closes[0] <= m * 1.01 or closes[-1] <= m * 1.01:
        return 0.0, {}
    # left trend down-ish, right trend up-ish
    if _trend_dir(left, window=max(5, len(left))) not in {"down", "flat"}:
        return 0.0, {}
    if _trend_dir(right, window=max(5, len(right))) not in {"up", "flat"}:
        return 0.0, {}
    return 0.62, {"window": n}


def _detect_deliberation(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    # Bearish reversal after uptrend: 2 long bullish then small bullish.
    a, b, c = cs[-3], cs[-2], cs[-1]
    if not (_is_bull(a) and _is_bull(b) and _is_bull(c)):
        return 0.0, {}
    if _body(c) >= min(_body(a), _body(b)) * 0.6:
        return 0.0, {}
    return 0.6, {}


def _detect_upside_gap_two_crows(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    if not (_is_bull(a) and _is_bull(b) and _is_bear(c)):
        return 0.0, {}
    if not _gap_up(a, b, min_pct=0.0005):
        return 0.0, {}
    # c opens above b and closes into the gap
    if c.open <= b.open:
        return 0.0, {}
    if c.close >= a.close:
        return 0.0, {}
    return 0.6, {}


def _detect_advance_block(cs: list[Candle]) -> tuple[float, dict[str, Any]]:
    a, b, c = cs[-3], cs[-2], cs[-1]
    if not (_is_bull(a) and _is_bull(b) and _is_bull(c)):
        return 0.0, {}
    # progressively smaller bodies and/or increasing upper wicks
    if not (_body(b) <= _body(a) and _body(c) <= _body(b)):
        return 0.0, {}
    if not (_upper_wick(b) >= _upper_wick(a) or _upper_wick(c) >= _upper_wick(b)):
        return 0.0, {}
    return 0.6, {}


def _detect_hikkake(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    # Simplified 3-candle inside bar + breakout.
    a, b, c = cs[-3], cs[-2], cs[-1]
    # b inside a
    if not (b.high <= a.high and b.low >= a.low):
        return 0.0, {}
    if bullish:
        if c.close <= a.high:
            return 0.0, {}
        return 0.62, {}
    else:
        if c.close >= a.low:
            return 0.0, {}
        return 0.62, {}


def _detect_kicker(cs: list[Candle], *, bullish: bool) -> tuple[float, dict[str, Any]]:
    a, b = cs[-2], cs[-1]
    # Strong reversal candle + gap-ish open.
    if bullish:
        if not (_is_bear(a) and _is_bull(b)):
            return 0.0, {}
        if not _gap_up(a, b, min_pct=0.0005):
            return 0.0, {}
        if _body(b) < _body(a) * 0.8:
            return 0.0, {}
        return 0.75, {}
    else:
        if not (_is_bull(a) and _is_bear(b)):
            return 0.0, {}
        if not _gap_down(a, b, min_pct=0.0005):
            return 0.0, {}
        if _body(b) < _body(a) * 0.8:
            return 0.0, {}
        return 0.75, {}


def _detect_hanging_or_shooting(cs: list[Candle], *, shooting: bool) -> tuple[float, dict[str, Any]]:
    # Same shape as hammer/inverted hammer; trend context handled outside.
    return _detect_hammer_like(cs, inverted=shooting)


def _build_catalog() -> list[PatternDef]:
    # Best timeframe strings are advisory only.
    return [
        PatternDef("Hammer", "bullish_reversal", "buy", "All", 1, 0.9, lambda cs: _detect_hammer_like(cs, inverted=False)),
        PatternDef("Inverted Hammer", "bullish_reversal", "buy", "H1–H4", 1, 0.8, lambda cs: _detect_hammer_like(cs, inverted=True)),
        PatternDef("Bullish Engulfing", "bullish_reversal", "buy", "H4–Daily", 2, 0.9, lambda cs: _detect_engulfing(cs, bullish=True)),
        PatternDef("Piercing Line", "bullish_reversal", "buy", "H4–Daily", 2, 0.7, lambda cs: _detect_piercing_line(cs)),
        PatternDef("Bullish Marubozu", "bullish_continuation", "buy", "All", 1, 0.9, lambda cs: _detect_marubozu(cs, bearish=False)),
        PatternDef("Three White Soldiers", "bullish_continuation", "buy", "H4–Daily", 3, 0.9, lambda cs: _detect_three_soldiers(cs, bullish=True)),
        PatternDef("Three Inside Up", "bullish_reversal", "buy", "H4–Daily", 3, 0.7, lambda cs: _detect_three_inside(cs, bullish=True)),
        PatternDef("Bullish Harami", "bullish_reversal", "buy", "H1–H4", 2, 0.7, lambda cs: _detect_harami(cs, bullish=True)),
        PatternDef("Tweezer Bottom", "bullish_reversal", "buy", "All", 2, 0.7, lambda cs: _detect_tweezers(cs, top=False)),
        PatternDef("Bullish Counterattack", "bullish_reversal", "buy", "H1–H4", 2, 0.6, lambda cs: _detect_counterattack(cs, bullish=True)),
        PatternDef("Bullish Kicker", "bullish_reversal", "buy", "H4–Daily", 2, 1.0, lambda cs: _detect_kicker(cs, bullish=True)),
        PatternDef("Bullish Abandoned Baby", "bullish_reversal", "buy", "Daily", 3, 0.9, lambda cs: _detect_abandoned_baby(cs, bullish=True)),
        PatternDef("Morning Star Doji", "bullish_reversal", "buy", "Daily–Weekly", 3, 0.9, lambda cs: _detect_star_doji(cs, bullish=True)),
        PatternDef("Dragonfly Doji", "bullish_reversal", "buy", "All", 1, 0.7, lambda cs: _detect_dragonfly_doji(cs)),
        PatternDef("Bullish Tri-Star", "bullish_reversal", "buy", "Daily", 3, 0.6, lambda cs: _detect_tristar(cs, bullish=True)),
        PatternDef("Bullish Hikkake", "bullish_reversal", "buy", "H1–H4", 3, 0.7, lambda cs: _detect_hikkake(cs, bullish=True)),
        PatternDef("Concealing Baby Swallow", "bullish_reversal", "buy", "H4–Daily", 4, 0.7, lambda cs: _detect_concealing_baby_swallow(cs)),
        PatternDef("Unique Three Rivers", "bullish_reversal", "buy", "H4–Daily", 3, 0.6, lambda cs: _detect_unique_three_rivers(cs)),
        PatternDef("Rounding Bottom", "bullish_continuation", "buy", "Daily–Weekly", 10, 0.9, lambda cs: _detect_rounding_bottom(cs, window=20)),
        PatternDef("Bullish Belt Hold", "bullish_reversal", "buy", "H1–H4", 1, 0.7, lambda cs: _detect_belt_hold(cs, bullish=True)),
        PatternDef("Mat Hold (Bullish)", "bullish_continuation", "buy", "H4–Daily", 5, 0.9, lambda cs: _detect_mat_hold(cs)),
        PatternDef("Rising Three Methods", "bullish_continuation", "buy", "H4–Daily", 5, 0.9, lambda cs: _detect_rising_three_methods(cs)),
        PatternDef("Homing Pigeon", "bullish_reversal", "buy", "H4–Daily", 2, 0.7, lambda cs: _detect_homing_pigeon(cs)),
        PatternDef("Stick Sandwich", "bullish_reversal", "buy", "Daily", 3, 0.7, lambda cs: _detect_stick_sandwich(cs)),

        PatternDef("Hanging Man", "bearish_reversal", "sell", "All", 1, 0.9, lambda cs: _detect_hanging_or_shooting(cs, shooting=False)),
        PatternDef("Dark Cloud Cover", "bearish_reversal", "sell", "H4–Daily", 2, 0.9, lambda cs: _detect_dark_cloud_cover(cs)),
        PatternDef("Bearish Engulfing", "bearish_reversal", "sell", "H4–Daily", 2, 0.9, lambda cs: _detect_engulfing(cs, bullish=False)),
        PatternDef("Bearish Marubozu", "bearish_continuation", "sell", "All", 1, 0.9, lambda cs: _detect_marubozu(cs, bearish=True)),
        PatternDef("Three Black Crows", "bearish_continuation", "sell", "H4–Daily", 3, 0.9, lambda cs: _detect_three_soldiers(cs, bullish=False)),
        PatternDef("Three Inside Down", "bearish_reversal", "sell", "H4–Daily", 3, 0.7, lambda cs: _detect_three_inside(cs, bullish=False)),
        PatternDef("Bearish Harami", "bearish_reversal", "sell", "H1–H4", 2, 0.7, lambda cs: _detect_harami(cs, bullish=False)),
        PatternDef("Shooting Star", "bearish_reversal", "sell", "All", 1, 0.9, lambda cs: _detect_hanging_or_shooting(cs, shooting=True)),
        PatternDef("Tweezer Top", "bearish_reversal", "sell", "All", 2, 0.7, lambda cs: _detect_tweezers(cs, top=True)),
        PatternDef("Bearish Counterattack", "bearish_reversal", "sell", "H1–H4", 2, 0.6, lambda cs: _detect_counterattack(cs, bullish=False)),
        PatternDef("Bearish Spinning Top", "bearish_reversal", "sell", "All", 1, 0.6, lambda cs: _detect_spinning_top(cs, bearish=True)),
        PatternDef("Bearish Kicker", "bearish_reversal", "sell", "H4–Daily", 2, 1.0, lambda cs: _detect_kicker(cs, bullish=False)),
        PatternDef("Evening Star Doji", "bearish_reversal", "sell", "Daily–Weekly", 3, 0.9, lambda cs: _detect_star_doji(cs, bullish=False)),
        PatternDef("Bearish Abandoned Baby", "bearish_reversal", "sell", "Daily", 3, 0.9, lambda cs: _detect_abandoned_baby(cs, bullish=False)),
        PatternDef("Gravestone Doji", "bearish_reversal", "sell", "All", 1, 0.7, lambda cs: _detect_gravestone_doji(cs)),
        PatternDef("Bearish Tri-Star", "bearish_reversal", "sell", "Daily", 3, 0.6, lambda cs: _detect_tristar(cs, bullish=False)),
        PatternDef("Deliberation", "bearish_reversal", "sell", "Daily–Weekly", 3, 0.7, lambda cs: _detect_deliberation(cs)),
        PatternDef("Upside Gap Two Crows", "bearish_reversal", "sell", "H4–Daily", 3, 0.6, lambda cs: _detect_upside_gap_two_crows(cs)),
        PatternDef("Advance Block", "bearish_reversal", "sell", "H4–Daily", 3, 0.6, lambda cs: _detect_advance_block(cs)),
    ]


CATALOG: list[PatternDef] = _build_catalog()


def to_candles(raw: list[Any]) -> list[Candle]:
    out: list[Candle] = []
    for c in raw or []:
        if isinstance(c, dict):
            ts = int(c.get("ts") or 0)
            out.append(
                Candle(
                    ts=ts,
                    open=_f(c.get("open")),
                    high=_f(c.get("high")),
                    low=_f(c.get("low")),
                    close=_f(c.get("close")),
                    volume=_f(c.get("volume")),
                )
            )
        else:
            ts = int(getattr(c, "ts", 0) or 0)
            out.append(
                Candle(
                    ts=ts,
                    open=_f(getattr(c, "open", 0.0)),
                    high=_f(getattr(c, "high", 0.0)),
                    low=_f(getattr(c, "low", 0.0)),
                    close=_f(getattr(c, "close", 0.0)),
                    volume=_f(getattr(c, "volume", 0.0)),
                )
            )
    return _sorted(out)


def detect_patterns(
    candles: list[Candle],
    *,
    weights: dict[str, float] | None = None,
    max_results: int = 8,
) -> list[PatternMatch]:
    cs = _sorted(candles)
    if len(cs) < 2:
        return []

    closes = [float(c.close) for c in cs]
    trend = _trend_dir(closes, window=20)

    out: list[PatternMatch] = []
    for p in CATALOG:
        if len(cs) < p.min_window:
            continue
        window_cs = cs[-p.min_window :]
        raw_score, details = p.detector(window_cs)
        raw_score = _score_clamp(raw_score)
        if raw_score <= 0:
            continue

        # Trend context boosts for certain single-candle patterns.
        context = 1.0
        if p.name in {"Hammer", "Inverted Hammer", "Bullish Engulfing", "Piercing Line", "Bullish Harami", "Tweezer Bottom", "Morning Star Doji"}:
            if trend == "down":
                context += 0.08
        if p.name in {"Hanging Man", "Shooting Star", "Bearish Engulfing", "Dark Cloud Cover", "Bearish Harami", "Tweezer Top", "Evening Star Doji", "Advance Block", "Deliberation"}:
            if trend == "up":
                context += 0.08

        w = float((weights or {}).get(p.name, 1.0))
        base = float(max(0.0, min(1.0, p.base_reliability / 10.0 if p.base_reliability > 1.5 else p.base_reliability)))
        score = raw_score
        score = score * (0.75 + 0.25 * base) * context
        score = score * max(0.4, min(1.6, w))
        score = _score_clamp(score)

        out.append(
            PatternMatch(
                name=p.name,
                family=p.family,
                side=p.side,
                window=p.min_window,
                confidence=score,
                details={**(details or {}), "trend": trend, "weight": w, "base": base, "best_timeframe": p.best_timeframe},
            )
        )

    out.sort(key=lambda m: (-float(m.confidence), m.name))
    return out[: max(1, int(max_results))]
