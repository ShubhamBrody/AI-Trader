from __future__ import annotations

from app.ai.candlestick_patterns import detect_patterns, to_candles
from app.ai.pattern_memory import apply_feedback, ensure_catalog_rows, get_weights


def _mk(ts: int, o: float, h: float, l: float, c: float, v: float = 1.0):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_detect_hammer_in_downtrend():
    # Downtrend into a hammer-like candle.
    raw = []
    p = 100.0
    for i in range(15):
        p *= 0.998
        raw.append(_mk(1700000000 + i * 60, p + 0.2, p + 0.4, p - 0.6, p))

    # Last candle: small body near top, long lower wick.
    raw.append(_mk(1700000000 + 15 * 60, 97.8, 98.1, 96.9, 98.0))

    matches = detect_patterns(to_candles(raw), max_results=10)
    names = [m.name for m in matches]

    assert "Hammer" in names
    m = next(x for x in matches if x.name == "Hammer")
    assert m.side == "buy"
    assert m.confidence > 0.35


def test_detect_bearish_engulfing_in_uptrend():
    raw = []
    p = 100.0
    for i in range(10):
        p *= 1.002
        raw.append(_mk(1700001000 + i * 60, p - 0.2, p + 0.4, p - 0.4, p))

    # Two-candle bearish engulfing: small bull then large bear.
    raw.append(_mk(1700001000 + 10 * 60, 102.0, 102.6, 101.9, 102.5))  # bull
    raw.append(_mk(1700001000 + 11 * 60, 102.7, 102.8, 101.6, 101.7))  # bear engulfing

    matches = detect_patterns(to_candles(raw), max_results=10)
    names = [m.name for m in matches]

    assert "Bearish Engulfing" in names
    m = next(x for x in matches if x.name == "Bearish Engulfing")
    assert m.side == "sell"
    assert m.confidence > 0.4


def test_pattern_memory_feedback_updates_weight(tmp_path):
    # Ensure a single pattern exists in DB, then apply feedback.
    ensure_catalog_rows(
        [
            {
                "name": "Hammer",
                "family": "bullish_reversal",
                "side": "buy",
                "base_reliability": 0.9,
                "params": {"min_window": 1},
            }
        ]
    )

    w0 = get_weights().get("Hammer", 1.0)
    s1 = apply_feedback(pattern="Hammer", good=True, magnitude=2.0, note="test")
    assert s1 is not None

    w1 = get_weights().get("Hammer", 1.0)
    assert w1 != w0
    assert 0.55 <= w1 <= 1.45
