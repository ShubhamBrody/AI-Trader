from __future__ import annotations


def _mk_candle(ts: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> dict:
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_intraday_overlays_trend_following_plan_exists():
    from app.ai.intraday_overlays import analyze_intraday

    # Simple uptrend
    candles = []
    base = 100.0
    for i in range(60):
        c = base + i * 0.1
        candles.append(_mk_candle(1_700_000_000 + i * 60, c - 0.05, c + 0.15, c - 0.2, c, 100 + i))

    res = analyze_intraday(instrument_key="NSE_EQ|TEST", interval="1m", candles=candles)
    assert res["n"] == 60
    assert res["trend"]["dir"] in ("up", "flat")
    # Trade plan may be trend-following
    trade = res.get("trade")
    assert trade is None or (trade["side"] in ("buy", "sell") and trade["entry"] != trade["stop"])


def test_intraday_overlays_breakout_generates_buy_trade():
    from app.ai.intraday_overlays import analyze_intraday

    candles = []
    ts0 = 1_700_100_000

    # Create repeated highs around 105 to form resistance
    price = 100.0
    for i in range(40):
        hi = 105.0 if i % 5 == 0 else price + 0.2
        lo = price - 0.3
        close = price + 0.05
        candles.append(_mk_candle(ts0 + i * 60, price, hi, lo, close, 100.0))
        price += 0.02

    # Breakout candle above resistance with higher volume
    candles.append(_mk_candle(ts0 + 40 * 60, 104.8, 106.5, 104.6, 106.1, 300.0))

    res = analyze_intraday(instrument_key="NSE_EQ|TEST", interval="1m", candles=candles)
    trade = res.get("trade")
    assert trade is not None
    assert trade["side"] == "buy"
    assert trade["stop"] < trade["entry"] < trade["target"]
