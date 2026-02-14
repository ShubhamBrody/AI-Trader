from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _mk_candle(ts: int, o: float, h: float, l: float, c: float, v: float):
    from app.candles.models import Candle

    return Candle(ts=int(ts), open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v))


def test_trader_decision_response_shape_offline():
    """Smoke test for /api/trader/decision contract.

    This runs offline (Upstox creds are disabled by tests/conftest.py) and relies on DB candles.
    """

    from app.core.db import init_db
    from app.candles.persistence_sql import upsert_candles
    from app.main import app

    init_db()

    instrument_key = "NSE_EQ|TEST"
    interval = "1m"

    now = int(time.time())
    # Insert ~5 hours of candles so the default 240m lookback window has enough.
    start_ts = now - 5 * 60 * 60

    candles = []
    price = 100.0
    for i in range(5 * 60):
        ts = start_ts + i * 60
        # gentle drift + small oscillation
        price = price + 0.01 + (0.02 if i % 10 == 0 else -0.005)
        o = price - 0.03
        c = price + 0.01
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        v = 100.0 + (20.0 if i % 15 == 0 else 0.0)
        candles.append(_mk_candle(ts, o, h, l, c, v))

    upsert_candles(instrument_key, interval, candles)

    client = TestClient(app)
    r = client.get(
        "/api/trader/decision",
        params={
            "instrument_key": instrument_key,
            "interval": interval,
            "lookback_minutes": 240,
            "horizon_steps": 12,
        },
    )

    assert r.status_code == 200
    data = r.json()

    assert data.get("ok") is True
    assert data.get("instrument_key") == instrument_key
    assert data.get("interval") == interval

    assert "ai" in data and isinstance(data["ai"], dict)
    assert data["ai"].get("action") in {"BUY", "SELL", "HOLD"}

    assert "indicators" in data and isinstance(data["indicators"], dict)
    assert all(k in data["indicators"] for k in ("ema20", "ema50", "ema200", "sma20", "rsi14", "bollinger", "macd", "stochastic"))

    assert "decision" in data and isinstance(data["decision"], dict)
    assert data["decision"].get("action") in {"BUY", "SELL", "HOLD"}
    assert isinstance(data["decision"].get("reasons"), list)
    assert isinstance(data["decision"].get("risk_multiplier"), (int, float))
    assert isinstance(data["decision"].get("risk_fraction_suggested"), (int, float))
