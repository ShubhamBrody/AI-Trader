from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.agent.service import AutoTraderAgent
from app.agent.state import list_open_trades, record_trade_open
from app.candles.models import Candle, CandleSeries
from app.core import settings as settings_module


def _series(prices: list[float], *, instrument_key: str = "TEST_EQ|OVR", interval: str = "1m") -> CandleSeries:
    now = int(datetime.now(timezone.utc).timestamp())
    candles: list[Candle] = []
    # spread candles one minute apart
    for i, p in enumerate(prices):
        ts = now - (len(prices) - 1 - i) * 60
        candles.append(Candle(ts=ts, open=p, high=p, low=p, close=p, volume=1))
    return CandleSeries(instrument_key=instrument_key, interval=interval, candles=candles)


def test_overlay_management_updates_stop_and_target_paper(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_BROKER", "paper", raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_TRAILING_ENABLED", False, raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_OVERLAYS_MANAGEMENT_ENABLED", True, raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_OVERLAYS_MIN_CANDLES", 1, raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_OVERLAYS_MIN_CONFIDENCE", 0.0, raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_OVERLAYS_MIN_STOP_MOVE_PCT", 0.0, raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_OVERLAYS_MIN_TARGET_MOVE_PCT", 0.0, raising=False)

    agent = AutoTraderAgent()

    # Open a paper long position.
    agent._paper.execute(symbol="TEST_EQ|OVR", side="BUY", qty=1, price=100.0)

    record_trade_open(
        instrument_key="TEST_EQ|OVR",
        side="BUY",
        qty=1,
        entry=100.0,
        stop=95.0,
        target=110.0,
        entry_order_id=None,
        sl_order_id=None,
        monitor_app_stop=False,
        meta={"mode": "test"},
    )

    # Candle polling used both for last price (5m/30m lookback) and overlays (90m by default).
    monkeypatch.setattr(agent._candles, "poll_intraday", lambda *args, **kwargs: _series([99.0, 101.0, 105.0]))

    # Patch overlays output to a deterministic plan.
    import app.agent.service as agent_service

    asof = int(datetime.now(timezone.utc).timestamp())

    def _fake_analyze_intraday(*, instrument_key: str, interval: str, candles: list) -> dict:
        return {
            "instrument_key": instrument_key,
            "interval": interval,
            "asof_ts": asof,
            "n": len(candles),
            "atr": 1.0,
            "buffer": 0.1,
            "trend": {"dir": "up", "strength": 0.9},
            "levels": {"support": [], "resistance": []},
            "patterns": [],
            "trade": {"side": "buy", "entry": 105.0, "stop": 101.0, "target": 130.0, "confidence": 0.9},
            "ai": {"enabled": False, "model": None},
        }

    monkeypatch.setattr(agent_service, "analyze_intraday", _fake_analyze_intraday, raising=False)

    res = asyncio.run(agent._manage_open_trades())
    assert res.get("overlay_updates", 0) >= 1

    open_trades = list_open_trades(limit=50)
    t = next((x for x in open_trades if x.get("instrument_key") == "TEST_EQ|OVR"), None)
    assert t is not None
    assert float(t["stop"]) == 101.0
    assert float(t["target"]) == 130.0
    assert bool(t["monitor_app_stop"]) is True
