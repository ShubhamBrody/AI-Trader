from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.agent.service import AutoTraderAgent
from app.agent.state import has_open_trade, record_trade_open
from app.candles.models import Candle, CandleSeries
from app.core import settings as settings_module


def _fake_series(price: float) -> CandleSeries:
    now = int(datetime.now(timezone.utc).timestamp())
    return CandleSeries(
        instrument_key="TEST_EQ|MGR",
        interval="1m",
        candles=[
            Candle(ts=now - 60, open=price, high=price, low=price, close=price, volume=1),
            Candle(ts=now, open=price, high=price, low=price, close=price, volume=1),
        ],
    )


def test_manage_open_trades_target_exit_paper(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_BROKER", "paper", raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_TARGET_EXIT_ENABLED", True, raising=False)

    agent = AutoTraderAgent()

    # Open a paper long position so exit SELL is possible.
    agent._paper.execute(symbol="TEST_EQ|MGR", side="BUY", qty=1, price=100.0)

    record_trade_open(
        instrument_key="TEST_EQ|MGR",
        side="BUY",
        qty=1,
        entry=100.0,
        stop=95.0,
        target=101.0,
        entry_order_id=None,
        sl_order_id=None,
        monitor_app_stop=False,
        meta={"mode": "test"},
    )

    monkeypatch.setattr(agent._candles, "poll_intraday", lambda *args, **kwargs: _fake_series(105.0))

    res = asyncio.run(agent._manage_open_trades())
    assert res["exited"] >= 1
    assert has_open_trade(instrument_key="TEST_EQ|MGR") is False


def test_manage_open_trades_trailing_updates_stop(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_BROKER", "paper", raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_TRAILING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_TRAIL_ACTIVATION_R", 0.0, raising=False)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_TRAIL_PCT", 0.01, raising=False)

    agent = AutoTraderAgent()

    # Open a paper long position.
    agent._paper.execute(symbol="TEST_EQ|MGR", side="BUY", qty=1, price=100.0)

    record_trade_open(
        instrument_key="TEST_EQ|MGR",
        side="BUY",
        qty=1,
        entry=100.0,
        stop=95.0,
        target=200.0,
        entry_order_id=None,
        sl_order_id=None,
        monitor_app_stop=False,
        meta={"mode": "test"},
    )

    monkeypatch.setattr(agent._candles, "poll_intraday", lambda *args, **kwargs: _fake_series(120.0))

    res = asyncio.run(agent._manage_open_trades())
    assert res["trailed"] >= 1
