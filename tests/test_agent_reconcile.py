from __future__ import annotations

from app.agent.state import list_open_trades, record_trade_open
from app.agent.service import AutoTraderAgent
from app.core import settings as settings_module
import asyncio


class FakeUpstoxClient:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._placed = []

    def close(self) -> None:
        return

    def order_details_v2(self, order_id: str) -> dict:
        # Simulate filled entry
        return {"status": "success", "data": {"order_id": order_id, "status": "complete"}}

    def place_order_v3(self, body: dict) -> dict:
        self._placed.append(body)
        return {"status": "success", "data": {"order_id": "SL123"}}


def test_reconcile_places_missing_sl(monkeypatch):
    # Force live upstox path but with fakes (no network)
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_BROKER", "upstox", raising=False)
    monkeypatch.setattr(settings_module.settings, "SAFE_MODE", False, raising=False)
    monkeypatch.setattr(settings_module.settings, "UPSTOX_USE_BROKER_STOP", True, raising=False)
    monkeypatch.setattr(settings_module.settings, "UPSTOX_FALLBACK_APP_STOP", True, raising=False)

    # Patch UpstoxClient class used by agent
    import app.agent.service as agent_service

    monkeypatch.setattr(agent_service, "UpstoxClient", FakeUpstoxClient)

    # Patch universe list to provide token mapping
    agent = AutoTraderAgent()

    def fake_list(limit=5000, cap_tier=None):
        return [{"instrument_key": "TEST_EQ|REC", "upstox_token": "TKN"}]

    monkeypatch.setattr(agent._uni, "list", fake_list)

    trade_id = record_trade_open(
        instrument_key="TEST_EQ|REC",
        side="BUY",
        qty=1,
        entry=100.0,
        stop=95.0,
        target=110.0,
        entry_order_id="E123",
        sl_order_id=None,
        monitor_app_stop=False,
        meta={"mode": "test"},
    )

    # Run reconcile directly
    res = asyncio.run(agent._reconcile_open_trades())

    assert res["checked"] >= 1

    open_trades = list_open_trades(limit=50)
    t = next(x for x in open_trades if x["id"] == trade_id)
    assert t.get("sl_order_id") == "SL123"
