from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.main import create_app
from app.agent.state import record_trade_open, has_open_trade
from app.core import settings as settings_module


def test_agent_flatten_paper_ok(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_BROKER", "paper", raising=False)

    app = create_app()
    client = TestClient(app)

    # Create a paper long position
    r1 = client.post(
        "/api/orders/place",
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100},
    )
    assert r1.status_code == 200

    # Create an open agent trade for the same symbol
    record_trade_open(
        instrument_key="DEMO",
        side="BUY",
        qty=1,
        entry=100.0,
        stop=95.0,
        target=110.0,
        entry_order_id=None,
        sl_order_id=None,
        monitor_app_stop=True,
        meta={"mode": "test"},
    )
    assert has_open_trade(instrument_key="DEMO") is True

    r2 = client.post("/api/agent/flatten")
    assert r2.status_code == 200
    body = r2.json()
    assert body.get("ok") is True
    assert has_open_trade(instrument_key="DEMO") is False


class FakeUpstoxClient:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.cancelled: list[str] = []

    def close(self) -> None:
        return

    def order_book_v2(self) -> dict:
        return {
            "status": "success",
            "data": [
                {"order_id": "O1", "status": "open"},
                {"order_id": "O2", "status": "trigger_pending"},
                {"order_id": "O3", "status": "complete"},
            ],
        }

    def cancel_order_v3(self, order_id: str) -> dict:
        self.cancelled.append(order_id)
        return {"status": "success", "data": {"order_id": order_id}}


def test_agent_cancel_open_orders_upstox_ok(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "AUTOTRADER_BROKER", "upstox", raising=False)
    monkeypatch.setattr(settings_module.settings, "SAFE_MODE", False, raising=False)
    monkeypatch.setattr(settings_module.settings, "LIVE_TRADING_ENABLED", True, raising=False)

    import app.agent.service as agent_service

    monkeypatch.setattr(agent_service, "UpstoxClient", FakeUpstoxClient)

    app = create_app()
    client = TestClient(app)

    r = client.post("/api/agent/cancel-open-orders")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["cancelled"] == 2
