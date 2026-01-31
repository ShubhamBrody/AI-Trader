from __future__ import annotations

from app.core import settings as settings_module
from app.orders import state as state_module


class FakeUpstoxClient:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def close(self) -> None:
        return

    def order_book_v2(self):
        return {
            "status": "success",
            "data": [
                # Known order
                {
                    "order_id": "O1",
                    "status": "open",
                    "tag": "agent:KNOWN",
                    "instrument_token": "12345",
                    "transaction_type": "BUY",
                    "quantity": 1,
                },
                # Unknown open order that looks like it came from us
                {
                    "order_id": "O999",
                    "status": "open",
                    "tag": "agent:UNKNOWN",
                    "instrument_token": "99999",
                    "transaction_type": "SELL",
                    "quantity": 2,
                },
            ],
        }

    def order_details_v2(self, order_id: str):
        return {"status": "success", "data": {"order_id": order_id, "status": "open"}}


def test_reconcile_reports_unknown_open_orders(monkeypatch):
    monkeypatch.setattr(settings_module.settings, "UPSTOX_ACCESS_TOKEN", "TEST_TOKEN", raising=False)
    monkeypatch.setattr(state_module, "UpstoxClient", FakeUpstoxClient, raising=False)

    oid = state_module.create_order(
        broker="upstox",
        instrument_key="TEST_EQ|KNOWN",
        side="BUY",
        qty=1,
        order_kind="ENTRY",
        order_type="MARKET",
        broker_order_id="O1",
        status="SUBMITTED",
        meta={"tag": "agent:KNOWN"},
    )

    res = state_module.reconcile_upstox_orders(order_ids=[oid])
    assert res.get("ok") is True
    assert res.get("unknown_open_count") == 1

    unknown = res.get("unknown_open") or []
    assert isinstance(unknown, list) and unknown
    assert unknown[0].get("order_id") == "O999"
    assert unknown[0].get("ours") is True
