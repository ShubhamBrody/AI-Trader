from __future__ import annotations

from app.core import settings as settings_module
from app.orders import state as state_module


class FakeUpstoxClient:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def close(self) -> None:
        return

    def order_book_v2(self):
        # Minimal plausible shape: {status, data:[...]}
        return {
            "status": "success",
            "data": [
                {
                    "order_id": "O123",
                    "status": "complete",
                    "tag": "agent:TEST_EQ|ABC",
                    "instrument_token": "12345",
                    "transaction_type": "BUY",
                    "quantity": 1,
                }
            ],
        }

    def order_details_v2(self, order_id: str):
        # When reconcile switches to details after discovering broker_order_id.
        return {"status": "success", "data": {"order_id": order_id, "status": "complete"}}


def test_reconcile_discovers_missing_broker_order_id_from_book(monkeypatch):
    # Enable reconciliation path.
    monkeypatch.setattr(settings_module.settings, "UPSTOX_ACCESS_TOKEN", "TEST_TOKEN", raising=False)

    # Patch the Upstox client used by the order-state module.
    monkeypatch.setattr(state_module, "UpstoxClient", FakeUpstoxClient, raising=False)

    # Create an upstox order without broker_order_id (simulates crash between submit and capture).
    oid = state_module.create_order(
        broker="upstox",
        instrument_key="TEST_EQ|ABC",
        side="BUY",
        qty=1,
        order_kind="ENTRY",
        order_type="MARKET",
        status="NEW",
        meta={
            "tag": "agent:TEST_EQ|ABC",
            "body": {"instrument_token": "12345", "quantity": 1, "transaction_type": "BUY", "tag": "agent:TEST_EQ|ABC"},
        },
    )

    before = state_module.get_order(oid)
    assert before is not None
    assert before.broker_order_id is None

    res = state_module.reconcile_upstox_orders(order_ids=[oid])
    assert res.get("ok") is True

    after = state_module.get_order(oid)
    assert after is not None
    assert after.broker_order_id == "O123"
    assert after.status in {"FILLED", "OPEN", "UNKNOWN"}
