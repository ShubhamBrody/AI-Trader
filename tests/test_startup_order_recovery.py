from __future__ import annotations

from app.core import settings as settings_module
from app.orders import startup_recovery as startup_module
from app.orders import state as state_module


def test_startup_recovery_selects_recent_upstox_orders(monkeypatch):
    # Enable startup recovery and configure token.
    monkeypatch.setattr(settings_module.settings, "ORDER_RECOVERY_ON_STARTUP", True, raising=False)
    monkeypatch.setattr(settings_module.settings, "ORDER_RECOVERY_LOOKBACK_HOURS", 24, raising=False)
    monkeypatch.setattr(settings_module.settings, "ORDER_RECOVERY_STARTUP_LIMIT", 50, raising=False)
    monkeypatch.setattr(settings_module.settings, "UPSTOX_ACCESS_TOKEN", "TEST_TOKEN", raising=False)

    oid = state_module.create_order(
        broker="upstox",
        instrument_key="TEST_EQ|ABC",
        side="BUY",
        qty=1,
        order_kind="ENTRY",
        order_type="MARKET",
        status="NEW",
        meta={"tag": "startup:test"},
    )

    called: dict[str, object] = {}

    def fake_reconcile_upstox_orders(*, order_ids=None, limit=200):
        called["order_ids"] = list(order_ids or [])
        called["limit"] = limit
        return {"ok": True, "checked": len(called["order_ids"]), "updated": 0, "errors": []}

    monkeypatch.setattr(startup_module, "reconcile_upstox_orders", fake_reconcile_upstox_orders, raising=True)

    res = startup_module.run_startup_order_recovery()
    assert res.get("ok") is True
    assert oid in (called.get("order_ids") or [])
