from __future__ import annotations

from app.core import settings as settings_module
from app.portfolio import positions_state as pos_module


class FakeUpstoxClient:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def close(self) -> None:
        return

    def positions_v2(self):
        return {
            "status": "success",
            "data": [
                {
                    "instrument_token": "12345",
                    "net_qty": 10,
                    "avg_price": 100.5,
                    "last_price": 101.0,
                    "pnl": 5.0,
                }
            ],
        }


def test_reconcile_upstox_positions_writes_state(monkeypatch):
    monkeypatch.setattr(settings_module.settings, "UPSTOX_ACCESS_TOKEN", "TEST_TOKEN", raising=False)
    monkeypatch.setattr(pos_module, "UpstoxClient", FakeUpstoxClient, raising=False)

    res = pos_module.reconcile_upstox_positions()
    assert res.get("ok") is True

    items = pos_module.list_positions_state(broker="upstox", limit=50)
    assert items
    assert items[0].get("instrument_token") == "12345"
    assert float(items[0].get("net_qty") or 0.0) == 10.0
