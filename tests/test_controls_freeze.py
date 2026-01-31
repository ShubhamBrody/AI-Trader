from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_controls_freeze_blocks_new_orders() -> None:
    app = create_app()
    client = TestClient(app)

    # Enable freeze
    r1 = client.post("/api/controls/freeze", json={"enabled": True})
    assert r1.status_code == 200
    assert r1.json().get("controls", {}).get("freeze_new_orders") is True

    # Paper orders should be blocked
    r2 = client.post(
        "/api/orders/place",
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100},
    )
    assert r2.status_code == 403

    # Disable freeze
    r3 = client.post("/api/controls/freeze", json={"enabled": False})
    assert r3.status_code == 200
    assert r3.json().get("controls", {}).get("freeze_new_orders") is False

    # Now order is allowed
    r4 = client.post(
        "/api/orders/place",
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100},
    )
    assert r4.status_code == 200
    assert r4.json().get("order_state_id")
