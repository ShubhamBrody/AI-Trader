from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.orders import state as state_module


def test_paper_place_idempotent_client_order_id() -> None:
    app = create_app()
    client = TestClient(app)

    client_order_id = "test-idem-paper-001"

    r1 = client.post(
        "/api/orders/place",
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100, "client_order_id": client_order_id},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1.get("order_state_id")

    r2 = client.post(
        "/api/orders/place",
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100, "client_order_id": client_order_id},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2.get("order_state_id") == body1.get("order_state_id")
    assert body2.get("idempotent_replay") is True

    st = state_module.get_order_by_client_order_id(client_order_id)
    assert st is not None
    assert st.id == body1.get("order_state_id")
