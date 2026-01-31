from fastapi.testclient import TestClient

from app.main import create_app
from app.core.settings import settings


def test_api_key_required_for_protected_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_API_KEY", True, raising=False)
    monkeypatch.setattr(settings, "API_KEY", "TESTKEY", raising=False)

    app = create_app()
    client = TestClient(app)

    # Protected: orders
    r = client.post("/api/orders/place", json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100})
    assert r.status_code == 401

    r2 = client.post(
        "/api/orders/place",
        headers={"x-api-key": "TESTKEY"},
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100},
    )
    assert r2.status_code == 200

    # Protected: controls (write)
    r3 = client.post("/api/controls/freeze", json={"enabled": True})
    assert r3.status_code == 401

    r4 = client.post("/api/controls/freeze", headers={"x-api-key": "TESTKEY"}, json={"enabled": True})
    assert r4.status_code == 200
