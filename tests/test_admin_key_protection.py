from fastapi.testclient import TestClient

from app.main import create_app
from app.core.settings import settings


def test_admin_key_required_for_emergency_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_API_KEY", True, raising=False)
    monkeypatch.setattr(settings, "API_KEY_WRITE", "WRITE", raising=False)
    monkeypatch.setattr(settings, "API_KEY_READ", "READ", raising=False)
    monkeypatch.setattr(settings, "API_KEY_ADMIN", "ADMIN", raising=False)

    app = create_app()
    client = TestClient(app)

    # Write key is not sufficient for emergency.
    r1 = client.post("/api/controls/emergency/flatten", headers={"x-api-key": "WRITE"})
    assert r1.status_code == 401

    # Admin key works (paper broker flatten is allowed even in SAFE_MODE).
    r2 = client.post("/api/controls/emergency/flatten", headers={"x-api-key": "ADMIN"})
    assert r2.status_code in (200, 400)


def test_admin_key_required_for_agent_start(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_API_KEY", True, raising=False)
    monkeypatch.setattr(settings, "API_KEY_WRITE", "WRITE", raising=False)
    monkeypatch.setattr(settings, "API_KEY_ADMIN", "ADMIN", raising=False)

    app = create_app()
    client = TestClient(app)

    r1 = client.post("/api/agent/start", headers={"x-api-key": "WRITE"})
    assert r1.status_code == 401

    r2 = client.post("/api/agent/start", headers={"x-api-key": "ADMIN"})
    # Agent is disabled by config by default, but auth must pass.
    assert r2.status_code == 400
