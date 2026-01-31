from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_agent_disabled_blocks_start() -> None:
    app = create_app()
    client = TestClient(app)

    r1 = client.post("/api/controls/agent/disabled", json={"disabled": True})
    assert r1.status_code == 200
    assert r1.json().get("controls", {}).get("agent_disabled") is True

    r2 = client.post("/api/agent/start")
    # agent start returns 400 if not ok
    assert r2.status_code in (400, 422)
