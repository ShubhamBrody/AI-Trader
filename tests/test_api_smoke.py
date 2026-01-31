from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_safety_status_ok():
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/safety/status")
    assert r.status_code == 200
    body = r.json()
    assert "market_state" in body
    assert "trade_lock" in body


def test_chart_overlay_ok():
    app = create_app()
    client = TestClient(app)
    r = client.get(
        "/api/chart/overlay",
        params={
            "instrument_key": "NSE_EQ|INE002A01018",
            "interval": "1d",
            "lookback_days": 40,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["instrument_key"]
    assert "overlays" in body
    assert "trade" in body["overlays"]


def test_trade_state_ok():
    app = create_app()
    client = TestClient(app)
    r = client.get(
        "/api/trade-state",
        params={
            "instrument_key": "NSE_EQ|INE002A01018",
            "interval": "1d",
            "lookback_days": 60,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["instrument_key"]
    assert "decision" in body


def test_audit_endpoint_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/audit/recent")
    assert r.status_code == 200
    assert "events" in r.json()


def test_learning_status_no_model() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get(
        "/api/learning/status",
        params={"instrument_key": "NSE_EQ|INE002A01018", "interval": "1d", "horizon_steps": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] in (True, False)


def test_learning_jobs_endpoints_ok() -> None:
    app = create_app()
    client = TestClient(app)

    r1 = client.get("/api/learning/jobs")
    assert r1.status_code == 200
    assert r1.json().get("ok") is True

    r2 = client.post(
        "/api/learning/train-async",
        json={"instrument_key": "NSE_EQ|INE002A01018", "interval": "1d", "lookback_days": 365, "horizon_steps": 1},
    )
    assert r2.status_code == 200
    job_id = r2.json().get("job_id")
    assert isinstance(job_id, str) and job_id

    r3 = client.get(f"/api/learning/jobs/{job_id}")
    assert r3.status_code == 200
    assert r3.json().get("ok") is True


def test_automation_endpoints_ok() -> None:
    app = create_app()
    client = TestClient(app)

    r1 = client.get("/api/automation/status")
    assert r1.status_code == 200
    assert r1.json().get("ok") is True

    # By default AUTO_RETRAIN_ENABLED=false, so starting should be rejected.
    r2 = client.post("/api/automation/start")
    assert r2.status_code in (400, 422)

    r3 = client.get("/api/automation/runs")
    assert r3.status_code == 200
    assert r3.json().get("ok") is True


def test_prediction_endpoints_ok() -> None:
    app = create_app()
    client = TestClient(app)

    r1 = client.get("/api/prediction/status")
    assert r1.status_code == 200
    assert r1.json().get("ok") is True

    r2 = client.get("/api/prediction/events")
    assert r2.status_code == 200
    assert r2.json().get("ok") is True


def test_learning_deep_async_returns_501_without_torch() -> None:
    app = create_app()
    client = TestClient(app)

    r = client.post(
        "/api/learning/train-deep-async",
        json={"instrument_key": "NSE_EQ|INE002A01018", "interval": "1d", "lookback_days": 365, "horizon_steps": 1},
    )
    assert r.status_code in (200, 501)


def test_upstox_status_requires_token() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/upstox/status")
    # token is not configured in test env
    assert r.status_code in (400, 502)


def test_candles_auto_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get(
        "/api/candles/auto",
        params={"instrument_key": "NSE_EQ|INE002A01018", "live_interval": "1m", "lookback_minutes": 5, "eod_days": 30},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["instrument_key"]
    assert "candles" in body


def test_watchlist_crud_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r1 = client.post("/api/watchlist", json={"instrument_key": "NSE_EQ|INE002A01018", "label": "RELIANCE"})
    assert r1.status_code == 200
    r2 = client.get("/api/watchlist")
    assert r2.status_code == 200
    assert "items" in r2.json()
    r3 = client.delete("/api/watchlist", params={"instrument_key": "NSE_EQ|INE002A01018"})
    assert r3.status_code == 200


def test_news_endpoints_ok() -> None:
    app = create_app()
    client = TestClient(app)
    # refresh may be a no-op if feedparser is missing or network blocked; should still return JSON
    r1 = client.post("/api/news/refresh")
    assert r1.status_code == 200
    r2 = client.get("/api/news/latest")
    assert r2.status_code == 200
    assert "items" in r2.json()
    r3 = client.get("/api/news/market-sentiment")
    assert r3.status_code == 200


def test_portfolio_balance_ok() -> None:
    app = create_app()
    client = TestClient(app)

    r1 = client.get("/api/portfolio/balance")
    assert r1.status_code == 200
    body1 = r1.json()
    assert "paper_balance" in body1
    assert "live_balance" in body1
    assert "total_balance" in body1
    assert "updated_at" in body1
    assert body1["paper_balance"] is not None


def test_upstox_funds_requires_token() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/orders/upstox/funds")
    assert r.status_code in (400, 502)


def test_order_state_persists_paper_orders() -> None:
    app = create_app()
    client = TestClient(app)

    r1 = client.post(
        "/api/orders/place",
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1.get("order_state_id")

    r2 = client.get("/api/orders/state/list", params={"limit": 50, "broker": "paper"})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2.get("ok") is True
    assert any(x.get("id") == body1.get("order_state_id") for x in body2.get("items") or [])


def test_alerts_recent_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/alerts/recent")
    assert r.status_code == 200
    assert "events" in r.json()


def test_recommendations_top_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/recommendations/top", params={"n": 3, "min_confidence": 0.0, "max_risk": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    for item in body["results"]:
        assert "instrument_key" in item
        assert "confidence" in item
        assert "model_confidence" in item
        assert "confidence_breakdown" in item


def test_universe_and_agent_endpoints_ok() -> None:
    app = create_app()
    client = TestClient(app)

    r1 = client.get("/api/universe")
    assert r1.status_code == 200
    assert "items" in r1.json()

    r2 = client.get("/api/agent/status")
    assert r2.status_code == 200
    assert "running" in r2.json()

    r2b = client.get("/api/agent/trades/open")
    assert r2b.status_code == 200
    assert "trades" in r2b.json()

    r2c = client.get("/api/agent/trades/recent")
    assert r2c.status_code == 200
    assert "trades" in r2c.json()

    r2d = client.get("/api/agent/performance/today")
    assert r2d.status_code == 200
    assert r2d.json().get("ok") is True

    r2e = client.get("/api/agent/preview", params={"instrument_key": "NSE_EQ|INE002A01018"})
    assert r2e.status_code == 200
    assert r2e.json().get("ok") is True

    # By default AUTOTRADER_ENABLED=false, so start should fail cleanly.
    r3 = client.post("/api/agent/start")
    assert r3.status_code == 400
    detail = r3.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("ok") is False
    assert "AUTOTRADER_ENABLED=false" in str(detail.get("detail"))


def test_universe_endpoints_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r1 = client.post(
        "/api/universe",
        json={"instrument_key": "NSE_EQ|INE002A01018", "tradingsymbol": "RELIANCE", "cap_tier": "large"},
    )
    assert r1.status_code == 200
    r2 = client.get("/api/universe")
    assert r2.status_code == 200
    assert "items" in r2.json()


def test_agent_control_endpoints_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r1 = client.get("/api/agent/status")
    assert r1.status_code == 200
    # By default AUTOTRADER_ENABLED=false, so start should fail cleanly.
    r2 = client.post("/api/agent/start")
    assert r2.status_code == 400


def test_agent_run_once_ok() -> None:
    app = create_app()
    client = TestClient(app)

    r = client.post("/api/agent/run-once")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    # run-once can skip when market closed; still must return a stable shape
    assert "placed" in body or body.get("skipped") is True


def test_orders_paper_and_upstox_guard() -> None:
    app = create_app()
    client = TestClient(app)
    # Paper broker should work
    r1 = client.post(
        "/api/orders/place",
        json={"broker": "paper", "symbol": "DEMO", "side": "BUY", "qty": 1, "price": 100},
    )
    assert r1.status_code == 200

    # Upstox broker should be blocked while SAFE_MODE=true
    r2 = client.post(
        "/api/orders/place",
        json={
            "broker": "upstox",
            "upstox_body": {"dummy": "payload"},
        },
    )
    assert r2.status_code == 403
