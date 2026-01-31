"""Smoke test for frontend API contracts.

Runs a minimal set of requests against the FastAPI app using TestClient and
asserts the response shapes the frontend expects.

Usage:
  python scripts/smoke_frontend_contract.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


def _assert_keys(obj: Any, keys: list[str], *, where: str) -> None:
    if not isinstance(obj, dict):
        raise AssertionError(f"{where}: expected dict, got {type(obj)}")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AssertionError(f"{where}: missing keys: {missing}")


def main() -> None:
    # Ensure repo root is on sys.path so `import app` works when running as a script.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Keep it deterministic/offline.
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("UPSTOX_STRICT", "false")
    os.environ.setdefault("SAFE_MODE", "true")

    from app.main import create_app

    app = create_app()
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    _assert_keys(r.json(), ["status", "gpu_available", "gpu_name"], where="/health")

    # News
    r = client.get("/api/news/recent", params={"limit": 5, "days": 7})
    assert r.status_code == 200
    body = r.json()
    _assert_keys(body, ["news"], where="/api/news/recent")
    assert isinstance(body["news"], list)

    # Recommendations
    r = client.get("/api/recommendations/top", params={"n": 3, "min_confidence": 0.0, "max_risk": 1.0})
    assert r.status_code == 200
    body = r.json()
    _assert_keys(body, ["recommendations", "count", "meta"], where="/api/recommendations/top")
    assert isinstance(body["recommendations"], list)

    # Intraday (may be empty)
    r = client.get("/api/intraday", params={"instrument_key": "NSE_EQ|INE002A01018", "interval": "1m"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # Intraday poll: can be blocked outside market hours; both shapes are OK.
    r = client.post(
        "/api/intraday/poll",
        params={"instrument_key": "NSE_EQ|INE002A01018", "interval": "1m", "lookback_minutes": 60},
    )
    assert r.status_code == 200
    body = r.json()
    _assert_keys(body, ["status"], where="/api/intraday/poll")

    # AI predict: should always provide the UI-friendly keys (even if it falls back).
    r = client.get("/api/ai/predict", params={"instrument_key": "NSE_EQ|INE002A01018"})
    assert r.status_code == 200
    body = r.json()
    _assert_keys(
        body,
        [
            "instrument_key",
            "timestamp",
            "predicted_ohlc",
            "action",
            "overall_confidence",
            "uncertainty",
            "ensemble_agreement",
            "reasons",
            "data_quality_score",
        ],
        where="/api/ai/predict",
    )
    assert isinstance(body["predicted_ohlc"], dict)

    # Paper trading: endpoints used by frontend.
    r = client.get("/api/paper/account")
    assert r.status_code == 200
    _assert_keys(r.json(), ["cash_balance"], where="/api/paper/account")

    r = client.post("/api/paper/deposit", params={"amount": 1000})
    assert r.status_code == 200
    _assert_keys(r.json(), ["status", "cash_balance"], where="/api/paper/deposit")

    r = client.get("/api/paper/positions")
    assert r.status_code == 200
    _assert_keys(r.json(), ["positions"], where="/api/paper/positions")

    # Execute can legitimately reject if no candle price; just validate shape.
    r = client.post(
        "/api/paper/execute",
        params={
            "instrument_key": "NSE_EQ|INE002A01018",
            "interval": "1m",
            "account_balance": 100000,
            "lot_size": 1,
        },
    )
    assert r.status_code == 200
    _assert_keys(r.json(), ["status"], where="/api/paper/execute")

    r = client.get("/api/paper/journal")
    assert r.status_code == 200
    _assert_keys(r.json(), ["trades"], where="/api/paper/journal")

    # Trade decide
    r = client.get(
        "/api/trade/decide",
        params={"instrument_key": "NSE_EQ|INE002A01018", "interval": "1m", "account_balance": 100000, "lot_size": 1},
    )
    assert r.status_code == 200
    _assert_keys(r.json(), ["instrument_key", "action", "quantity"], where="/api/trade/decide")

    print("OK: frontend contract smoke test passed")


if __name__ == "__main__":
    main()
