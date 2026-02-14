from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _seed_resolved_predictions(*, instrument_key: str, interval: str, horizon_steps: int) -> None:
    from app.core.db import init_db
    from app.prediction.persistence import insert_prediction, resolve_prediction
    from app.prediction.calibration import update_ema

    init_db()

    now = int(time.time())

    # 10 resolved predictions:
    # - 3 TP (up/up), 3 TN (down/down)
    # - 2 FP (up/down), 2 FN (down/up)
    cases = [
        (+0.01, +0.008),
        (+0.02, +0.005),
        (+0.015, +0.001),
        (-0.01, -0.012),
        (-0.02, -0.004),
        (-0.005, -0.002),
        (+0.01, -0.006),
        (+0.02, -0.010),
        (-0.01, +0.006),
        (-0.02, +0.015),
    ]

    for i, (pred_adj, actual) in enumerate(cases):
        ts_pred = now - 3600 - i * 60
        ts_target = ts_pred + 60 * int(horizon_steps)
        pred_id = insert_prediction(
            ts_pred=int(ts_pred),
            instrument_key=str(instrument_key),
            interval=str(interval),
            horizon_steps=int(horizon_steps),
            model_kind="ridge",
            model_key=None,
            pred_ret=float(pred_adj),
            pred_ret_adj=float(pred_adj),
            calib_bias=0.0,
            ts_target=int(ts_target),
            meta={"seed": True, "i": i},
        )
        resolve_prediction(pred_id=int(pred_id), actual_ret=float(actual))

    # Seed a calibration entry to test /analytics/calibrations.
    key = f"{instrument_key}::{interval}::h{int(horizon_steps)}"
    update_ema(key, residual=0.01, abs_error=0.01, alpha=0.1)


def test_prediction_analytics_rollups_reliability_drift_and_calibrations():
    from app.main import app

    instrument_key = "NSE_EQ|TEST"
    interval = "1m"
    horizon_steps = 12

    _seed_resolved_predictions(instrument_key=instrument_key, interval=interval, horizon_steps=horizon_steps)

    client = TestClient(app)

    r = client.get("/api/prediction/analytics/rollups", params={"since_days": 365, "instrument_key": instrument_key, "interval": interval, "horizon_steps": horizon_steps})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert isinstance(data.get("items"), list)
    assert data["items"], "expected at least one rollup row"

    row = data["items"][0]
    assert row["instrument_key"] == instrument_key
    assert row["interval"] == interval
    assert int(row["horizon_steps"]) == int(horizon_steps)
    assert int(row["n"]) == 10
    assert abs(float(row["accuracy"]) - 0.6) < 1e-6

    conf = row.get("confusion") or {}
    assert int(conf.get("tp")) == 3
    assert int(conf.get("tn")) == 3
    assert int(conf.get("fp")) == 2
    assert int(conf.get("fn")) == 2

    r2 = client.get("/api/prediction/analytics/reliability", params={"since_days": 365, "instrument_key": instrument_key, "interval": interval, "horizon_steps": horizon_steps})
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2.get("ok") is True
    items2 = data2.get("items")
    assert isinstance(items2, list)
    assert sum(int(x.get("n") or 0) for x in items2) == 10

    r3 = client.get(
        "/api/prediction/analytics/drift",
        params={"baseline_days": 30, "recent_days": 7, "instrument_key": instrument_key, "interval": interval, "horizon_steps": horizon_steps},
    )
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3.get("ok") is True
    assert isinstance(data3.get("items"), list)

    r4 = client.get("/api/prediction/analytics/calibrations", params={"instrument_key": instrument_key, "interval": interval, "horizon_steps": horizon_steps})
    assert r4.status_code == 200
    data4 = r4.json()
    assert data4.get("ok") is True
    assert isinstance(data4.get("items"), list)
    assert any(str(x.get("key")) == f"{instrument_key}::{interval}::h{horizon_steps}" for x in data4["items"])
