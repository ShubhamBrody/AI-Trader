from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _seed_predictions(n: int) -> None:
    from app.core.db import init_db
    from app.prediction.persistence import insert_prediction

    init_db()

    now = int(time.time())
    instrument_key = "NSE_EQ|RET"
    interval = "1m"
    horizon_steps = 5

    for i in range(n):
        ts_pred = now - i * 60
        ts_target = ts_pred + 60 * horizon_steps
        insert_prediction(
            ts_pred=int(ts_pred),
            instrument_key=instrument_key,
            interval=interval,
            horizon_steps=horizon_steps,
            model_kind="ridge",
            model_key=None,
            pred_ret=0.001,
            pred_ret_adj=0.001,
            calib_bias=0.0,
            ts_target=int(ts_target),
            meta={"seed": True, "i": i},
        )

    # Add a much older row we can delete by age.
    old_ts = now - 10 * 24 * 3600
    insert_prediction(
        ts_pred=int(old_ts),
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=horizon_steps,
        model_kind="ridge",
        model_key=None,
        pred_ret=0.001,
        pred_ret_adj=0.001,
        calib_bias=0.0,
        ts_target=int(old_ts + 60 * horizon_steps),
        meta={"seed": True, "old": True},
    )


def test_prediction_retention_cleanup_endpoint_deletes_by_age_and_max_rows():
    from app.main import app
    from app.core.db import db_conn

    _seed_predictions(12)

    client = TestClient(app)

    r = client.post("/api/prediction/retention/cleanup", params={"max_age_days": 7, "max_rows": 5})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert int(data.get("max_age_days")) == 7
    assert int(data.get("max_rows")) == 5
    assert int(data.get("remaining_rows")) == 5

    with db_conn() as conn:
        n = int(conn.execute("SELECT COUNT(*) FROM prediction_events").fetchone()[0])
    assert n == 5
