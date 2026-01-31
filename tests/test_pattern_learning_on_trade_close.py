from __future__ import annotations

import json

from app.agent.state import mark_trade_closed, record_trade_open
from app.core.db import db_conn


def test_pattern_learning_triggers_on_trade_close():
    trade_id = record_trade_open(
        instrument_key="NSE_EQ|TEST",
        side="BUY",
        qty=10,
        entry=100.0,
        stop=99.0,
        target=102.0,
        entry_order_id=None,
        sl_order_id=None,
        monitor_app_stop=True,
        meta={
            "overlays": {
                "interval": "1m",
                "asof_ts": 1700000000,
                "candle_patterns": [
                    {"name": "Hammer", "side": "buy", "confidence": 0.8, "details": {"weight": 1.0}},
                    {"name": "Bullish Engulfing", "side": "buy", "confidence": 0.7, "details": {"weight": 1.0}},
                ],
                "trade": {"pattern_hint": {"name": "Hammer", "side": "buy", "confidence": 0.8}},
            }
        },
    )

    # Close profitable -> good feedback.
    mark_trade_closed(int(trade_id), reason="target_hit", close_price=101.0)

    with db_conn() as conn:
        # Feedback rows should be recorded.
        n = conn.execute("SELECT COUNT(1) AS c FROM candle_pattern_feedback").fetchone()["c"]
        assert int(n) >= 1

        # Weights should exist for the patterns we fed.
        row = conn.execute("SELECT weight, seen, wins, losses FROM candle_pattern_stats WHERE pattern='Hammer'").fetchone()
        assert row is not None
        assert float(row["weight"]) != 1.0 or int(row["seen"]) >= 1
        assert int(row["wins"]) + int(row["losses"]) == int(row["seen"])

        # Trade meta should contain close_price.
        t = conn.execute("SELECT meta_json FROM agent_trades WHERE id=?", (int(trade_id),)).fetchone()
        meta = json.loads(t["meta_json"])
        assert float(meta["close_price"]) == 101.0
