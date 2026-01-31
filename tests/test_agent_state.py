from __future__ import annotations

from datetime import datetime, timezone

from app.agent.state import (
    count_open_trades,
    count_trades_today,
    has_open_trade,
    has_traded_today,
    list_open_trades,
    mark_trade_closed,
    realized_pnl_today,
    record_trade_open,
    update_trade,
)


def test_agent_trade_state_counts_and_pnl_today() -> None:
    tz_name = "Asia/Kolkata"
    now_utc = datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc)

    base_trades_today = count_trades_today(tz_name=tz_name, now_utc=now_utc)
    base_pnl_today = realized_pnl_today(tz_name=tz_name, now_utc=now_utc)

    instrument_key_1 = "TEST_EQ|STATE_1"
    instrument_key_2 = "TEST_EQ|STATE_2"

    t1 = record_trade_open(
        instrument_key=instrument_key_1,
        side="BUY",
        qty=1,
        entry=100.0,
        stop=95.0,
        target=110.0,
        entry_order_id=None,
        sl_order_id=None,
        monitor_app_stop=True,
        ts_open=int(now_utc.timestamp()) - 60,
        meta={"mode": "test"},
    )
    t2 = record_trade_open(
        instrument_key=instrument_key_2,
        side="SELL",
        qty=2,
        entry=200.0,
        stop=210.0,
        target=180.0,
        entry_order_id=None,
        sl_order_id=None,
        monitor_app_stop=False,
        ts_open=int(now_utc.timestamp()) - 30,
        meta={"mode": "test"},
    )

    assert t1 > 0
    assert t2 > 0

    assert count_trades_today(tz_name=tz_name, now_utc=now_utc) >= base_trades_today + 2
    assert has_open_trade(instrument_key=instrument_key_1) is True
    assert has_traded_today(instrument_key=instrument_key_1, tz_name=tz_name, now_utc=now_utc) is True

    open_before = count_open_trades()
    open_list = list_open_trades(limit=500)
    assert any(x["id"] == t1 for x in open_list)

    # Close trade 1 with a known pnl
    update_trade(t2, meta_patch={"note": "still_open"}, monitor_app_stop=True)

    mark_trade_closed(
        t1,
        reason="test_close",
        ts_close=int(now_utc.timestamp()) + 10,
        close_price=90.0,
        pnl=-10.0,
    )

    assert count_open_trades() <= open_before - 1
    assert realized_pnl_today(tz_name=tz_name, now_utc=now_utc) <= base_pnl_today - 10.0
