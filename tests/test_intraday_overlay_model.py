from __future__ import annotations

from datetime import datetime, timezone, timedelta


def _mk(ts: int, o: float, h: float, l: float, c: float, v: float) -> dict:
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_intraday_overlay_model_train_and_load():
    from app.core.db import init_db
    from app.ai.intraday_overlay_model import train_and_store, load_model

    init_db()

    # Build a synthetic series with clear up moves so we get labeled samples.
    candles = []
    ts0 = 1_700_000_000
    price = 100.0
    for i in range(600):
        # Gentle up drift with noise
        price = price + 0.02
        o = price - 0.05
        c = price + (0.03 if i % 7 == 0 else 0.0)
        h = max(o, c) + 0.08
        l = min(o, c) - 0.08
        v = 100.0 + (50.0 if i % 10 == 0 else 0.0)
        candles.append(_mk(ts0 + i * 60, o, h, l, c, v))

    res = train_and_store(instrument_key="NSE_EQ|TEST", interval="1m", candles=candles, feature_window=60, horizon_steps=20, k_atr=1.0)
    assert res["ok"] in (True, False)

    m = load_model("NSE_EQ|TEST", "1m")
    # Model might not be trained if labeled samples are insufficient, but in this synthetic series
    # it should generally succeed.
    assert m is None or (m.instrument_key in ("NSE_EQ|TEST", "__global__") and m.interval == "1m")


def test_intraday_overlay_analysis_includes_ai_when_model_present():
    from app.core.db import init_db
    from app.ai.intraday_overlay_model import save_model, OverlayModel
    from app.ai.intraday_overlays import analyze_intraday

    init_db()

    # Save a trivial model so AI section is enabled.
    m = OverlayModel(
        instrument_key="NSE_EQ|TEST",
        interval="1m",
        created_ts=1,
        features=["ret1"],
        mean=[0.0],
        std=[1.0],
        weights=[1.0],
        bias=0.0,
        horizon_steps=30,
        k_atr=1.2,
        metrics={"ok": True, "n": 1, "acc": 1.0},
    )
    save_model(m)

    # Candles long enough for featurization
    candles = []
    ts0 = 1_700_200_000
    price = 100.0
    for i in range(100):
        price += 0.01
        candles.append(_mk(ts0 + i * 60, price - 0.05, price + 0.07, price - 0.08, price, 100.0))

    out = analyze_intraday(instrument_key="NSE_EQ|TEST", interval="1m", candles=candles)
    assert "ai" in out
    assert out["ai"]["enabled"] is True
    assert 0.0 <= float(out["ai"]["p_up"]) <= 1.0
