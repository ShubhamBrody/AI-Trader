import sys
import types

from app.ai.engine import AIEngine
from app.candles.models import Candle, CandleSeries


def test_ai_engine_prefers_deep_model(monkeypatch) -> None:
    # Provide a fake deep_service module so this test doesn't require torch.
    fake = types.ModuleType("app.learning.deep_service")

    class DummyDeep:
        metrics = {"val_huber": 0.1}

    def load_deep_model(*args, **kwargs):
        return DummyDeep()

    def predict_deep_return(model, **kwargs):
        return {"ok": True, "predicted_return": 0.01, "signal": "BUY", "confidence": 0.9, "device": "cpu"}

    fake.load_deep_model = load_deep_model
    fake.predict_deep_return = predict_deep_return

    monkeypatch.setitem(sys.modules, "app.learning.deep_service", fake)

    # Make ridge path unavailable.
    import app.ai.engine as engine_mod

    monkeypatch.setattr(engine_mod, "load_model", lambda *a, **k: None)

    # Stub candles/universe to keep this unit test deterministic.
    series = CandleSeries(
        instrument_key="NSE_EQ|INE002A01018",
        interval="1d",
        candles=[
            Candle(ts=1 + i, open=100 + i, high=101 + i, low=99 + i, close=100 + i * 0.1, volume=1000)
            for i in range(200)
        ],
    )

    eng = AIEngine()
    monkeypatch.setattr(eng._candles, "load_historical", lambda *a, **k: series)
    monkeypatch.setattr(eng._universe, "get_cap_tier", lambda *a, **k: "large")

    out = eng.predict("NSE_EQ|INE002A01018", interval="1d", lookback_days=60, horizon_steps=1)
    assert out["meta"]["model"] == "torch_transformer_regression_v1"
    assert out["prediction"]["signal"] == "BUY"
    assert out["prediction"]["confidence"] >= 0.9
