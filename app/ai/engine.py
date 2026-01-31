from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np

from app.ai.feature_engineering import compute_features
from app.candles.service import CandleService
from app.learning.service import load_model
from app.universe.service import UniverseService


class AIEngine:
    def __init__(self) -> None:
        self._candles = CandleService()
        self._universe = UniverseService()

    def predict(
        self,
        instrument_key: str,
        interval: str = "1d",
        lookback_days: int = 60,
        horizon_steps: int = 1,
        include_nifty: bool = False,
        *,
        model_family: str | None = None,
        cap_tier: str | None = None,
    ) -> dict:
        # Pull some recent history (stub uses generated candles if DB empty)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(lookback_days))
        series = self._candles.load_historical(instrument_key, interval, start, end)

        closes = [c.close for c in series.candles][-max(10, int(lookback_days)) :]
        feats = compute_features(closes)

        if feats.last_close <= 0:
            return {
                "instrument_key": instrument_key,
                "error": "no data",
                "confidence": 0.0,
                "uncertainty": 1.0,
            }

        # Prefer learned model if available (trained post-market), else fall back to simple statistical stub.
        fam = (model_family or ("intraday" if interval.endswith("m") else "long"))
        cap = cap_tier or self._universe.get_cap_tier(instrument_key)

        # 1) Prefer deep model if present and torch is installed.
        deep_signal: str | None = None
        deep_conf: float | None = None
        used_deep = False
        trained = load_model(instrument_key, interval, horizon_steps=horizon_steps, model_family=fam, cap_tier=cap)

        try:
            from app.learning.deep_service import load_deep_model, predict_deep_return

            deep = load_deep_model(instrument_key, interval, horizon_steps=horizon_steps, model_family=fam, cap_tier=cap)
            if deep is not None:
                closes2 = [c.close for c in series.candles]
                highs2 = [c.high for c in series.candles]
                lows2 = [c.low for c in series.candles]
                vols2 = [c.volume for c in series.candles]
                out = predict_deep_return(deep, closes=closes2, highs=highs2, lows=lows2, volumes=vols2)
                if out.get("ok"):
                    forecast_ret = float(out["predicted_return"])
                    model_name = "torch_transformer_regression_v1"
                    extra_metrics = deep.metrics
                    deep_signal = str(out.get("signal") or "") or None
                    deep_conf = float(out.get("confidence") or 0.0)
                    used_deep = True
        except Exception:
            pass

        if used_deep:
            pass
        elif trained is not None:
            forecast_ret = float(trained.predict_return(closes))
            model_name = "ridge_regression_v1"
            extra_metrics = trained.metrics
        else:
            # Simple statistical forecast: next return is mean of recent returns, shrunk by volatility.
            arr = np.asarray(closes, dtype=float)
            rets = np.diff(arr) / np.where(arr[:-1] == 0, 1e-9, arr[:-1])
            mean_ret = float(np.mean(rets[-10:])) if rets.size >= 1 else 0.0
            vol = float(np.std(rets[-20:])) if rets.size >= 2 else max(feats.volatility20, 1e-6)

            shrink = 1.0 / (1.0 + 50.0 * vol)
            forecast_ret = mean_ret * shrink
            model_name = "statistical_stub_v1"
            extra_metrics = None

        # Derive volatility for confidence heuristics
        arr = np.asarray(closes, dtype=float)
        rets = np.diff(arr) / np.where(arr[:-1] == 0, 1e-9, arr[:-1])
        vol = float(np.std(rets[-20:])) if rets.size >= 2 else max(feats.volatility20, 1e-6)

        next_close = feats.last_close * (1.0 + forecast_ret)
        next_open = feats.last_close
        spread = max(0.001, 2.0 * vol) * feats.last_close
        next_high = max(next_open, next_close) + spread
        next_low = min(next_open, next_close) - spread

        # Classification
        if deep_signal in {"BUY", "SELL", "HOLD"}:
            action = str(deep_signal)
        else:
            if forecast_ret > 0.002 and feats.rsi14 < 70:
                action = "BUY"
            elif forecast_ret < -0.002 and feats.rsi14 > 30:
                action = "SELL"
            else:
                action = "HOLD"

        # Confidence/uncertainty heuristics
        uncertainty = float(min(1.0, max(0.0, vol * 10.0)))
        confidence = float(max(0.0, min(1.0, (1.0 - uncertainty) * (0.5 + abs(forecast_ret) * 50.0))))
        if deep_conf is not None:
            confidence = float(max(confidence, min(1.0, max(0.0, deep_conf))))

        agreement = float(max(0.0, min(1.0, 0.6 + (0.2 if action != "HOLD" else 0.0) - uncertainty * 0.2)))
        data_quality = float(max(0.0, min(1.0, len(closes) / 60.0)))

        return {
            "instrument_key": instrument_key,
            "interval": interval,
            "horizon_steps": int(horizon_steps),
            "include_nifty": include_nifty,
            "prediction": {
                "next_hour_ohlc": {
                    "open": float(next_open),
                    "high": float(next_high),
                    "low": float(next_low),
                    "close": float(next_close),
                },
                "signal": action,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "ensemble_agreement": agreement,
            },
            "features": asdict(feats),
            "meta": {
                "data_quality": data_quality,
                "model": model_name,
                "model_metrics": extra_metrics,
            },
        }
