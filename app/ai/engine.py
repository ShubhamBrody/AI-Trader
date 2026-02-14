from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np

from app.ai.feature_engineering import compute_features
from app.ai.trend_confluence import analyze_multi_timeframe_confluence
from app.candles.service import CandleService
from app.core.settings import settings
from app.learning.service import load_model
from app.universe.service import UniverseService


class AIEngine:
    def __init__(self) -> None:
        self._candles = CandleService()
        self._universe = UniverseService()

    def analyze_trend_confluence(self, instrument_key: str) -> dict:
        """Multi-timeframe trend confluence analysis.

        Uses the classic combo:
        - SMA(50/200)
        - MACD(12,26,9) crossover + sign
        - RSI(14) around the 50 line (avoid >70 / <30)
        - ADX(14) with +DI/-DI for direction (avoid <20)
        - Volume > EMA(10)

        Timeframes (weighted): 1d (3x), 4h (2x), 1h (1x).
        """

        end = datetime.now(timezone.utc)

        d_days = int(getattr(settings, "TRADER_TREND_CONFLUENCE_DAILY_DAYS", 420) or 420)
        h4_days = int(getattr(settings, "TRADER_TREND_CONFLUENCE_H4_DAYS", 90) or 90)
        h1_days = int(getattr(settings, "TRADER_TREND_CONFLUENCE_H1_DAYS", 30) or 30)

        daily_s = self._candles.load_historical(str(instrument_key), "1d", end - timedelta(days=d_days), end)
        h4_s = self._candles.load_historical(str(instrument_key), "4h", end - timedelta(days=h4_days), end)
        h1_s = self._candles.load_historical(str(instrument_key), "1h", end - timedelta(days=h1_days), end)

        def _pack(series) -> dict[str, list[float]]:
            cs = list(series.candles or [])
            return {
                "highs": [float(c.high) for c in cs],
                "lows": [float(c.low) for c in cs],
                "closes": [float(c.close) for c in cs],
                "volumes": [float(getattr(c, "volume", 0.0) or 0.0) for c in cs],
            }

        return analyze_multi_timeframe_confluence(daily=_pack(daily_s), h4=_pack(h4_s), h1=_pack(h1_s))

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

        all_closes = [float(c.close) for c in (series.candles or [])]
        # Feature computations should have enough samples; use a sensible tail window.
        if str(interval).endswith("m"):
            tail_n = 600  # ~10h of 1m candles; enough for RSI/SMA/EMA/volatility stability
        elif str(interval).endswith("h"):
            tail_n = 400
        else:
            tail_n = 200
        closes = all_closes[-min(len(all_closes), max(10, int(tail_n))):]
        feats = compute_features(closes)

        if feats.last_close <= 0:
            return {
                "instrument_key": instrument_key,
                "interval": interval,
                "horizon_steps": int(horizon_steps),
                "include_nifty": include_nifty,
                "prediction": {
                    "next_hour_ohlc": {"open": None, "high": None, "low": None, "close": None},
                    "signal": "HOLD",
                    "position_side": "FLAT",
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "ensemble_agreement": 0.0,
                },
                "features": asdict(feats),
                "meta": {
                    "data_quality": 0.0,
                    "model": "no_data",
                    "model_metrics": None,
                    "note": "no candle history available",
                },
            }

        # Prefer learned model if available (trained post-market).
        # If no explicit suffix is configured, also try the common "_lightweight" family
        # so models produced by scripts/train_all_models_lightweight.py are discoverable.
        fam_base = (model_family or ("intraday" if interval.endswith("m") else "long"))
        suf = str(getattr(settings, "MODEL_FAMILY_SUFFIX", "") or "")
        fam_candidates: list[str] = []
        if suf:
            if not suf.startswith("_"):
                suf = "_" + suf
            fam_candidates.append(f"{str(fam_base)}{suf}")
        else:
            fam_candidates.append(str(fam_base))
            fam_candidates.append(f"{str(fam_base)}_lightweight")
        cap = cap_tier or self._universe.get_cap_tier(instrument_key)

        # 1) Prefer deep model if present and torch is installed.
        deep_signal: str | None = None
        deep_conf: float | None = None
        used_deep = False
        trained = None
        used_family: str | None = None
        for fam in fam_candidates:
            trained = load_model(instrument_key, interval, horizon_steps=horizon_steps, model_family=fam, cap_tier=cap)
            if trained is not None:
                used_family = fam
                break

        try:
            from app.learning.deep_service import load_deep_model, predict_deep_return

            deep = None
            deep_family: str | None = None
            for fam in fam_candidates:
                deep = load_deep_model(instrument_key, interval, horizon_steps=horizon_steps, model_family=fam, cap_tier=cap)
                if deep is not None:
                    deep_family = fam
                    break

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
                    used_family = deep_family or used_family
        except Exception:
            pass

        if used_deep:
            pass
        elif trained is not None:
            forecast_ret = float(trained.predict_return(all_closes if all_closes else closes))
            model_name = "ridge_regression_v1"
            extra_metrics = trained.metrics
        else:
            # Rule-based (explainable) fallback when no trained model is available.
            # Goal: produce a stable directional signal (BUY/SELL/HOLD) with bounded magnitude.
            arr = np.asarray(all_closes if all_closes else closes, dtype=float)
            rets = np.diff(arr) / np.where(arr[:-1] == 0, 1e-9, arr[:-1])
            mean_ret = float(np.mean(rets[-10:])) if rets.size >= 1 else 0.0
            vol = float(np.std(rets[-20:])) if rets.size >= 2 else max(feats.volatility20, 1e-6)

            # Momentum over recent window.
            mom_window = min(6, int(arr.size))
            if mom_window >= 2:
                momentum = float(arr[-1] / max(1e-9, arr[-mom_window]) - 1.0)
            else:
                momentum = 0.0

            # Normalized biases.
            rsi_bias = float((feats.rsi14 - 50.0) / 50.0)  # -1..+1
            trend_bias = float((feats.last_close - feats.ema20) / max(1e-9, feats.last_close))  # ~-1..+1

            # Shrink in high volatility regimes.
            shrink = 1.0 / (1.0 + 35.0 * vol)

            raw_score = (
                0.45 * mean_ret
                + 0.35 * momentum
                + 0.15 * trend_bias
                + 0.05 * rsi_bias * 0.01
            )
            # Bound the forecast to avoid absurd OHLC projections.
            forecast_ret = float(np.tanh(raw_score * 3.0) * 0.01) * float(shrink)
            model_name = "rules_fallback_v1"
            extra_metrics = {
                "vol": float(vol),
                "momentum": float(momentum),
                "trend_bias": float(trend_bias),
                "rsi_bias": float(rsi_bias),
            }

        # Derive volatility for confidence heuristics
        arr = np.asarray(all_closes if all_closes else closes, dtype=float)
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

        position_side = "LONG" if action == "BUY" else "SHORT" if action == "SELL" else "FLAT"

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
                "position_side": position_side,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "ensemble_agreement": agreement,
            },
            "features": asdict(feats),
            "meta": {
                "data_quality": data_quality,
                "model": model_name,
                "model_family": used_family,
                "model_metrics": extra_metrics,
            },
        }
