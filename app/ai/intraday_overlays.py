from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.ai.candlestick_patterns import CATALOG as CANDLESTICK_PATTERN_CATALOG
from app.ai.candlestick_patterns import detect_patterns as detect_candlestick_patterns
from app.ai.candlestick_patterns import to_candles as to_pattern_candles
from app.ai.intraday_overlay_model import featurize, load_model
from app.ai.pattern_memory import ensure_catalog_rows, get_weights_cached
from app.core.settings import settings
from app.learning.service import load_model as load_ridge_model
from app.universe.service import UniverseService


_PATTERN_ROWS_READY = False


def _ensure_pattern_rows() -> None:
    global _PATTERN_ROWS_READY
    if _PATTERN_ROWS_READY:
        return
    ensure_catalog_rows(
        [
            {
                "name": p.name,
                "family": p.family,
                "side": p.side,
                "base_reliability": float(p.base_reliability),
                "params": {"min_window": int(p.min_window), "best_timeframe": p.best_timeframe},
            }
            for p in CANDLESTICK_PATTERN_CATALOG
        ]
    )
    _PATTERN_ROWS_READY = True


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _candle_get(c: Any, k: str, default: Any = None) -> Any:
    if isinstance(c, dict):
        return c.get(k, default)
    return getattr(c, k, default)


def _sorted_candles(candles: list[Any]) -> list[Any]:
    return sorted(list(candles or []), key=lambda c: _i(_candle_get(c, "ts", 0)))


def compute_atr(candles: list[Any], period: int = 14) -> float:
    cs = _sorted_candles(candles)
    if len(cs) < 2:
        return 0.0

    trs: list[float] = []
    for prev, cur in zip(cs[:-1], cs[1:]):
        ph = _f(_candle_get(prev, "high"))
        pl = _f(_candle_get(prev, "low"))
        pc = _f(_candle_get(prev, "close"))
        ch = _f(_candle_get(cur, "high"))
        cl = _f(_candle_get(cur, "low"))
        tr = max(ch - cl, abs(ch - pc), abs(cl - pc), ph - pl)
        trs.append(float(max(0.0, tr)))

    if not trs:
        return 0.0

    n = min(int(period), len(trs))
    window = trs[-n:]
    return float(sum(window) / max(1, len(window)))


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r2)."""
    n = len(xs)
    if n < 2:
        return 0.0, float(ys[0] if ys else 0.0), 0.0

    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, float(sy / n), 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    yhat = [slope * x + intercept for x in xs]
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return float(slope), float(intercept), float(r2)


def compute_trend(candles: list[Any], window: int = 30) -> dict[str, Any]:
    cs = _sorted_candles(candles)
    if len(cs) < 5:
        return {"dir": "flat", "slope": 0.0, "strength": 0.0, "window": len(cs)}

    cs = cs[-max(5, int(window)) :]
    closes = [_f(_candle_get(c, "close")) for c in cs]
    xs = list(range(len(closes)))
    slope, _, r2 = _linreg([float(x) for x in xs], closes)
    last = closes[-1] or 1.0
    slope_pct = float((slope / last) * 100.0)

    if abs(slope_pct) < 0.02:
        d = "flat"
    elif slope_pct > 0:
        d = "up"
    else:
        d = "down"

    return {"dir": d, "slope": float(slope), "slope_pct": slope_pct, "strength": float(r2), "window": len(cs)}


@dataclass(frozen=True)
class Levels:
    support: list[float]
    resistance: list[float]


def _pivot_points(candles: list[Any], span: int = 3) -> tuple[list[float], list[float]]:
    cs = _sorted_candles(candles)
    if len(cs) < (2 * span + 3):
        return [], []

    highs: list[float] = []
    lows: list[float] = []
    for i in range(span, len(cs) - span):
        w = cs[i - span : i + span + 1]
        hi = _f(_candle_get(cs[i], "high"))
        lo = _f(_candle_get(cs[i], "low"))
        if hi <= 0 or lo <= 0:
            continue
        if hi == max(_f(_candle_get(x, "high")) for x in w):
            highs.append(hi)
        if lo == min(_f(_candle_get(x, "low")) for x in w):
            lows.append(lo)
    return highs, lows


def _cluster_levels(values: list[float], *, tol_abs: float, max_levels: int) -> list[float]:
    vals = [float(v) for v in values if v > 0]
    if not vals:
        return []
    vals.sort()

    clusters: list[list[float]] = []
    for v in vals:
        placed = False
        for cl in clusters:
            m = sum(cl) / len(cl)
            if abs(v - m) <= tol_abs:
                cl.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])

    # Rank by frequency then proximity to median
    med = vals[len(vals) // 2]
    clusters.sort(key=lambda cl: (-len(cl), abs((sum(cl) / len(cl)) - med)))
    out = [float(sum(cl) / len(cl)) for cl in clusters[: int(max_levels)]]
    out.sort()
    return out


def compute_support_resistance(
    candles: list[Any],
    *,
    max_levels: int = 3,
    pivot_span: int = 3,
    atr_period: int = 14,
) -> Levels:
    cs = _sorted_candles(candles)
    if len(cs) < 10:
        return Levels(support=[], resistance=[])

    atr = compute_atr(cs, period=atr_period)
    last_close = _f(_candle_get(cs[-1], "close"), 1.0) or 1.0
    tol_abs = max(atr * 0.6, last_close * 0.003)  # 0.3% or ~0.6 ATR

    highs, lows = _pivot_points(cs, span=int(pivot_span))
    supp = _cluster_levels(lows, tol_abs=tol_abs, max_levels=int(max_levels))
    res = _cluster_levels(highs, tol_abs=tol_abs, max_levels=int(max_levels))
    return Levels(support=supp, resistance=res)


def _nearest_above(levels: list[float], price: float) -> float | None:
    for lv in sorted(levels):
        if lv > price:
            return float(lv)
    return None


def _nearest_below(levels: list[float], price: float) -> float | None:
    for lv in sorted(levels, reverse=True):
        if lv < price:
            return float(lv)
    return None


def analyze_intraday(
    *,
    instrument_key: str,
    interval: str,
    candles: list[Any],
) -> dict[str, Any]:
    cs = _sorted_candles(candles)
    if len(cs) < 5:
        return {
            "instrument_key": instrument_key,
            "interval": interval,
            "asof_ts": _i(_candle_get(cs[-1], "ts")) if cs else None,
            "n": len(cs),
            "atr": 0.0,
            "buffer": 0.0,
            "trend": {"dir": "flat", "slope": 0.0, "strength": 0.0, "window": len(cs)},
            "levels": {"support": [], "resistance": []},
            "patterns": [],
            "candle_patterns": [],
            "trade": None,
        }

    last = cs[-1]
    prev = cs[-2]
    last_close = _f(_candle_get(last, "close"))
    prev_close = _f(_candle_get(prev, "close"))
    last_ts = _i(_candle_get(last, "ts"))

    atr = compute_atr(cs, period=14)
    trend = compute_trend(cs, window=30)
    levels = compute_support_resistance(cs, max_levels=3, pivot_span=3, atr_period=14)

    buffer = max(atr * 0.25, last_close * 0.0008)  # ~0.08% minimum
    nearest_res = _nearest_above(levels.resistance, prev_close) or _nearest_above(levels.resistance, last_close)
    nearest_sup = _nearest_below(levels.support, prev_close) or _nearest_below(levels.support, last_close)

    patterns: list[dict[str, Any]] = []
    trade: dict[str, Any] | None = None

    # -------------------------------
    # AI-first price forecast (ridge if available; else rules fallback)
    # -------------------------------
    closes_for_ai = [_f(_candle_get(c, "close")) for c in cs]

    def _interval_minutes(iv: str) -> int:
        s = str(iv or "").strip().lower()
        if s.endswith("m"):
            try:
                return max(1, int(s[:-1]))
            except Exception:
                return 1
        if s.endswith("h"):
            try:
                return max(1, int(s[:-1]) * 60)
            except Exception:
                return 60
        return 1

    iv_min = _interval_minutes(interval)
    # Prefer common horizons used by lightweight training (e.g., 1m -> h60).
    horizon_candidates: list[int]
    if iv_min <= 1:
        horizon_candidates = [60, 30, 15, 5, 1]
    elif iv_min <= 3:
        horizon_candidates = [20, 10, 5, 1]
    elif iv_min <= 5:
        horizon_candidates = [12, 6, 3, 1]
    elif iv_min <= 15:
        horizon_candidates = [4, 2, 1]
    else:
        horizon_candidates = [1]

    fam_base = "intraday" if str(interval).endswith("m") else "long"
    suf = str(getattr(settings, "MODEL_FAMILY_SUFFIX", "") or "")
    fam_candidates: list[str] = []
    if suf:
        if not suf.startswith("_"):
            suf = "_" + suf
        fam_candidates.append(f"{fam_base}{suf}")
    else:
        fam_candidates.append(str(fam_base))
        fam_candidates.append(f"{fam_base}_lightweight")

    try:
        cap = UniverseService().get_cap_tier(instrument_key)
    except Exception:
        cap = None

    ai_forecast_ret: float | None = None
    ai_model_name: str = "rules_fallback_v1"
    ai_model_family: str | None = None
    ai_horizon: int | None = None
    ai_metrics: dict[str, Any] | None = None

    # Try learned ridge model first.
    for h in horizon_candidates:
        for fam in fam_candidates:
            try:
                m = load_ridge_model(instrument_key, interval, horizon_steps=int(h), model_family=fam, cap_tier=cap)
                if m is None:
                    continue
                ai_forecast_ret = float(m.predict_return(closes_for_ai))
                ai_model_name = "ridge_regression_v1"
                ai_model_family = fam
                ai_horizon = int(h)
                ai_metrics = {"model_key": m.model_key, "trained_ts": int(m.trained_ts), "metrics": dict(m.metrics or {})}
                break
            except Exception:
                continue
        if ai_forecast_ret is not None:
            break

    # If no model, compute a bounded fallback return from recent momentum/mean return.
    if ai_forecast_ret is None:
        arr = [float(x) for x in closes_for_ai if float(x) > 0]
        if len(arr) >= 3:
            rets = [(arr[i] / max(1e-9, arr[i - 1]) - 1.0) for i in range(1, len(arr))]
            mean_ret = float(sum(rets[-10:]) / max(1, min(10, len(rets))))
            # Momentum over last ~6 candles.
            mom_window = min(6, len(arr))
            momentum = float(arr[-1] / max(1e-9, arr[-mom_window]) - 1.0) if mom_window >= 2 else 0.0
            # Use ATR-derived vol proxy when available.
            vol = 0.0
            if rets and len(rets) >= 2:
                mu = float(sum(rets[-20:]) / max(1, min(20, len(rets))))
                var = float(sum((r - mu) ** 2 for r in rets[-20:]) / max(1, min(20, len(rets))))
                vol = float(max(0.0, var) ** 0.5)
            if vol <= 0 and last_close > 0 and atr > 0:
                vol = float(max(1e-6, atr / last_close))
            shrink = 1.0 / (1.0 + 40.0 * float(vol))
            raw_score = 0.55 * mean_ret + 0.45 * momentum
            ai_forecast_ret = float(math.tanh(raw_score * 3.0) * 0.008) * float(shrink)
        else:
            ai_forecast_ret = 0.0

    # Confidence/uncertainty heuristics for overlays.
    # Note: keep these stable for UI even when learned model is missing.
    # Vol proxy for uncertainty.
    vol_proxy = 0.0
    try:
        if last_close > 0 and atr > 0:
            vol_proxy = float(max(1e-6, atr / last_close))
    except Exception:
        vol_proxy = 0.0
    uncertainty = float(max(0.0, min(1.0, vol_proxy * 8.0)))

    dir_acc = None
    try:
        if isinstance(ai_metrics, dict):
            mm = ai_metrics.get("metrics") or {}
            dir_acc = float(mm.get("direction_acc")) if "direction_acc" in mm else None
    except Exception:
        dir_acc = None

    acc_score = 0.0
    if dir_acc is not None:
        acc_score = float(max(0.0, min(1.0, (dir_acc - 0.5) * 2.0)))

    thr = float(max(0.0008, vol_proxy * 0.25))
    if ai_horizon:
        thr = float(min(0.01, thr * (ai_horizon**0.5) * 0.8))

    if float(ai_forecast_ret) > thr:
        ai_signal = "buy"
    elif float(ai_forecast_ret) < -thr:
        ai_signal = "sell"
    else:
        ai_signal = "neutral"

    mag_score = float(min(1.0, abs(float(ai_forecast_ret)) / max(1e-6, thr * 2.0)))
    base_conf = 0.45 if ai_model_name == "ridge_regression_v1" else 0.35
    confidence = float(max(0.0, min(1.0, base_conf + 0.35 * acc_score + 0.25 * mag_score - 0.25 * uncertainty)))

    predicted_close = float(last_close * (1.0 + float(ai_forecast_ret))) if last_close > 0 else float(last_close)
    ai_price = {
        "model": ai_model_name,
        "model_family": ai_model_family,
        "horizon_steps": ai_horizon,
        "forecast_return": float(ai_forecast_ret),
        "predicted_close": float(predicted_close),
        "signal": ai_signal,
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "metrics": ai_metrics,
    }

    # Prefer AI signal for the trade plan (keep heuristics as fallback only).
    if ai_signal in {"buy", "sell"} and last_close > 0 and atr > 0:
        entry = float(last_close)
        stop_dist = float(max(atr * 1.1, buffer * 2.0))
        tgt_min = float(max(atr * 1.8, buffer * 3.0))
        if ai_signal == "buy":
            stop = float(entry - stop_dist)
            target = float(max(entry + tgt_min, predicted_close))
            trade = {
                "side": "buy",
                "entry": entry,
                "stop": stop,
                "target": target,
                "confidence": float(confidence),
                "reason": "ai_forecast",
                "forecast_return": float(ai_forecast_ret),
                "predicted_close": float(predicted_close),
            }
        else:
            stop = float(entry + stop_dist)
            target = float(min(entry - tgt_min, predicted_close))
            trade = {
                "side": "sell",
                "entry": entry,
                "stop": stop,
                "target": target,
                "confidence": float(confidence),
                "reason": "ai_forecast",
                "forecast_return": float(ai_forecast_ret),
                "predicted_close": float(predicted_close),
            }

    patterns.append({"type": "ai_signal", "side": ai_signal, "confidence": float(confidence), "forecast_return": float(ai_forecast_ret)})

    # Candlestick pattern scan (multi-day sliding window via lookback upstream).
    candle_patterns: list[dict[str, Any]] = []
    try:
        _ensure_pattern_rows()
        weights = get_weights_cached(ttl_seconds=30)
        cps = detect_candlestick_patterns(to_pattern_candles(cs), weights=weights, max_results=8)
        candle_patterns = [
            {
                "name": m.name,
                "family": m.family,
                "side": m.side,
                "window": m.window,
                "start_ts": _i(_candle_get(cs[-int(m.window)], "ts")) if len(cs) >= int(m.window) else None,
                "end_ts": _i(_candle_get(cs[-1], "ts")) if cs else None,
                "confidence": float(m.confidence),
                "details": dict(m.details or {}),
            }
            for m in cps
        ]
        # Also show the top couple in the legacy `patterns` list for existing UI.
        for m in cps[:2]:
            patterns.append({"type": f"candle:{m.name}", "side": m.side, "confidence": float(m.confidence)})
    except Exception:
        candle_patterns = []

    # Volume spike signal is best-effort; some feeds have volume=0.
    vols = [_f(_candle_get(c, "volume")) for c in cs[-30:]]
    avg_vol = (sum(vols) / max(1, len(vols))) if vols else 0.0
    vol_ok = True
    try:
        vol_ok = avg_vol <= 0 or _f(_candle_get(last, "volume")) >= avg_vol * 1.2
    except Exception:
        vol_ok = True

    # Breakout / breakdown (heuristic fallback; do not override an AI trade)
    if nearest_res is not None and last_close > nearest_res + buffer and vol_ok:
        conf = 0.55 + 0.35 * float(trend.get("strength") or 0.0)
        if str(trend.get("dir")) == "up":
            conf += 0.1
        conf = float(max(0.0, min(1.0, conf)))
        patterns.append({"type": "breakout", "side": "buy", "level": float(nearest_res), "confidence": conf})

        if trade is None:
            stop = float(nearest_res - max(buffer, atr * 0.75))
            stop = float(min(stop, last_close - max(buffer, atr * 0.6)))
            target = float(last_close + max(atr * 2.0, abs(last_close - nearest_res) * 2.0))
            trade = {
                "side": "buy",
                "entry": float(last_close),
                "stop": float(stop),
                "target": float(target),
                "confidence": conf,
                "reason": "breakout_above_resistance",
            }

    if nearest_sup is not None and last_close < nearest_sup - buffer and vol_ok:
        conf = 0.55 + 0.35 * float(trend.get("strength") or 0.0)
        if str(trend.get("dir")) == "down":
            conf += 0.1
        conf = float(max(0.0, min(1.0, conf)))
        patterns.append({"type": "breakdown", "side": "sell", "level": float(nearest_sup), "confidence": conf})

        if trade is None:
            stop = float(nearest_sup + max(buffer, atr * 0.75))
            stop = float(max(stop, last_close + max(buffer, atr * 0.6)))
            target = float(last_close - max(atr * 2.0, abs(last_close - nearest_sup) * 2.0))
            trade = {
                "side": "sell",
                "entry": float(last_close),
                "stop": float(stop),
                "target": float(target),
                "confidence": conf,
                "reason": "breakdown_below_support",
            }

    # If no discrete pattern, still provide a "plan" based on trend + ATR and nearest levels.
    if trade is None:
        dirn = str(trend.get("dir") or "flat")
        strength = float(trend.get("strength") or 0.0)
        if dirn in ("up", "down") and strength >= 0.35 and atr > 0:
            if dirn == "up":
                sup = nearest_sup
                stop = float((sup - max(buffer, atr * 0.6)) if sup is not None else (last_close - atr * 1.5))
                res = nearest_res
                target = float((res + max(buffer, atr * 0.6)) if res is not None else (last_close + atr * 2.0))
                trade = {
                    "side": "buy",
                    "entry": float(last_close),
                    "stop": float(stop),
                    "target": float(target),
                    "confidence": float(min(0.65, 0.35 + strength * 0.5)),
                    "reason": "trend_following",
                }
            else:
                res = nearest_res
                stop = float((res + max(buffer, atr * 0.6)) if res is not None else (last_close + atr * 1.5))
                sup = nearest_sup
                target = float((sup - max(buffer, atr * 0.6)) if sup is not None else (last_close - atr * 2.0))
                trade = {
                    "side": "sell",
                    "entry": float(last_close),
                    "stop": float(stop),
                    "target": float(target),
                    "confidence": float(min(0.65, 0.35 + strength * 0.5)),
                    "reason": "trend_following",
                }

    # Basic sanity clamp: stop/target should be on correct side.
    if trade is not None:
        if trade["side"] == "buy":
            trade["stop"] = float(min(trade["stop"], trade["entry"] - buffer))
            trade["target"] = float(max(trade["target"], trade["entry"] + buffer))
        else:
            trade["stop"] = float(max(trade["stop"], trade["entry"] + buffer))
            trade["target"] = float(min(trade["target"], trade["entry"] - buffer))

        # Keep target/stop finite
        for k in ("entry", "stop", "target"):
            if not math.isfinite(float(trade[k])):
                trade = None
                break

    # Blend top candlestick pattern into trade confidence when it agrees/conflicts.
    if trade is not None and candle_patterns:
        best = candle_patterns[0]
        side = str(trade.get("side") or "")
        p_side = str(best.get("side") or "")
        p_conf = float(best.get("confidence") or 0.0)
        if p_side in {"buy", "sell"} and side in {"buy", "sell"}:
            delta = 0.12 * p_conf if p_side == side else (-0.10 * p_conf)
            trade["confidence"] = float(max(0.0, min(1.0, float(trade.get("confidence") or 0.0) + delta)))
            trade["pattern_hint"] = {"name": best.get("name"), "side": p_side, "confidence": p_conf}

    # Winner strategy: a single best-fit "setup" summary for UI/learning.
    winner_strategy: dict[str, Any] | None = None
    if trade is not None:
        rid = str(trade.get("reason") or "trade_plan")
        winner_strategy = {
            "id": rid,
            "label": rid.replace("_", " "),
            "side": str(trade.get("side") or "").lower(),
            "confidence": float(trade.get("confidence") or 0.0),
            "source": "trade",
        }
        if trade.get("pattern_hint") is not None:
            winner_strategy["pattern_hint"] = dict(trade.get("pattern_hint") or {})
    elif patterns:
        bestp = max(patterns, key=lambda p: float(p.get("confidence") or 0.0))
        pid = str(bestp.get("type") or "pattern")
        winner_strategy = {
            "id": pid,
            "label": pid.replace("_", " "),
            "side": str(bestp.get("side") or "neutral"),
            "confidence": float(bestp.get("confidence") or 0.0),
            "source": "pattern",
        }
    elif candle_patterns:
        bestc = candle_patterns[0]
        pid = f"candle:{bestc.get('name')}"
        winner_strategy = {
            "id": pid,
            "label": str(bestc.get("name") or "candlestick").strip(),
            "side": str(bestc.get("side") or "neutral"),
            "confidence": float(bestc.get("confidence") or 0.0),
            "source": "candle_pattern",
        }

    base = {
        "instrument_key": instrument_key,
        "interval": interval,
        "asof_ts": int(last_ts),
        "n": len(cs),
        "atr": float(atr),
        "buffer": float(buffer),
        "trend": trend,
        "levels": {"support": [float(x) for x in levels.support], "resistance": [float(x) for x in levels.resistance]},
        "patterns": patterns,
        "candle_patterns": candle_patterns,
        "trade": trade,
        "winner_strategy": winner_strategy,
    }

    # ML-lite calibrator: predicts probability of favorable up-move.
    model = load_model(instrument_key, interval)
    if model is None:
        base["ai"] = {"enabled": False, "model": None, "price": ai_price}
        return base

    feats = featurize(cs[-max(80, 60) :])
    p_up = float(model.predict_proba_up(feats))

    # Merge model confidence into trade (if present)
    if base.get("trade") is not None:
        side = str(base["trade"].get("side") or "").lower()
        model_conf = p_up if side == "buy" else (1.0 - p_up)
        model_conf = float(max(0.0, min(1.0, model_conf)))

        base_conf = float(base["trade"].get("confidence") or 0.0)
        combined = float(max(0.0, min(1.0, 0.65 * base_conf + 0.35 * model_conf)))
        base["trade"]["model_confidence"] = model_conf
        base["trade"]["confidence"] = combined

        # Conservative risk adjustment: scale target/stop slightly based on confidence.
        entry = float(base["trade"].get("entry") or 0.0)
        stop = float(base["trade"].get("stop") or entry)
        target = float(base["trade"].get("target") or entry)
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0 and reward > 0:
            if combined >= 0.7:
                profile = "aggressive"
                target = entry + (target - entry) * 1.15
                stop = entry + (stop - entry) * 0.95
            elif combined <= 0.45:
                profile = "conservative"
                target = entry + (target - entry) * 0.85
                stop = entry + (stop - entry) * 1.05
            else:
                profile = "normal"
            base["trade"]["stop"] = float(stop)
            base["trade"]["target"] = float(target)
            base["trade"]["risk_profile"] = profile

    base["ai"] = {
        "enabled": True,
        "model": {
            "instrument_key": model.instrument_key,
            "interval": model.interval,
            "created_ts": int(model.created_ts),
            "metrics": dict(model.metrics or {}),
            "horizon_steps": int(model.horizon_steps),
            "k_atr": float(model.k_atr),
        },
        "p_up": p_up,
        "features": feats,
        "price": ai_price,
    }
    return base
