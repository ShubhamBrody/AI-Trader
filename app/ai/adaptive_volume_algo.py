from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class AlgoParams:
    period: int = 40
    multiplier: float = 2.0
    z_threshold: float = 0.8
    atr_period: int = 14
    rr: float = 2.0
    horizon_steps: int = 12
    use_last_closed: bool = True


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _rolling_zscore(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    if n == 0:
        return []
    period = max(2, min(int(period), n))

    out: list[float | None] = [None] * n
    s = 0.0
    s2 = 0.0

    for i, v in enumerate(values):
        v = float(v)
        s += v
        s2 += v * v

        if i >= period:
            old = float(values[i - period])
            s -= old
            s2 -= old * old

        if i + 1 >= period:
            mean = s / period
            var = max(0.0, (s2 / period) - (mean * mean))
            std = sqrt(var) if var > 1e-12 else 0.0
            out[i] = 0.0 if std <= 1e-12 else (v - mean) / std

    return out


def _ema_series(values: list[float], period: int) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    period = max(1, min(int(period), n))
    alpha = 2.0 / (period + 1.0)

    e = float(values[0])
    out = [e]
    for v in values[1:]:
        v = float(v)
        e = alpha * v + (1.0 - alpha) * e
        out.append(e)
    return out


def _atr_series(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float | None]:
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return [None] * n
    period = max(1, min(int(period), n - 1))

    out: list[float | None] = [None] * n

    # True range for bar i uses previous close.
    trs: list[float] = [0.0] * n
    for i in range(1, n):
        h = float(highs[i])
        l = float(lows[i])
        pc = float(closes[i - 1])
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs[i] = tr

    # Rolling mean of TR.
    s = 0.0
    for i in range(n):
        s += trs[i]
        if i >= period:
            s -= trs[i - period]
        if i >= period:
            out[i] = s / period

    return out


def _extract_ohlcv(candles: list[Any]) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    for c in candles:
        # candle objects in this codebase often have attributes (open/high/low/close/volume)
        # but API payloads may be dicts too.
        if isinstance(c, dict):
            o = float(c.get("open") or 0.0)
            h = float(c.get("high") or 0.0)
            l = float(c.get("low") or 0.0)
            cl = float(c.get("close") or 0.0)
            v = float(c.get("volume") or 0.0)
        else:
            o = float(getattr(c, "open"))
            h = float(getattr(c, "high"))
            l = float(getattr(c, "low"))
            cl = float(getattr(c, "close"))
            v = float(getattr(c, "volume", 0.0))

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(cl)
        volumes.append(v)

    return opens, highs, lows, closes, volumes


def compute_signal(candles: list[Any], params: AlgoParams) -> dict:
    """Compute a single signal from candles.

    This is a Swift‑Algo‑like concept: volume z-score ("force") + fair-value EMA,
    with adaptive ATR bands. Signals are computed on the last *closed* bar by default.

    Returns a dict with signal/levels for the frontend.
    """

    opens, highs, lows, closes, volumes = _extract_ohlcv(candles)
    n = len(closes)
    if n < max(30, params.period + 5):
        return {
            "ok": False,
            "reason": "not_enough_candles",
            "n": n,
        }

    # Non-repainting: prefer the last closed candle.
    idx = n - 2 if params.use_last_closed and n >= 2 else n - 1

    z = _rolling_zscore(volumes, params.period)
    fv = _ema_series(closes, params.period)
    atrs = _atr_series(highs, lows, closes, params.atr_period)

    z_i = z[idx]
    fv_i = fv[idx]
    atr_i = atrs[idx]
    close_i = float(closes[idx])

    if z_i is None or atr_i is None or atr_i <= 0:
        return {
            "ok": False,
            "reason": "insufficient_indicators",
        }

    upper = fv_i + params.multiplier * float(atr_i)
    lower = fv_i - params.multiplier * float(atr_i)

    side = "HOLD"
    if close_i > upper and float(z_i) >= params.z_threshold:
        side = "BUY"
    elif close_i < lower and float(z_i) <= -params.z_threshold:
        side = "SELL"

    # Risk distance scales with ATR, multiplier and force strength.
    z_mag = abs(float(z_i))
    z_boost = 1.0 + 0.25 * _clamp(z_mag, 0.0, 3.0)
    risk_dist = max(1e-9, float(atr_i) * max(0.5, float(params.multiplier) * 0.8) * z_boost)

    entry = close_i
    if side == "BUY":
        stop_loss = entry - risk_dist
        target = entry + params.rr * risk_dist
    elif side == "SELL":
        stop_loss = entry + risk_dist
        target = entry - params.rr * risk_dist
    else:
        stop_loss = None
        target = None

    strength = _clamp((z_mag - params.z_threshold) / max(1e-9, 3.0 - params.z_threshold), 0.0, 1.0)

    return {
        "ok": True,
        "asof_index": idx,
        "side": side,
        "strength": float(strength),
        "entry": float(entry),
        "stop_loss": float(stop_loss) if stop_loss is not None else None,
        "target": float(target) if target is not None else None,
        "fair_value": float(fv_i),
        "upper_band": float(upper),
        "lower_band": float(lower),
        "volume_z": float(z_i),
        "atr": float(atr_i),
        "reason": "volume_force_band_break" if side in ("BUY", "SELL") else "no_breakout",
    }


def backtest_params(candles: list[Any], params: AlgoParams) -> dict:
    opens, highs, lows, closes, volumes = _extract_ohlcv(candles)
    n = len(closes)
    if n < max(60, params.period + params.horizon_steps + 5):
        return {
            "ok": False,
            "reason": "not_enough_candles",
            "n": n,
        }

    z = _rolling_zscore(volumes, params.period)
    fv = _ema_series(closes, params.period)
    atrs = _atr_series(highs, lows, closes, params.atr_period)

    # Evaluate signals on closed bars only.
    last_eval = n - 1 - int(params.horizon_steps)
    if params.use_last_closed:
        last_eval -= 1

    trades = 0
    wins = 0

    for i in range(max(params.period, params.atr_period) + 5, max(0, last_eval)):
        z_i = z[i]
        atr_i = atrs[i]
        if z_i is None or atr_i is None or atr_i <= 0:
            continue

        close_i = float(closes[i])
        upper = fv[i] + params.multiplier * float(atr_i)
        lower = fv[i] - params.multiplier * float(atr_i)

        side: str | None = None
        if close_i > upper and float(z_i) >= params.z_threshold:
            side = "BUY"
        elif close_i < lower and float(z_i) <= -params.z_threshold:
            side = "SELL"

        if not side:
            continue

        j = i + int(params.horizon_steps)
        if j >= n:
            break

        ret = float(closes[j]) - close_i
        trades += 1
        if (side == "BUY" and ret > 0) or (side == "SELL" and ret < 0):
            wins += 1

    win_rate = (wins / trades) if trades > 0 else 0.0

    # Penalize extremely low sample sizes.
    sample_penalty = _clamp(trades / 20.0, 0.0, 1.0)
    score = win_rate * sample_penalty

    return {
        "ok": True,
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": float(win_rate),
        "score": float(score),
    }


def optimize(candles: list[Any], base: AlgoParams, periods: list[int] | None = None, multipliers: list[float] | None = None) -> dict:
    # Defaults chosen to yield 24 combos (6 x 4), similar to the feature request.
    periods = periods or [20, 30, 40, 50, 60, 80]
    multipliers = multipliers or [1.0, 1.5, 2.0, 2.5]

    leaderboard: list[dict] = []

    for p in periods:
        for m in multipliers:
            params = AlgoParams(
                period=int(p),
                multiplier=float(m),
                z_threshold=float(base.z_threshold),
                atr_period=int(base.atr_period),
                rr=float(base.rr),
                horizon_steps=int(base.horizon_steps),
                use_last_closed=bool(base.use_last_closed),
            )
            bt = backtest_params(candles, params)
            if not bt.get("ok"):
                continue
            leaderboard.append({"period": params.period, "multiplier": params.multiplier, **bt})

    leaderboard.sort(key=lambda r: (float(r.get("score") or 0.0), float(r.get("win_rate") or 0.0), int(r.get("trades") or 0)), reverse=True)

    best = leaderboard[0] if leaderboard else None
    return {
        "ok": True,
        "best": best,
        "leaderboard": leaderboard,
        "combos": int(len(leaderboard)),
    }


def trend_table(candles_by_interval: dict[str, list[Any]], fast: int = 20, slow: int = 50, use_last_closed: bool = True) -> list[dict]:
    out: list[dict] = []

    for interval, candles in candles_by_interval.items():
        _, _, _, closes, _ = _extract_ohlcv(candles)
        if len(closes) < slow + 5:
            out.append({"interval": interval, "trend": "UNKNOWN"})
            continue

        idx = len(closes) - 2 if use_last_closed and len(closes) >= 2 else len(closes) - 1
        ema_f = _ema_series(closes, fast)[idx]
        ema_s = _ema_series(closes, slow)[idx]
        if ema_f > ema_s:
            t = "UP"
        elif ema_f < ema_s:
            t = "DOWN"
        else:
            t = "FLAT"
        out.append({"interval": interval, "trend": t, "ema_fast": float(ema_f), "ema_slow": float(ema_s)})

    return out
