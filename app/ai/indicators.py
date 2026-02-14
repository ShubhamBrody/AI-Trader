from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ema_series(arr: np.ndarray, period: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    p = max(1, int(period))
    alpha = 2.0 / (float(p) + 1.0)
    out = np.empty(arr.size, dtype=float)
    out[0] = float(arr[0])
    for i in range(1, arr.size):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * float(out[i - 1])
    return out


def _sma_last(arr: np.ndarray, period: int) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    p = max(1, int(period))
    w = min(p, int(arr.size))
    return float(np.mean(arr[-w:]))


def _rsi_last(closes: np.ndarray, period: int = 14) -> float:
    closes = np.asarray(closes, dtype=float)
    if closes.size < 2:
        return 50.0
    p = max(1, int(period))
    diffs = np.diff(closes)
    w = min(p, int(diffs.size))
    if w <= 0:
        return 50.0
    tail = diffs[-w:]
    gains = np.where(tail > 0, tail, 0.0)
    losses = np.where(tail < 0, -tail, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0.0, 100.0))


def _bollinger_last(closes: np.ndarray, period: int = 20, k: float = 2.0) -> dict[str, float]:
    closes = np.asarray(closes, dtype=float)
    if closes.size == 0:
        return {"mid": 0.0, "upper": 0.0, "lower": 0.0, "std": 0.0, "bandwidth": 0.0}
    p = max(1, int(period))
    w = min(p, int(closes.size))
    window = closes[-w:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    upper = float(mid + float(k) * std)
    lower = float(mid - float(k) * std)
    bw = float((upper - lower) / mid) if mid != 0 else 0.0
    return {"mid": mid, "upper": upper, "lower": lower, "std": std, "bandwidth": bw}


def _macd_last(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float]:
    closes = np.asarray(closes, dtype=float)
    if closes.size == 0:
        return {"macd": 0.0, "signal": 0.0, "hist": 0.0}
    fast_ema = _ema_series(closes, int(fast))
    slow_ema = _ema_series(closes, int(slow))
    macd_line = fast_ema - slow_ema
    sig_line = _ema_series(macd_line, int(signal))
    hist = macd_line - sig_line
    return {"macd": float(macd_line[-1]), "signal": float(sig_line[-1]), "hist": float(hist[-1])}


def _stoch_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14, d_period: int = 3) -> dict[str, float]:
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = int(closes.size)
    if n == 0:
        return {"k": 0.0, "d": 0.0}

    kp = max(1, int(k_period))
    dp = max(1, int(d_period))

    k_vals = []
    # compute last dp values of %K so we can get %D SMA(dp)
    for offset in range(dp):
        end = n - offset
        if end <= 0:
            break
        start = max(0, end - kp)
        hh = float(np.max(highs[start:end]))
        ll = float(np.min(lows[start:end]))
        c = float(closes[end - 1])
        denom = (hh - ll)
        if denom <= 1e-12:
            k = 50.0
        else:
            k = 100.0 * (c - ll) / denom
        k_vals.append(float(np.clip(k, 0.0, 100.0)))

    if not k_vals:
        return {"k": 50.0, "d": 50.0}

    k_last = float(k_vals[0])
    d_last = float(np.mean(k_vals[: min(dp, len(k_vals))]))
    return {"k": k_last, "d": d_last}


@dataclass(frozen=True)
class IndicatorSnapshot:
    ema20: float
    ema50: float
    ema200: float
    sma20: float
    rsi14: float
    bollinger: dict[str, float]
    macd: dict[str, float]
    stochastic: dict[str, float]


def compute_indicator_snapshot(
    *,
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> IndicatorSnapshot:
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs if highs is not None else closes, dtype=float)
    l = np.asarray(lows if lows is not None else closes, dtype=float)

    ema20 = float(_ema_series(c, 20)[-1]) if c.size else 0.0
    ema50 = float(_ema_series(c, 50)[-1]) if c.size else 0.0
    ema200 = float(_ema_series(c, 200)[-1]) if c.size else 0.0
    sma20 = float(_sma_last(c, 20))
    rsi14 = float(_rsi_last(c, 14))
    bb = _bollinger_last(c, 20, 2.0)
    macd = _macd_last(c, 12, 26, 9)
    stoch = _stoch_last(h, l, c, 14, 3)

    return IndicatorSnapshot(
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        sma20=sma20,
        rsi14=rsi14,
        bollinger=bb,
        macd=macd,
        stochastic=stoch,
    )
