from __future__ import annotations

import numpy as np


def sma(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    period = max(1, min(period, len(values)))
    arr = np.asarray(values[-period:], dtype=float)
    return float(np.mean(arr))


def ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    period = max(1, min(period, len(values)))
    alpha = 2 / (period + 1)
    e = float(values[0])
    for v in values[1:]:
        e = alpha * float(v) + (1 - alpha) * e
    return float(e)


def rsi(values: list[float], period: int = 14) -> float:
    if len(values) < 2:
        return 50.0
    period = max(1, min(period, len(values) - 1))
    arr = np.asarray(values, dtype=float)
    diffs = np.diff(arr)[-period:]
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    rs = avg_gain / (avg_loss + 1e-9)
    return float(100.0 - 100.0 / (1.0 + rs))


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return 0.0
    period = max(1, min(period, n - 1))

    h = np.asarray(highs[-n:], dtype=float)
    l = np.asarray(lows[-n:], dtype=float)
    c = np.asarray(closes[-n:], dtype=float)

    prev_c = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
    return float(np.mean(tr[-period:]))


def vwap(prices: list[float], volumes: list[float]) -> float:
    n = min(len(prices), len(volumes))
    if n == 0:
        return 0.0
    p = np.asarray(prices[-n:], dtype=float)
    v = np.asarray(volumes[-n:], dtype=float)
    denom = float(np.sum(v))
    if denom <= 0:
        return float(np.mean(p))
    return float(np.sum(p * v) / denom)


def volatility(returns: list[float], period: int = 20) -> float:
    if not returns:
        return 0.0
    period = max(1, min(period, len(returns)))
    arr = np.asarray(returns[-period:], dtype=float)
    return float(np.std(arr))
