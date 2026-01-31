from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_returns(closes: np.ndarray) -> np.ndarray:
    if len(closes) < 2:
        return np.array([], dtype=float)
    prev = closes[:-1]
    nxt = closes[1:]
    prev = np.where(prev == 0, 1e-9, prev)
    return (nxt - prev) / prev


@dataclass(frozen=True)
class Features:
    rsi14: float
    sma20: float
    ema20: float
    volatility20: float
    last_close: float


def compute_features(closes: list[float]) -> Features:
    arr = np.asarray(closes, dtype=float)
    if arr.size == 0:
        return Features(rsi14=50.0, sma20=0.0, ema20=0.0, volatility20=0.0, last_close=0.0)

    last_close = float(arr[-1])

    # SMA/EMA
    window = min(20, arr.size)
    sma20 = float(np.mean(arr[-window:]))

    # EMA (simple iterative)
    alpha = 2 / (window + 1)
    ema = float(arr[0])
    for v in arr[1:]:
        ema = alpha * float(v) + (1 - alpha) * ema
    ema20 = float(ema)

    # RSI14
    rsi_window = min(14, arr.size - 1)
    if rsi_window <= 0:
        rsi14 = 50.0
    else:
        diffs = np.diff(arr)[-rsi_window:]
        gains = np.where(diffs > 0, diffs, 0.0)
        losses = np.where(diffs < 0, -diffs, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        rs = avg_gain / (avg_loss + 1e-9)
        rsi14 = 100.0 - (100.0 / (1.0 + rs))

    # Volatility (std of returns)
    rets = _safe_returns(arr)
    vol_window = min(20, rets.size)
    volatility20 = float(np.std(rets[-vol_window:])) if vol_window > 0 else 0.0

    return Features(
        rsi14=rsi14,
        sma20=sma20,
        ema20=ema20,
        volatility20=volatility20,
        last_close=last_close,
    )
