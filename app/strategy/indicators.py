from __future__ import annotations

import numpy as np


def _ema_series(values: list[float] | np.ndarray, period: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    p = max(1, int(period))
    alpha = 2.0 / (float(p) + 1.0)
    out = np.empty(arr.size, dtype=float)
    out[0] = float(arr[0])
    for i in range(1, int(arr.size)):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * float(out[i - 1])
    return out


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


def macd_last(closes: list[float], *, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float]:
    """Return the last MACD line/signal/hist.

    This is intentionally lightweight (no pandas) and is suitable for intraday/HFT loops.
    """
    if not closes:
        return {"macd": 0.0, "signal": 0.0, "hist": 0.0}
    c = np.asarray(closes, dtype=float)
    if c.size == 0:
        return {"macd": 0.0, "signal": 0.0, "hist": 0.0}

    fast_ema = _ema_series(c, int(fast))
    slow_ema = _ema_series(c, int(slow))
    macd_line = fast_ema - slow_ema
    sig_line = _ema_series(macd_line, int(signal))
    hist = macd_line - sig_line
    return {"macd": float(macd_line[-1]), "signal": float(sig_line[-1]), "hist": float(hist[-1])}


def adx_last(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    period: int = 14,
) -> dict[str, float]:
    """Return last ADX and directionals (+DI/-DI).

    Implementation uses Wilder-style smoothing and returns a best-effort value even
    when history is shorter than the canonical 2*period.
    """
    n = min(len(highs), len(lows), len(closes))
    if n < 3:
        return {"adx": 0.0, "+di": 0.0, "-di": 0.0}

    p = max(2, int(period))
    h = np.asarray(highs[-n:], dtype=float)
    l = np.asarray(lows[-n:], dtype=float)
    c = np.asarray(closes[-n:], dtype=float)

    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_c = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))

    m = int(tr.size)
    if m < 1:
        return {"adx": 0.0, "+di": 0.0, "-di": 0.0}

    w = min(p, m)
    tr_smooth = float(np.sum(tr[:w]))
    plus_smooth = float(np.sum(plus_dm[:w]))
    minus_smooth = float(np.sum(minus_dm[:w]))

    dx_list: list[float] = []
    last_pdi = 0.0
    last_mdi = 0.0

    def _dx(tr_sm: float, p_sm: float, m_sm: float) -> tuple[float, float, float]:
        if tr_sm <= 1e-12:
            return 0.0, 0.0, 0.0
        pdi = 100.0 * (p_sm / tr_sm)
        mdi = 100.0 * (m_sm / tr_sm)
        denom = pdi + mdi
        dx = 0.0 if denom <= 1e-12 else 100.0 * abs(pdi - mdi) / denom
        return float(pdi), float(mdi), float(dx)

    last_pdi, last_mdi, dx0 = _dx(tr_smooth, plus_smooth, minus_smooth)
    dx_list.append(dx0)

    # Wilder smoothing forward
    for i in range(w, m):
        tr_smooth = tr_smooth - (tr_smooth / float(w)) + float(tr[i])
        plus_smooth = plus_smooth - (plus_smooth / float(w)) + float(plus_dm[i])
        minus_smooth = minus_smooth - (minus_smooth / float(w)) + float(minus_dm[i])
        last_pdi, last_mdi, dxi = _dx(tr_smooth, plus_smooth, minus_smooth)
        dx_list.append(dxi)

    if not dx_list:
        return {"adx": 0.0, "+di": float(last_pdi), "-di": float(last_mdi)}

    # ADX: Wilder smoothing of DX; if insufficient history, fall back to mean DX.
    if len(dx_list) >= w:
        adx = float(np.mean(dx_list[:w]))
        for j in range(w, len(dx_list)):
            adx = (adx * (float(w) - 1.0) + float(dx_list[j])) / float(w)
    else:
        adx = float(np.mean(dx_list))

    return {"adx": float(np.clip(adx, 0.0, 100.0)), "+di": float(last_pdi), "-di": float(last_mdi)}
