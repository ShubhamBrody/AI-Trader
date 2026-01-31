from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Levels:
    support: list[float]
    resistance: list[float]


def detect_support_resistance(highs: list[float], lows: list[float], lookback: int = 50, swing: int = 3) -> Levels:
    n = min(len(highs), len(lows))
    if n == 0:
        return Levels(support=[], resistance=[])

    start = max(0, n - lookback)
    hs = highs[start:n]
    ls = lows[start:n]

    support: list[float] = []
    resistance: list[float] = []

    # Very simple swing detection
    for i in range(swing, len(hs) - swing):
        window_h = hs[i - swing : i + swing + 1]
        window_l = ls[i - swing : i + swing + 1]
        if hs[i] == max(window_h):
            resistance.append(float(hs[i]))
        if ls[i] == min(window_l):
            support.append(float(ls[i]))

    # Deduplicate (bucket within tolerance)
    def _dedupe(vals: list[float], tol: float = 0.002) -> list[float]:
        vals = sorted(vals)
        out: list[float] = []
        for v in vals:
            if not out:
                out.append(v)
                continue
            if abs(v - out[-1]) / max(out[-1], 1e-9) > tol:
                out.append(v)
        return out

    return Levels(support=_dedupe(support), resistance=_dedupe(resistance))
