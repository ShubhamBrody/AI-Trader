from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.candles.persistence_sql import get_candles


def return_stats(instrument_key: str, interval: str, *, start_ts: int, end_ts: int) -> dict[str, Any]:
    candles = get_candles(str(instrument_key), str(interval), int(start_ts), int(end_ts))
    closes = np.asarray([float(c.close) for c in candles], dtype=float)
    if closes.size < 3:
        return {"ok": False, "reason": "not enough data", "n": int(closes.size)}

    closes = np.where(closes <= 0, 1e-9, closes)
    rets = np.diff(closes) / closes[:-1]
    mu = float(np.mean(rets)) if rets.size else 0.0
    sd = float(np.std(rets)) if rets.size else 0.0
    return {"ok": True, "n": int(rets.size), "ret_mean": float(mu), "ret_std": float(sd)}


def drift_z(
    *,
    baseline_mean: float,
    baseline_std: float,
    recent_mean: float,
    recent_n: int,
) -> float:
    # Simple mean-shift z-score using baseline std.
    if recent_n <= 3:
        return 0.0
    sd = float(baseline_std)
    if sd <= 1e-12:
        return 0.0
    se = sd / float(np.sqrt(float(recent_n)))
    if se <= 1e-12:
        return 0.0
    return float(abs(float(recent_mean) - float(baseline_mean)) / se)


def drift_check_recent(
    instrument_key: str,
    interval: str,
    *,
    baseline: dict[str, Any],
    recent_minutes: int = 240,
    recent_days: int = 10,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if str(interval).endswith("m"):
        start = now - timedelta(minutes=int(recent_minutes))
    else:
        start = now - timedelta(days=int(recent_days))

    rs = return_stats(str(instrument_key), str(interval), start_ts=int(start.timestamp()), end_ts=int(now.timestamp()))
    if not rs.get("ok"):
        return {"ok": False, "reason": rs.get("reason"), "recent": rs}

    z = drift_z(
        baseline_mean=float((baseline or {}).get("ret_mean") or 0.0),
        baseline_std=float((baseline or {}).get("ret_std") or 0.0),
        recent_mean=float(rs.get("ret_mean") or 0.0),
        recent_n=int(rs.get("n") or 0),
    )

    return {"ok": True, "z": float(z), "recent": rs, "baseline": {"ret_mean": float((baseline or {}).get("ret_mean") or 0.0), "ret_std": float((baseline or {}).get("ret_std") or 0.0)}}
