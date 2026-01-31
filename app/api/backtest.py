from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.candles.service import CandleService
from app.prediction.calibration import update_ema

router = APIRouter(prefix="/backtest", tags=["backtest"])


def _parse_iso(dt: str) -> datetime:
    try:
        d = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid datetime: {dt}") from e
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


@router.post("/run")
def run_backtest(instrument_key: str, interval: str = "15m", start: str | None = None, end: str | None = None) -> dict:
    """Very lightweight backtest endpoint for the frontend.

    This is intentionally simple (good for UI wiring): it computes basic return stats
    for the requested window and writes an online calibration entry.
    """

    ik = str(instrument_key or "").strip()
    if not ik:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    now = datetime.now(timezone.utc)
    end_dt = _parse_iso(end) if end else now
    start_dt = _parse_iso(start) if start else (end_dt.replace(hour=0, minute=0, second=0, microsecond=0))

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    svc = CandleService()

    # Ensure there is data; if DB is empty, load and persist.
    series = svc.get_historical(ik, interval, start=start_dt, end=end_dt)
    if not series.candles:
        series = svc.load_historical(ik, interval, start=start_dt, end=end_dt)

    candles = list(series.candles or [])
    if len(candles) < 2:
        raise HTTPException(status_code=404, detail="not enough candles")

    first = float(candles[0].close)
    last = float(candles[-1].close)
    ret = (last - first) / first if first else 0.0

    # Persist a simple calibration record using EMA update.
    # We treat the model as predicting 0 return; bias becomes the realized mean return.
    key = f"backtest::{ik}::{interval}"
    update_ema(key, residual=float(ret), abs_error=abs(float(ret)), alpha=0.1)

    return {
        "ok": True,
        "instrument_key": ik,
        "interval": str(interval),
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "n": len(candles),
        "first_close": first,
        "last_close": last,
        "return": float(ret),
        "calibration_key": key,
    }
