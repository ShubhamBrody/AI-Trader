from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.prediction.calibration import get_calibration
from app.prediction.live import LivePredictorConfig, PREDICTOR
from app.prediction.persistence import list_predictions

router = APIRouter(prefix="/prediction", tags=["prediction"])


class PredictorStartRequest(BaseModel):
    instrument_keys: list[str] = Field(default_factory=list)
    interval: str = "1m"
    horizon_steps: int = 60
    lookback_minutes: int = 240
    poll_seconds: int = 10
    calibration_alpha: float = 0.05


@router.get("/status")
def status() -> dict[str, Any]:
    return {"ok": True, **PREDICTOR.status()}


@router.post("/start")
async def start(req: PredictorStartRequest) -> dict[str, Any]:
    if not req.instrument_keys:
        raise HTTPException(status_code=400, detail="instrument_keys is required")

    cfg = LivePredictorConfig(
        interval=str(req.interval),
        horizon_steps=int(req.horizon_steps),
        lookback_minutes=int(req.lookback_minutes),
        poll_seconds=int(req.poll_seconds),
        calibration_alpha=float(req.calibration_alpha),
    )
    return await PREDICTOR.start(instrument_keys=[str(x) for x in req.instrument_keys], cfg=cfg)


@router.post("/stop")
async def stop() -> dict[str, Any]:
    return await PREDICTOR.stop()


@router.get("/events")
def events(limit: int = 200, instrument_key: str | None = None, status: str | None = None) -> dict[str, Any]:
    return {"ok": True, "items": list_predictions(limit=int(limit), instrument_key=instrument_key, status=status)}


@router.get("/calibration")
def calibration(key: str | None = None, instrument_key: str | None = None, interval: str = "1m", horizon_steps: int = 60) -> dict[str, Any]:
    if not key:
        if not instrument_key:
            raise HTTPException(status_code=400, detail="Provide key or instrument_key")
        key = f"{instrument_key}::{interval}::h{int(horizon_steps)}"

    cal = get_calibration(str(key))
    return {"ok": True, "key": str(key), "calibration": cal}
