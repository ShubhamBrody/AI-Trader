from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.db import db_conn
from app.core.settings import settings
from app.prediction.calibration import get_calibration
from app.prediction.live import LivePredictorConfig, PREDICTOR
from app.prediction.persistence import list_predictions
from app.prediction.analytics import drift as analytics_drift
from app.prediction.analytics import list_calibrations as analytics_list_calibrations
from app.prediction.analytics import reliability_bins as analytics_reliability_bins
from app.prediction.analytics import rollups as analytics_rollups
from app.prediction.retention import cleanup_prediction_events

router = APIRouter(prefix="/prediction", tags=["prediction"])


class ConfusionMatrix(BaseModel):
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


class PredictionRollupRow(BaseModel):
    instrument_key: str
    interval: str
    horizon_steps: int
    n: int
    accuracy: float
    mae: float
    rmse: float
    mean_error: float
    avg_pred: float
    avg_actual: float
    confusion: ConfusionMatrix


class PredictionRollupsResponse(BaseModel):
    ok: bool = True
    since_days: int
    items: list[PredictionRollupRow] = Field(default_factory=list)


class ReliabilityBinRow(BaseModel):
    lo: float
    hi: float
    n: int
    accuracy: float
    mae: float
    mean_error: float


class PredictionReliabilityResponse(BaseModel):
    ok: bool = True
    since_days: int
    bins: list[float] = Field(default_factory=list)
    items: list[ReliabilityBinRow] = Field(default_factory=list)


class PredictionDriftDelta(BaseModel):
    accuracy: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mean_error: float | None = None


class PredictionDriftRow(BaseModel):
    key: str
    baseline: PredictionRollupRow | None = None
    recent: PredictionRollupRow | None = None
    delta: PredictionDriftDelta


class PredictionDriftResponse(BaseModel):
    ok: bool = True
    baseline_days: int
    recent_days: int
    items: list[PredictionDriftRow] = Field(default_factory=list)


class CalibrationRow(BaseModel):
    key: str
    updated_ts: int
    n: int
    bias: float
    mae: float


class CalibrationsResponse(BaseModel):
    ok: bool = True
    items: list[CalibrationRow] = Field(default_factory=list)


class PredictionRetentionCleanupResponse(BaseModel):
    ok: bool = True
    enabled: bool
    max_age_days: int
    max_rows: int
    deleted_by_age: int
    deleted_by_rows: int
    remaining_rows: int


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


@router.get("/analytics/rollups", response_model=PredictionRollupsResponse)
def analytics_rollups_endpoint(
    since_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(200, ge=1, le=2000),
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
) -> dict[str, Any]:
    items = analytics_rollups(
        since_days=int(since_days),
        limit=int(limit),
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=horizon_steps,
    )
    return {"ok": True, "since_days": int(since_days), "items": items}


@router.get("/analytics/reliability", response_model=PredictionReliabilityResponse)
def analytics_reliability(
    since_days: int = Query(30, ge=1, le=3650),
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
    bins: str | None = None,
) -> dict[str, Any]:
    parsed_bins: list[float] | None = None
    if bins:
        try:
            parsed_bins = [float(x.strip()) for x in str(bins).split(",") if x.strip()]
        except Exception:
            parsed_bins = None

    # Return the bins used so the frontend can render x-axis labels.
    used_bins = parsed_bins or [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 1.0]
    items = analytics_reliability_bins(
        since_days=int(since_days),
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=horizon_steps,
        bins=used_bins,
    )
    return {"ok": True, "since_days": int(since_days), "bins": used_bins, "items": items}


@router.get("/analytics/drift", response_model=PredictionDriftResponse)
def analytics_drift_endpoint(
    baseline_days: int = Query(30, ge=2, le=3650),
    recent_days: int = Query(7, ge=1, le=3650),
    limit: int = Query(200, ge=1, le=2000),
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
) -> dict[str, Any]:
    items = analytics_drift(
        baseline_days=int(baseline_days),
        recent_days=int(recent_days),
        limit=int(limit),
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=horizon_steps,
    )
    return {"ok": True, "baseline_days": int(baseline_days), "recent_days": int(recent_days), "items": items}


@router.get("/analytics/calibrations", response_model=CalibrationsResponse)
def analytics_calibrations(
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
    limit: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    items = analytics_list_calibrations(
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=horizon_steps,
        limit=int(limit),
    )
    return {"ok": True, "items": items}


@router.post("/retention/cleanup", response_model=PredictionRetentionCleanupResponse)
def retention_cleanup(
    max_age_days: int | None = Query(None, ge=0, le=3650),
    max_rows: int | None = Query(None, ge=0, le=50_000_000),
) -> dict[str, Any]:
    # Explicit admin action: always runs cleanup once.
    used_max_age_days = int(settings.PREDICTION_RETENTION_MAX_AGE_DAYS or 0) if max_age_days is None else int(max_age_days)
    used_max_rows = int(settings.PREDICTION_RETENTION_MAX_ROWS or 0) if max_rows is None else int(max_rows)

    with db_conn() as conn:
        result = cleanup_prediction_events(
            conn=conn,
            enabled=True,
            max_age_days=used_max_age_days,
            max_rows=used_max_rows,
        )

    return {
        "ok": True,
        "enabled": True,
        "max_age_days": int(result.max_age_days),
        "max_rows": int(result.max_rows),
        "deleted_by_age": int(result.deleted_by_age),
        "deleted_by_rows": int(result.deleted_by_rows),
        "remaining_rows": int(result.remaining_rows),
    }
