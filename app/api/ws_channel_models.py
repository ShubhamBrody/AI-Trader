from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Payload(BaseModel):
    """Base payload model for WS channel contracts.

    These models are intended as a stable contract reference for UI consumers.
    Producers may include additional fields; we allow extras for forward compatibility.
    """

    model_config = ConfigDict(extra="allow")


# --------------------
# prediction channel
# --------------------


class PredictionPredictorStartPayload(_Payload):
    instruments: list[str] = Field(default_factory=list)
    cfg: dict[str, Any] = Field(default_factory=dict)


class PredictionPredictorErrorPayload(_Payload):
    error: str


class PredictionCreatedPayload(_Payload):
    id: int
    instrument_key: str
    interval: str
    horizon_steps: int
    ts_pred: int
    ts_target: int
    pred_ret: float
    pred_ret_adj: float
    calib_bias: float
    model_kind: str


class PredictionResolvedPayload(_Payload):
    id: int
    instrument_key: str
    interval: str
    horizon_steps: int
    ts_pred: int
    ts_target: int
    actual_ret: float
    pred_ret_adj: float
    error: float
    calibration: dict[str, Any] = Field(default_factory=dict)


class PredictionPredictorCyclePayload(_Payload):
    created: int
    resolved: int
    anchor: int


# --------------------
# automation channel
# --------------------


class AutomationRunStartedPayload(_Payload):
    job_id: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class AutomationRunUpdatedPayload(_Payload):
    job_id: str
    progress: float | None = None
    message: str | None = None
    status: str | None = None
    stats: dict[str, Any] | None = None
    error: str | None = None


# --------------------
# training channel
# --------------------


class TrainingStartedPayload(_Payload):
    job_id: str
    kind: str
    instrument_key: str
    interval: str


class TrainingProgressPayload(_Payload):
    job_id: str
    progress: float
    message: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class TrainingSucceededPayload(_Payload):
    job_id: str
    result: dict[str, Any] = Field(default_factory=dict)


class TrainingFailedPayload(_Payload):
    job_id: str
    result: dict[str, Any] | None = None
    error: str | None = None


class TrainingAutomationErrorPayload(_Payload):
    error: str


# --------------------
# hft channel
# --------------------


class HftBrokerPayload(_Payload):
    broker: str


class HftStartPayload(_Payload):
    broker: str
    safe_mode: bool
    live_trading_enabled: bool


class HftErrorPayload(_Payload):
    error: str


class HftFlattenPayload(_Payload):
    closed: int
    errors: list[str] = Field(default_factory=list)


class HftEntryPayload(_Payload):
    trade_id: int
    instrument_key: str
    qty: int
    entry: float


class HftExitPayload(_Payload):
    trade_id: int
    instrument_key: str
    price: float
    pnl: float
    reason: str


class HftTradeRiskPayload(_Payload):
    trade_id: int
    stop: float
    target: float


class HftManageErrorPayload(_Payload):
    error: str
