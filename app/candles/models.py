from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Candle(BaseModel):
    ts: int = Field(..., description="Epoch seconds")
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleSeries(BaseModel):
    instrument_key: str
    interval: str
    candles: list[Candle]


class LoadHistoricalRequest(BaseModel):
    instrument_key: str
    interval: str = Field("1d", description="e.g. 1d, 5m, 15m")
    start: datetime
    end: datetime


class BulkLoadRequest(BaseModel):
    instrument_key: str
    interval: str = "1d"
    num_trading_sessions: int = Field(100, ge=1, le=1000)


class IntradayPollRequest(BaseModel):
    instrument_key: str
    interval: str = "5m"
    lookback_minutes: int = Field(60, ge=5, le=6 * 60)
