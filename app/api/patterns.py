from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.candlestick_patterns import CATALOG, detect_patterns, to_candles
from app.ai.pattern_memory import apply_feedback, ensure_catalog_rows, get_stats, get_weights
from app.candles.service import CandleService
from app.core.settings import settings

router = APIRouter(prefix="/patterns", tags=["patterns"])


def _ensure() -> None:
    # Ensure DB has rows for all catalog patterns.
    ensure_catalog_rows(
        [
            {
                "name": p.name,
                "family": p.family,
                "side": p.side,
                "base_reliability": float(p.base_reliability),
                "params": {"min_window": int(p.min_window), "best_timeframe": p.best_timeframe},
            }
            for p in CATALOG
        ]
    )


@router.get("/catalog")
def catalog() -> dict[str, Any]:
    _ensure()
    weights = get_weights()
    return {
        "ok": True,
        "count": len(CATALOG),
        "patterns": [
            {
                "name": p.name,
                "family": p.family,
                "side": p.side,
                "best_timeframe": p.best_timeframe,
                "min_window": p.min_window,
                "base_reliability": p.base_reliability,
                "weight": float(weights.get(p.name, 1.0)),
            }
            for p in CATALOG
        ],
    }


@router.get("/stats")
def stats(limit: int = 200) -> dict[str, Any]:
    _ensure()
    items = get_stats(limit=limit)
    return {
        "ok": True,
        "items": [
            {
                "name": s.name,
                "weight": s.weight,
                "seen": s.seen,
                "wins": s.wins,
                "losses": s.losses,
                "ema_winrate": s.ema_winrate,
            }
            for s in items
        ],
    }


@router.get("/detect")
def detect(
    instrument_key: str,
    interval: str = "1m",
    lookback_minutes: int = 120,
    max_results: int = 8,
) -> dict[str, Any]:
    if not instrument_key:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    _ensure()
    svc = CandleService()
    series = svc.poll_intraday(instrument_key, interval=interval, lookback_minutes=int(lookback_minutes))
    cs = to_candles(series.candles or [])
    weights = get_weights()
    matches = detect_patterns(cs, weights=weights, max_results=max_results)
    return {
        "ok": True,
        "instrument_key": instrument_key,
        "interval": interval,
        "n": len(cs),
        "matches": [
            {
                "name": m.name,
                "family": m.family,
                "side": m.side,
                "window": m.window,
                "confidence": m.confidence,
                "details": m.details,
            }
            for m in matches
        ],
        "best": (None if not matches else {"name": matches[0].name, "side": matches[0].side, "confidence": matches[0].confidence}),
    }


@router.get("/seq/status")
def seq_status() -> dict[str, Any]:
    """Status for the optional sequence-based model."""

    try:
        from app.learning.pattern_seq import torch_available

        torch_ok = bool(torch_available())
    except Exception:
        torch_ok = False

    return {
        "ok": True,
        "enabled": bool(getattr(settings, "PATTERN_SEQ_MODEL_ENABLED", False)),
        "torch_available": torch_ok,
        "model_path": str(getattr(settings, "PATTERN_SEQ_MODEL_PATH", "")),
        "seq_len": int(getattr(settings, "PATTERN_SEQ_MODEL_SEQ_LEN", 64) or 64),
    }


@router.get("/seq/predict")
def seq_predict(
    instrument_key: str,
    interval: str = "1m",
    lookback_minutes: int = 240,
    max_results: int = 8,
) -> dict[str, Any]:
    """Predict candlestick patterns using the sequence model.

    Requires:
    - settings.PATTERN_SEQ_MODEL_ENABLED=true
    - torch installed (requirements-deep.txt)
    - model file exists at settings.PATTERN_SEQ_MODEL_PATH
    """

    if not instrument_key:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    if not bool(getattr(settings, "PATTERN_SEQ_MODEL_ENABLED", False)):
        raise HTTPException(status_code=400, detail="sequence pattern model disabled (PATTERN_SEQ_MODEL_ENABLED=false)")

    try:
        from app.learning.pattern_seq import predict_patterns_from_candles
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"sequence model unavailable: {e}")

    svc = CandleService()
    series = svc.poll_intraday(instrument_key, interval=interval, lookback_minutes=int(lookback_minutes))
    candles = list(series.candles or [])

    model_path = str(getattr(settings, "PATTERN_SEQ_MODEL_PATH", "data/models/pattern_seq.pt"))
    out = predict_patterns_from_candles(candles, model_path=model_path, max_results=int(max_results))

    if not out.get("ok"):
        return {
            **out,
            "instrument_key": instrument_key,
            "interval": interval,
            "n": len(candles),
        }

    return {
        **out,
        "instrument_key": instrument_key,
        "interval": interval,
        "n": len(candles),
    }


class FeedbackRequest(BaseModel):
    pattern: str
    good: bool
    magnitude: float = Field(default=1.0, ge=0.2, le=5.0)
    note: str | None = None
    instrument_key: str | None = None
    interval: str | None = None
    asof_ts: int | None = None


@router.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    _ensure()
    res = apply_feedback(
        pattern=req.pattern,
        good=bool(req.good),
        magnitude=float(req.magnitude),
        note=req.note,
        instrument_key=req.instrument_key,
        interval=req.interval,
        asof_ts=req.asof_ts,
    )
    if res is None:
        raise HTTPException(status_code=404, detail="unknown pattern")
    return {
        "ok": True,
        "pattern": res.name,
        "weight": res.weight,
        "seen": res.seen,
        "wins": res.wins,
        "losses": res.losses,
        "ema_winrate": res.ema_winrate,
    }
