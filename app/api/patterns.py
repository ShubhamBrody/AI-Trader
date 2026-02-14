from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.candlestick_patterns import CATALOG, detect_patterns, to_candles
from app.ai.pattern_memory import apply_feedback, ensure_catalog_rows, get_stats, get_weights
from app.ai.engine import AIEngine
from app.candles.service import CandleService
from app.core.settings import settings

router = APIRouter(prefix="/patterns", tags=["patterns"])

_ai = AIEngine()


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
    raw_candles = list(series.candles or [])

    # Optional seq model: if enabled and available, prefer it for the primary output.
    seq_used = False
    matches: list[Any] = []
    if bool(getattr(settings, "PATTERN_SEQ_MODEL_ENABLED", False)):
        try:
            from app.learning.pattern_seq import PATTERN_INDEX, torch_available, predict_patterns_from_candles

            if torch_available():
                model_path = str(getattr(settings, "PATTERN_SEQ_MODEL_PATH", "data/models/pattern_seq.pt"))
                out = predict_patterns_from_candles(raw_candles, model_path=model_path, max_results=int(max_results))
                if out.get("ok") and isinstance(out.get("top"), list):
                    # Map seq probs back into catalog metadata.
                    catalog_by_name = {p.name: p for p in CATALOG}
                    seq_matches: list[dict[str, Any]] = []
                    for row in list(out.get("top") or [])[: int(max_results)]:
                        name = str(row.get("name") or "")
                        prob = float(row.get("prob") or 0.0)
                        p = catalog_by_name.get(name)
                        if p is None:
                            continue
                        seq_matches.append(
                            {
                                "name": p.name,
                                "family": p.family,
                                "side": p.side,
                                "window": int(getattr(p, "min_window", 0) or 0),
                                "confidence": float(prob),
                                "details": {"prob": float(prob), "source": "seq"},
                            }
                        )

                    # Ensure deterministic ordering
                    seq_matches.sort(key=lambda d: float(d.get("confidence") or 0.0), reverse=True)
                    matches = seq_matches
                    seq_used = True
        except Exception:
            seq_used = False

    # Fallback: classic pattern detector + AI re-ranking.
    if not seq_used:
        cs = to_candles(raw_candles)
        weights = get_weights()
        matches = list(detect_patterns(cs, weights=weights, max_results=max_results))

    # AI-first re-ranking: keep the same match objects, but prefer those
    # whose side aligns with the current AI directional signal.
    ai_side: str | None = None
    try:
        lb_days = 3 if str(interval).endswith("m") else 60
        pred = _ai.predict(str(instrument_key), interval=str(interval), lookback_days=int(lb_days), horizon_steps=1)
        sig = str((pred.get("prediction") or {}).get("signal") or "").upper()
        if sig == "BUY":
            ai_side = "buy"
        elif sig == "SELL":
            ai_side = "sell"
        elif sig == "HOLD":
            ai_side = "neutral"
    except Exception:
        ai_side = None

    if (not seq_used) and ai_side in {"buy", "sell"} and matches:
        def _score(m: Any) -> float:
            c = float(getattr(m, "confidence", 0.0) or 0.0)
            side = str(getattr(m, "side", "neutral") or "neutral").lower()
            # Strong bonus when aligned; small penalty when opposed.
            if side == ai_side:
                return c * 1.20
            if side in {"buy", "sell"} and side != ai_side:
                return c * 0.92
            return c

        matches = sorted(list(matches), key=_score, reverse=True)
    return {
        "ok": True,
        "instrument_key": instrument_key,
        "interval": interval,
        "n": len(raw_candles),
        "matches": (
            matches
            if seq_used
            else [
                {
                    "name": m.name,
                    "family": m.family,
                    "side": m.side,
                    "window": m.window,
                    "confidence": m.confidence,
                    "details": m.details,
                }
                for m in matches
            ]
        ),
        "best": (
            None
            if not matches
            else (
                {"name": matches[0]["name"], "side": matches[0]["side"], "confidence": matches[0]["confidence"]}
                if seq_used
                else {"name": matches[0].name, "side": matches[0].side, "confidence": matches[0].confidence}
            )
        ),
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
