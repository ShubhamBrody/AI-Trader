from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from app.ai.engine import AIEngine

router = APIRouter(prefix="/ai")
engine = AIEngine()


@router.get("/predict")
def predict(
    instrument_key: str = Query(...),
    include_nifty: bool = Query(False),
) -> dict:
    # BackendComplete + frontend contract: flattened keys.
    raw = engine.predict(
        instrument_key=instrument_key,
        interval="1d",
        lookback_days=60,
        horizon_steps=1,
        include_nifty=include_nifty,
    )

    pred = (raw.get("prediction") or {}) if isinstance(raw, dict) else {}
    meta = (raw.get("meta") or {}) if isinstance(raw, dict) else {}
    ohlc = (pred.get("next_hour_ohlc") or {}) if isinstance(pred, dict) else {}

    action = pred.get("signal") if isinstance(pred, dict) else None
    confidence = pred.get("confidence") if isinstance(pred, dict) else None
    uncertainty = pred.get("uncertainty") if isinstance(pred, dict) else None
    agreement = pred.get("ensemble_agreement") if isinstance(pred, dict) else None
    model_name = meta.get("model") if isinstance(meta, dict) else None
    data_quality = meta.get("data_quality") if isinstance(meta, dict) else None

    reasons = [
        f"Model={model_name}" if model_name else "Model=unknown",
        f"DataQuality={data_quality:.2f}" if isinstance(data_quality, (int, float)) else "DataQuality=unknown",
    ]
    if isinstance(action, str):
        reasons.append(f"Action={action}")

    return {
        "instrument_key": raw.get("instrument_key") if isinstance(raw, dict) else instrument_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_ohlc": {
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "close": ohlc.get("close"),
        },
        "action": action,
        "action_confidence": confidence,
        "overall_confidence": confidence,
        "uncertainty": uncertainty,
        "ensemble_agreement": agreement,
        "reasons": reasons,
        "model_version": model_name,
        "inference_time_ms": None,
        "data_quality_score": data_quality,
        "raw": raw,
    }


@router.get("/predict/batch")
def predict_batch(
    instrument_keys: str = Query(..., description="Comma-separated instrument keys"),
    include_nifty: bool = Query(False),
) -> dict:
    keys = [k.strip() for k in instrument_keys.split(",") if k.strip()]
    predictions: list[dict] = []
    for k in keys:
        try:
            raw = engine.predict(
                instrument_key=k,
                interval="1d",
                lookback_days=60,
                horizon_steps=1,
                include_nifty=include_nifty,
            )
            pred = (raw.get("prediction") or {}) if isinstance(raw, dict) else {}
            ohlc = (pred.get("next_hour_ohlc") or {}) if isinstance(pred, dict) else {}
            predictions.append(
                {
                    "instrument_key": raw.get("instrument_key") if isinstance(raw, dict) else k,
                    "action": pred.get("signal") if isinstance(pred, dict) else None,
                    "confidence": pred.get("confidence") if isinstance(pred, dict) else None,
                    "predicted_close": ohlc.get("close"),
                }
            )
        except Exception as e:
            predictions.append({"instrument_key": k, "error": str(e)})

    return {"predictions": predictions, "count": len(predictions)}
