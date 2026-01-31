from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.api.strategy import trade_decision as _strategy_trade_decision

router = APIRouter(prefix="/trade", tags=["trade"])


@router.get("/decide")
def decide_trade(
    instrument_key: str = Query(...),
    interval: str = Query(...),
    account_balance: float = Query(..., gt=0),
    lot_size: int = Query(1, gt=0),
) -> dict:
    """BackendComplete-compatible trade decision endpoint.

    This is a thin wrapper around the existing sizing logic in /api/strategy/decision,
    normalized to the TradeDecision field names the frontend expects.
    """

    try:
        sized = _strategy_trade_decision(
            instrument_key=instrument_key,
            interval=interval,
            account_balance=account_balance,
            lot_size=lot_size,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    action = str(sized.get("action") or "HOLD")
    entry_price = sized.get("entry")
    stop_loss = sized.get("stop_loss")
    target = sized.get("target")
    confidence = sized.get("confidence")
    rr = sized.get("rr")
    qty = int(sized.get("qty") or 0)

    risk_pct = 1.0
    try:
        capital_used = float(qty) * float(entry_price or 0.0)
    except Exception:
        capital_used = 0.0

    return {
        "instrument_key": instrument_key,
        "interval": str(interval),
        "action": action,
        "quantity": qty,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "confidence": confidence,
        "confidence_raw": rr,
        "risk_pct": risk_pct,
        "capital_used": capital_used,
        "reason": sized.get("reason"),
        "timestamp": datetime.now(timezone.utc),
        # keep raw for debugging/UI Details panel
        "raw": sized,
    }
