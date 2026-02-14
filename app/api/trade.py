from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.api.strategy import trade_decision as _strategy_trade_decision
from app.candles.service import CandleService
from app.core.audit import log_event
from app.core.controls import assert_new_orders_allowed
from app.core.db import db_conn
from app.core.guards import assert_trade_allowed
from app.core.settings import settings
from app.execution.service import ExecutionService
from app.broker.paper import PaperBroker
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError
from app.integrations.upstox.order_builder import build_equity_intraday_market_order
from app.orders.state import create_order, update_order

router = APIRouter(prefix="/trade", tags=["trade"])

_candles = CandleService()
_paper_execution = ExecutionService(PaperBroker())


def _resolve_upstox_token(*, instrument_key: str) -> str | None:
    key = str(instrument_key or "").strip()
    if not key:
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT upstox_token FROM instrument_meta WHERE instrument_key=? LIMIT 1",
            (key,),
        ).fetchone()
        token = None if row is None else row["upstox_token"]
        token_s = (str(token).strip() if token is not None else "")
        return token_s or None


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


@router.post("/execute")
def execute_trade(
    instrument_key: str = Query(...),
    interval: str = Query("1m"),
    account_balance: float = Query(..., gt=0),
    lot_size: int = Query(1, gt=0),
    broker: str = Query("paper"),
) -> dict:
    """Strategy → sizing → execution.

    - `broker=paper`: uses PaperBroker.
    - `broker=upstox`: places a live Upstox intraday MARKET order.
    """

    try:
        assert_new_orders_allowed()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    ik = str(instrument_key or "").strip()
    if not ik:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    b = str(broker or "paper").strip().lower()
    if b not in {"paper", "upstox"}:
        raise HTTPException(status_code=400, detail="broker must be paper|upstox")

    # Market-hours guard for live trading only.
    if b == "upstox":
        if not bool(getattr(settings, "LIVE_TRADING_ENABLED", False)):
            raise HTTPException(status_code=403, detail="LIVE_TRADING_ENABLED=false: live trading disabled")
        if settings.SAFE_MODE:
            raise HTTPException(status_code=403, detail="SAFE_MODE=true: live trading disabled")
        try:
            assert_trade_allowed()
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

    # Sizing based on current strategy decision.
    sized = _strategy_trade_decision(
        instrument_key=ik,
        interval=str(interval),
        account_balance=float(account_balance),
        lot_size=int(lot_size),
    )

    action = str(sized.get("action") or "HOLD").upper()
    qty = int(sized.get("qty") or 0)
    if action not in {"BUY", "SELL"} or qty <= 0:
        return {"status": "REJECTED", "reason": "NO_TRADE", "action": action, "quantity": int(qty)}

    # Best-effort last price for reporting (and paper fill).
    price = None
    try:
        series = _candles.poll_intraday(ik, str(interval), lookback_minutes=60)
        candles = list(series.candles or [])
        if candles:
            price = float(candles[-1].close)
    except Exception:
        price = None

    if b == "paper":
        if price is None or float(price) <= 0:
            return {"status": "REJECTED", "reason": "no price available; run intraday poll first"}
        try:
            res = _paper_execution.place(symbol=ik, side=action, qty=float(qty), price=float(price))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "ACCEPTED", "result": res, "action": action, "quantity": qty, "price": float(price)}

    # Live Upstox
    token = _resolve_upstox_token(instrument_key=ik)
    if not token:
        raise HTTPException(
            status_code=400,
            detail="instrument_key not mapped to upstox_token. Import exchange instruments via /api/universe/import-upstox-exchange",
        )

    tag = f"strategy:{ik}"[:24]
    body = build_equity_intraday_market_order(instrument_token=str(token), side=action, qty=int(qty), tag=tag)

    # Persist intent BEFORE calling the broker (crash-safe).
    state_id = create_order(
        broker="upstox",
        instrument_key=str(body.get("instrument_token") or ""),
        side=str(body.get("transaction_type") or action).upper(),
        qty=float(body.get("quantity") or qty),
        order_kind="STRATEGY",
        order_type=str(body.get("order_type") or "UPSTOX"),
        status="NEW",
        meta={"body": body, "strategy": sized},
    )

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.place_order_v2(body)
        data = res.get("data") if isinstance(res, dict) else None
        broker_order_id = None
        if isinstance(data, dict):
            broker_order_id = data.get("order_id") or data.get("orderId")

        update_order(
            state_id,
            status="SUBMITTED",
            broker_order_id=(None if not broker_order_id else str(broker_order_id)),
            meta_patch={"result": res, "last_price": price},
        )
        log_event("trade.execute.upstox", {"instrument_key": ik, "action": action, "qty": qty, "result": res})
        return {**res, "order_state_id": state_id, "action": action, "quantity": qty, "price": price}
    except UpstoxError as e:
        update_order(state_id, status="ERROR", last_error=str(e)[:500], meta_patch={"error": str(e)})
        log_event("trade.execute.upstox_error", {"instrument_key": ik, "error": str(e)})
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()
