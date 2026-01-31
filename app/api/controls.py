from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.service import AGENT_SINGLETON
from app.core.audit import log_event
from app.core.controls import get_controls, set_agent_disabled, set_freeze_new_orders
from app.core.guards import assert_trade_allowed
from app.core.settings import settings
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError
from app.integrations.upstox.order_builder import build_equity_intraday_market_order
from app.portfolio.positions_state import reconcile_upstox_positions, get_position_by_instrument_key
from app.realtime.bus import publish_sync

router = APIRouter(prefix="/controls", tags=["controls"])


class FreezeRequest(BaseModel):
    enabled: bool = Field(...)


class AgentDisabledRequest(BaseModel):
    disabled: bool = Field(...)


class SquareOffInstrumentRequest(BaseModel):
    instrument_key: str = Field(..., min_length=3)


@router.get("/status")
def status() -> dict[str, Any]:
    c = get_controls()
    return {"ok": True, "controls": c.__dict__}


@router.post("/freeze")
def freeze(req: FreezeRequest) -> dict[str, Any]:
    c = set_freeze_new_orders(enabled=bool(req.enabled), actor="api")
    log_event("controls.freeze", {"enabled": bool(req.enabled)})
    publish_sync("agent", "controls.freeze", {"enabled": bool(req.enabled)})
    return {"ok": True, "controls": c.__dict__}


@router.post("/agent/disabled")
def agent_disabled(req: AgentDisabledRequest) -> dict[str, Any]:
    c = set_agent_disabled(disabled=bool(req.disabled), actor="api")
    log_event("controls.agent_disabled", {"disabled": bool(req.disabled)})
    publish_sync("agent", "controls.agent_disabled", {"disabled": bool(req.disabled)})
    return {"ok": True, "controls": c.__dict__}


@router.post("/emergency/cancel-all-open-orders")
async def emergency_cancel_all_open_orders() -> dict[str, Any]:
    # Calls agent helper; guarded there by SAFE_MODE.
    res = await AGENT_SINGLETON.cancel_open_orders()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    log_event("controls.emergency.cancel_all_open_orders", res)
    publish_sync("agent", "controls.emergency.cancel_all_open_orders", res)
    return res


@router.post("/emergency/flatten")
async def emergency_flatten() -> dict[str, Any]:
    res = await AGENT_SINGLETON.flatten()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    log_event("controls.emergency.flatten", res)
    publish_sync("agent", "controls.emergency.flatten", res)
    return res


@router.post("/reconcile-now")
def reconcile_now() -> dict[str, Any]:
    # Reconcile broker truth (orders + positions). Read-only broker calls.
    pos = reconcile_upstox_positions() if settings.UPSTOX_ACCESS_TOKEN else {"ok": False, "detail": "UPSTOX_ACCESS_TOKEN not configured"}
    log_event("controls.reconcile_now", {"positions": pos})
    publish_sync("agent", "controls.reconcile_now", {"positions": pos})
    return {"ok": True, "positions": pos}


@router.post("/emergency/squareoff-instrument")
def squareoff_instrument(req: SquareOffInstrumentRequest) -> dict[str, Any]:
    if settings.SAFE_MODE:
        raise HTTPException(status_code=403, detail="SAFE_MODE=true: refusing to place live orders")
    try:
        assert_trade_allowed()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # Update positions first so we act on broker truth.
    pos_res = reconcile_upstox_positions()
    if not pos_res.get("ok"):
        raise HTTPException(status_code=400, detail={"detail": "failed to fetch positions", "positions": pos_res})

    st = get_position_by_instrument_key(req.instrument_key)
    if st is None:
        return {"ok": True, "detail": "no position found", "instrument_key": req.instrument_key}

    net_qty = float(st.get("net_qty") or 0.0)
    token = st.get("instrument_token")
    if not token or abs(net_qty) < 1e-9:
        return {"ok": True, "detail": "already flat", "instrument_key": req.instrument_key}

    side = "SELL" if net_qty > 0 else "BUY"
    qty = int(abs(net_qty))
    tag = f"system:squareoff:{req.instrument_key.replace('|', '_')}"

    body = build_equity_intraday_market_order(instrument_token=str(token), side=side, qty=qty, tag=tag)

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client: UpstoxClient | None = None
    try:
        client = UpstoxClient(cfg)
        res = client.place_order_v3(body)
        log_event("controls.emergency.squareoff_instrument", {"instrument_key": req.instrument_key, "side": side, "qty": qty, "result": res})
        publish_sync(
            "agent",
            "controls.emergency.squareoff_instrument",
            {"instrument_key": req.instrument_key, "side": side, "qty": qty, "result": res},
        )
        return {"ok": True, "instrument_key": req.instrument_key, "side": side, "qty": qty, "result": res}
    except UpstoxError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        if client is not None:
            client.close()
