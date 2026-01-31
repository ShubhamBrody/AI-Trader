from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.orders.state import get_order, list_events, list_orders, reconcile_upstox_orders

router = APIRouter(prefix="/orders/state", tags=["orders"])


class ReconcileRequest(BaseModel):
    broker: str = Field(default="upstox")
    order_ids: list[str] | None = None
    limit: int = Field(default=200, ge=1, le=1000)


@router.get("/list")
def orders_state_list(
    limit: int = 100,
    broker: str | None = None,
    status: str | None = None,
    instrument_key: str | None = None,
) -> dict[str, Any]:
    items = list_orders(limit=int(limit), broker=broker, status=status, instrument_key=instrument_key)
    return {
        "ok": True,
        "items": [
            {
                **o.__dict__,
                "meta": o.meta,
            }
            for o in items
        ],
    }


@router.get("/{order_id}")
def orders_state_get(order_id: str) -> dict[str, Any]:
    o = get_order(str(order_id))
    if o is None:
        raise HTTPException(status_code=404, detail="order not found")
    return {"ok": True, "order": {**o.__dict__, "meta": o.meta}}


@router.get("/{order_id}/events")
def orders_state_events(order_id: str, limit: int = 200) -> dict[str, Any]:
    return {"ok": True, "order_id": str(order_id), "events": list_events(str(order_id), limit=int(limit))}


@router.post("/reconcile")
def orders_state_reconcile(req: ReconcileRequest) -> dict[str, Any]:
    broker = str(req.broker or "").lower().strip()
    if broker != "upstox":
        return {"ok": True, "detail": f"no reconciliation needed for broker={broker}", "checked": 0, "updated": 0}

    res = reconcile_upstox_orders(order_ids=req.order_ids, limit=int(req.limit))
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res
