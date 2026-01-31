from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.portfolio.service import PortfolioService

router = APIRouter(prefix="/positions", tags=["positions"])

svc = PortfolioService()


@router.get("/state")
def positions_state(broker: str = "upstox", limit: int = 500) -> dict[str, Any]:
    return svc.positions_state(broker=broker, limit=limit)


@router.post("/reconcile")
def positions_reconcile(broker: str = "upstox") -> dict[str, Any]:
    res = svc.reconcile_positions(broker=broker)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res
