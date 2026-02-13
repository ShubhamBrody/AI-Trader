from __future__ import annotations

from fastapi import APIRouter

from app.realtime.bus import BUS

router = APIRouter(prefix="/realtime", tags=["realtime"])


@router.get("/agent-events")
async def agent_events(limit: int = 200, since_id: int | None = None) -> dict:
    items = await BUS.recent("agent", limit=limit, since_id=since_id)
    return {"ok": True, "events": [e.to_dict() for e in items]}


@router.get("/training-events")
async def training_events(limit: int = 200, since_id: int | None = None) -> dict:
    items = await BUS.recent("training", limit=limit, since_id=since_id)
    return {"ok": True, "events": [e.to_dict() for e in items]}


@router.get("/hft-events")
async def hft_events(limit: int = 200, since_id: int | None = None) -> dict:
    items = await BUS.recent("hft", limit=limit, since_id=since_id)
    return {"ok": True, "events": [e.to_dict() for e in items]}
