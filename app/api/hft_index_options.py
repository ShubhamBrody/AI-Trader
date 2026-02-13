from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.settings import settings
from app.hft.index_options.service import HFT_INDEX_OPTIONS

router = APIRouter(prefix="/hft/index-options", tags=["hft"])


@router.get("/status")
def status() -> dict[str, Any]:
    return {"ok": True, **HFT_INDEX_OPTIONS.status()}


@router.post("/start")
async def start() -> dict[str, Any]:
    res = await HFT_INDEX_OPTIONS.start()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res


@router.post("/stop")
async def stop() -> dict[str, Any]:
    return await HFT_INDEX_OPTIONS.stop()


@router.post("/run-once")
async def run_once(force: bool = False) -> dict[str, Any]:
    if not bool(getattr(settings, "INDEX_OPTIONS_HFT_ENABLED", False)):
        raise HTTPException(status_code=400, detail="INDEX_OPTIONS_HFT_ENABLED=false")
    return await HFT_INDEX_OPTIONS.run_once(force_offmarket_paper=bool(force))


@router.post("/flatten")
async def flatten() -> dict[str, Any]:
    return await HFT_INDEX_OPTIONS.flatten()
