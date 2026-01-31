from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.watchlist.service import WatchlistService

router = APIRouter(prefix="/watchlist", tags=["watchlist"])
svc = WatchlistService()


class WatchItem(BaseModel):
    instrument_key: str
    label: str | None = None


@router.get("")
def list_watchlist() -> dict:
    return {"items": svc.list()}


@router.post("")
def add(item: WatchItem) -> dict:
    return svc.upsert(item.instrument_key, label=item.label)


@router.delete("")
def remove(instrument_key: str) -> dict:
    return svc.remove(instrument_key)
