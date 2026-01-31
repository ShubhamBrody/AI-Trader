from __future__ import annotations

from fastapi import APIRouter

from app.core.audit import recent_events

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/recent")
def audit_recent(limit: int = 100, event_type: str | None = None) -> dict:
    return {"events": recent_events(limit=limit, event_type=event_type)}
