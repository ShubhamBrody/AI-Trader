from __future__ import annotations

from fastapi import APIRouter

from app.alerts.service import AlertService

router = APIRouter(prefix="/alerts", tags=["alerts"])
svc = AlertService()


@router.get("/recent")
def recent(limit: int = 50) -> dict:
    return {"events": svc.recent_alert_events(limit=limit)}
