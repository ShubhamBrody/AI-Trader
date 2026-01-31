from __future__ import annotations

from fastapi import APIRouter, Response

from app.core.settings import settings

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
def metrics() -> Response:
    if not settings.METRICS_ENABLED:
        return Response(content="# METRICS_DISABLED\n", media_type="text/plain")

    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        payload = generate_latest()
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
    except Exception:
        return Response(content="# prometheus_client not available\n", media_type="text/plain", status_code=501)
