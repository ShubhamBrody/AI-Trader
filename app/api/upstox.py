from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException

from app.core.settings import settings
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError

router = APIRouter(prefix="/upstox", tags=["upstox"])


@router.get("/status")
def upstox_status(instrument_key: str = "NSE_INDEX|Nifty 50", interval: str = "days", interval_value: int = 1):
    """Lightweight diagnostics for Upstox connectivity.

    Calls Upstox V3 historical candle endpoint using the configured bearer token.
    Does not place orders and is safe under SAFE_MODE.
    """

    if not settings.UPSTOX_ACCESS_TOKEN:
        raise HTTPException(status_code=400, detail="UPSTOX_ACCESS_TOKEN is not configured")

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        # Ask for a small window (today and yesterday) to keep response small.
        now = datetime.now(timezone.utc).date()
        prev = (datetime.now(timezone.utc) - timedelta(days=5)).date()
        rows = client.historical_candles_v3(
            instrument_key=instrument_key,
            unit=interval,
            interval=interval_value,
            to_date=now,
            from_date=prev,
        )
        return {
            "ok": True,
            "strict": settings.UPSTOX_STRICT,
            "base_url": settings.UPSTOX_BASE_URL,
            "instrument_key": instrument_key,
            "unit": interval,
            "interval": interval_value,
            "rows": len(rows or []),
        }
    except UpstoxError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()
