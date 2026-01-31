from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query

from app.core.audit import log_event
from app.news.service import NewsService

router = APIRouter(prefix="/news", tags=["news"])
svc = NewsService()


@router.post("/refresh")
def refresh(limit_per_feed: int = 20, per_feed: int | None = None, days: int | None = None) -> dict:
    # Frontend may call: /api/news/refresh?days=7&per_feed=20
    if per_feed is not None:
        limit_per_feed = int(per_feed)
    _ = days  # reserved for future use

    res = svc.fetch_and_store(limit_per_feed=int(limit_per_feed))
    log_event("news.refresh", res)
    return res


@router.get("/latest")
def latest(limit: int = 50) -> dict:
    return {"items": svc.latest(limit=limit)}


@router.get("/recent")
def recent(limit: int = 50, days: int = 7) -> dict:
    # Frontend calls: /api/news/recent?limit=50&days=7
    d = max(0, min(int(days), 3650))
    min_ts = None
    if d > 0:
        min_ts = int((datetime.now(timezone.utc) - timedelta(days=d)).timestamp())
    items = svc.recent(limit=int(limit), min_ts=min_ts)
    # BackendComplete contract key
    return {"news": items, "items": items}


@router.post("/ingest")
def ingest_news(
    title: str = Query(..., min_length=1),
    source: str = Query(..., min_length=1),
    raw_text: str = Query(..., min_length=1),
    instrument_key: str | None = Query(None),
    url: str | None = Query(None),
) -> dict:
    """Manual ingestion endpoint (BackendComplete-compatible)."""
    svc.ingest_and_store(
        title=title,
        source=source,
        raw_text=raw_text,
        instrument_key=instrument_key,
        url=url,
    )
    return {"status": "OK"}


@router.post("/seed")
def seed_demo_news(count: int = Query(8, ge=1, le=50)) -> dict:
    inserted = svc.seed_demo_if_empty(max_items=int(count))
    return {"status": "OK", "inserted": inserted}


@router.get("/market-sentiment")
def market_sentiment(lookback: int = 200) -> dict:
    return svc.market_sentiment(lookback=lookback)


@router.get("/for")
def news_for_instrument(instrument_key: str, limit: int = 50) -> dict:
    return {"instrument_key": instrument_key, "items": svc.match_to_instrument(instrument_key, limit=limit)}


@router.get("/symbol")
def news_for_symbol(
    q: str = Query(..., min_length=1, description="Symbol/theme/instrument_key. Examples: RELIANCE, SILVER, NSE_EQ|INE002A01018"),
    limit: int = Query(80, ge=1, le=200),
    days: int = Query(7, ge=0, le=3650),
) -> dict:
    min_ts = None
    if int(days) > 0:
        min_ts = int((datetime.now(timezone.utc) - timedelta(days=int(days))).timestamp())
    return svc.for_query(q, limit=int(limit), min_ts=min_ts)


@router.post("/mentions/rebuild")
def rebuild_mentions(limit_news: int = Query(300, ge=1, le=2000)) -> dict:
    return svc.rebuild_mentions(limit_news=int(limit_news))
