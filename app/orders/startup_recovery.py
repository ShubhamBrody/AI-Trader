from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.core.db import db_conn
from app.core.settings import settings
from app.orders.state import reconcile_upstox_orders

logger = logging.getLogger(__name__)


_ACTIVE_STATUSES = ("NEW", "SUBMITTED", "OPEN", "UNKNOWN")


def run_startup_order_recovery() -> dict[str, Any]:
    """Best-effort crash recovery routine.

    Goal: after a restart, reconcile recent Upstox orders so we don't "lose" live broker
    orders that were submitted just before the app died.

    Notes:
    - Broker calls are read-only (order book + order details).
    - DB updates are local, to restore broker IDs + latest statuses.
    """

    if not settings.ORDER_RECOVERY_ON_STARTUP:
        return {"ok": True, "skipped": True, "reason": "disabled"}

    if not settings.UPSTOX_ACCESS_TOKEN:
        return {"ok": False, "skipped": True, "reason": "UPSTOX_ACCESS_TOKEN is not configured"}

    now = int(time.time())
    lookback_hours = max(1, int(settings.ORDER_RECOVERY_LOOKBACK_HOURS))
    since_ts = now - (lookback_hours * 3600)
    limit = max(1, min(int(settings.ORDER_RECOVERY_STARTUP_LIMIT), 1000))

    # Prefer orders that are "active" or missing broker_order_id.
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT id
            FROM order_states
            WHERE broker='upstox'
              AND ts_created >= ?
              AND (
                   broker_order_id IS NULL
                OR status IN ('NEW','SUBMITTED','OPEN','UNKNOWN')
              )
            ORDER BY ts_created DESC
            LIMIT ?
            """,
            (since_ts, limit),
        )
        ids = [str(r["id"]) for r in cur.fetchall()]

    if not ids:
        return {"ok": True, "skipped": True, "reason": "no candidates"}

    logger.info("startup order recovery: reconciling %s upstox orders", len(ids))
    try:
        return reconcile_upstox_orders(order_ids=ids, limit=limit)
    except Exception as e:
        logger.exception("startup order recovery failed: %s", e)
        return {"ok": False, "skipped": False, "detail": str(e)[:500]}


async def startup_order_recovery() -> dict[str, Any]:
    """Async wrapper suitable for FastAPI lifespan."""

    return await asyncio.to_thread(run_startup_order_recovery)
