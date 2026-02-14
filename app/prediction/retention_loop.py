from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.db import db_conn
from app.core.settings import settings
from app.prediction.retention import cleanup_prediction_events

logger = logging.getLogger(__name__)


@dataclass
class PredictionRetentionStatus:
    running: bool
    last_cycle_ts: int | None
    last_error: str | None
    last_result: dict[str, Any] | None


class PredictionRetentionLoop:
    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._status = PredictionRetentionStatus(running=False, last_cycle_ts=None, last_error=None, last_result=None)

    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._status.running),
            "last_cycle_ts": self._status.last_cycle_ts,
            "last_error": self._status.last_error,
            "enabled": bool(settings.PREDICTION_RETENTION_ENABLED),
            "max_age_days": int(settings.PREDICTION_RETENTION_MAX_AGE_DAYS or 0),
            "max_rows": int(settings.PREDICTION_RETENTION_MAX_ROWS or 0),
            "poll_seconds": int(settings.PREDICTION_RETENTION_POLL_SECONDS or 0),
            "last_result": self._status.last_result,
        }

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if not settings.PREDICTION_RETENTION_ENABLED:
            return
        self._status.running = True
        self._status.last_error = None
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._status.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run_loop(self) -> None:
        poll = max(30, int(settings.PREDICTION_RETENTION_POLL_SECONDS or 0) or 3600)
        while self._status.running:
            try:
                self.run_cycle()
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._status.last_error = str(e)
                logger.exception("prediction retention loop error: %s", e)
            await asyncio.sleep(poll)

    def run_cycle(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with db_conn() as conn:
            result = cleanup_prediction_events(
                conn=conn,
                enabled=True,
                max_age_days=int(settings.PREDICTION_RETENTION_MAX_AGE_DAYS or 0),
                max_rows=int(settings.PREDICTION_RETENTION_MAX_ROWS or 0),
                now=now,
            )
        payload: dict[str, Any] = {
            "enabled": bool(settings.PREDICTION_RETENTION_ENABLED),
            "max_age_days": int(result.max_age_days),
            "max_rows": int(result.max_rows),
            "deleted_by_age": int(result.deleted_by_age),
            "deleted_by_rows": int(result.deleted_by_rows),
            "remaining_rows": int(result.remaining_rows),
        }
        self._status.last_cycle_ts = int(now.timestamp())
        self._status.last_result = payload
        return payload


RETENTION = PredictionRetentionLoop()
