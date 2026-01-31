from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.core.audit import log_event
from app.core.controls import get_controls, set_agent_disabled, set_freeze_new_orders
from app.core.settings import settings
from app.orders.state import reconcile_upstox_orders
from app.portfolio.positions_state import broker_pnl_estimate, reconcile_upstox_positions
from app.realtime.bus import publish_sync
from app.alerts.service import AlertService

logger = logging.getLogger(__name__)

ALERTS = AlertService()


@dataclass
class LiveReconcileStatus:
    running: bool
    last_cycle_ts: int | None
    last_error: str | None


class LiveReconciler:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._status = LiveReconcileStatus(running=False, last_cycle_ts=None, last_error=None)

    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._status.running),
            "last_cycle_ts": self._status.last_cycle_ts,
            "last_error": self._status.last_error,
            "enabled": bool(settings.LIVE_RECONCILE_ENABLED),
        }

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if not settings.LIVE_RECONCILE_ENABLED:
            return
        self._status.running = True
        self._status.last_error = None
        self._task = asyncio.create_task(self._run_loop())
        publish_sync("agent", "reconcile.start", self.status())

    async def stop(self) -> None:
        self._status.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        publish_sync("agent", "reconcile.stop", self.status())

    async def _run_loop(self) -> None:
        poll = max(5, int(settings.LIVE_RECONCILE_POLL_SECONDS))
        while self._status.running:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._status.last_error = str(e)
                logger.exception("live reconcile loop error: %s", e)
                publish_sync("agent", "reconcile.error", {"error": str(e)[:400]})
            await asyncio.sleep(poll)

    async def run_cycle(self) -> dict[str, Any]:
        # Read-only broker calls. Even in SAFE_MODE this is ok.
        if not settings.UPSTOX_ACCESS_TOKEN:
            return {"ok": False, "detail": "UPSTOX_ACCESS_TOKEN not configured"}

        controls = get_controls()
        # Always run reconciliation even if frozen; freeze blocks NEW orders only.

        pos = reconcile_upstox_positions()
        orders = reconcile_upstox_orders(limit=200)

        # Alerts: unknown open orders at broker not in DB.
        unknown_n = int(orders.get("unknown_open_count") or 0)
        if unknown_n > 0:
            payload = {"unknown_open_count": unknown_n, "unknown_open": orders.get("unknown_open")}
            log_event("alert.broker_unknown_orders", payload)
            publish_sync("alerts", "alert.broker_unknown_orders", payload)
            publish_sync("agent", "alert.broker_unknown_orders", payload)
            ALERTS.notify(alert_type="broker_unknown_orders", payload=payload)

            # Optional automatic failsafes.
            if settings.FAILSAFE_AUTOFREEZE_ON_UNKNOWN_BROKER_ORDERS and not controls.freeze_new_orders:
                set_freeze_new_orders(enabled=True, actor="reconciler")
                publish_sync("alerts", "controls.autofreeze", {"reason": "unknown_broker_orders", "enabled": True})
                log_event("alert.autofreeze", {"reason": "unknown_broker_orders"})
                ALERTS.notify(alert_type="autofreeze", payload={"reason": "unknown_broker_orders"})
            if settings.FAILSAFE_DISABLE_AGENT_ON_UNKNOWN_BROKER_ORDERS and not controls.agent_disabled:
                set_agent_disabled(disabled=True, actor="reconciler")
                publish_sync("alerts", "controls.autodisable_agent", {"reason": "unknown_broker_orders", "disabled": True})
                log_event("alert.autodisable_agent", {"reason": "unknown_broker_orders"})
                ALERTS.notify(alert_type="autodisable_agent", payload={"reason": "unknown_broker_orders"})

        # Alerts: daily loss based on broker truth (best-effort)
        pnl = broker_pnl_estimate(broker="upstox")
        if pnl is not None and float(settings.AUTOTRADER_MAX_DAILY_LOSS_INR or 0.0) > 0:
            max_loss = float(settings.AUTOTRADER_MAX_DAILY_LOSS_INR)
            if float(pnl) <= -abs(max_loss):
                payload = {"pnl_estimate": float(pnl), "max_daily_loss": float(max_loss), "freeze": controls.freeze_new_orders}
                log_event("alert.daily_loss_breached", payload)
                publish_sync("alerts", "alert.daily_loss_breached", payload)
                publish_sync("agent", "alert.daily_loss_breached", payload)
                ALERTS.notify(alert_type="daily_loss_breached", payload=payload)

                # Optional automatic failsafes.
                if settings.FAILSAFE_AUTOFREEZE_ON_DAILY_LOSS and not controls.freeze_new_orders:
                    set_freeze_new_orders(enabled=True, actor="reconciler")
                    publish_sync("alerts", "controls.autofreeze", {"reason": "daily_loss_breached", "enabled": True, "pnl": float(pnl)})
                    log_event("alert.autofreeze", {"reason": "daily_loss_breached", "pnl": float(pnl)})
                    ALERTS.notify(alert_type="autofreeze", payload={"reason": "daily_loss_breached", "pnl": float(pnl)})
                if settings.FAILSAFE_DISABLE_AGENT_ON_DAILY_LOSS and not controls.agent_disabled:
                    set_agent_disabled(disabled=True, actor="reconciler")
                    publish_sync("alerts", "controls.autodisable_agent", {"reason": "daily_loss_breached", "disabled": True, "pnl": float(pnl)})
                    log_event("alert.autodisable_agent", {"reason": "daily_loss_breached", "pnl": float(pnl)})
                    ALERTS.notify(alert_type="autodisable_agent", payload={"reason": "daily_loss_breached", "pnl": float(pnl)})

        out = {
            "ok": True,
            "positions": pos,
            "orders": {
                "checked": orders.get("checked"),
                "updated": orders.get("updated"),
                "unknown_open_count": orders.get("unknown_open_count"),
            },
        }
        publish_sync("agent", "reconcile.cycle", out)
        return out


RECONCILER = LiveReconciler()
