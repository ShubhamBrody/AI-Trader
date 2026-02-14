from __future__ import annotations

from app.broker.base import Broker
from app.core.settings import settings


class UpstoxBroker(Broker):
    """Safe-mode stub.

    Intentionally refuses to place live orders unless SAFE_MODE=false and creds exist.
    """

    def _assert_enabled(self) -> None:
        if not bool(getattr(settings, "LIVE_TRADING_ENABLED", False)):
            raise PermissionError("LIVE_TRADING_ENABLED=false: live orders disabled")
        if settings.SAFE_MODE:
            raise PermissionError("SAFE_MODE=true: live orders disabled")
        if not settings.UPSTOX_ACCESS_TOKEN:
            raise PermissionError("UPSTOX_ACCESS_TOKEN not set")

    def place_order(self, symbol: str, side: str, qty: float, price: float | None = None) -> dict:
        self._assert_enabled()
        raise NotImplementedError("Wire actual Upstox order placement here")

    def cancel_order(self, order_id: str) -> dict:
        self._assert_enabled()
        raise NotImplementedError

    def positions(self) -> dict:
        self._assert_enabled()
        raise NotImplementedError

    def balance(self) -> dict:
        self._assert_enabled()
        raise NotImplementedError
