from __future__ import annotations

from app.broker.base import Broker
from app.paper_trading.service import PaperTradingService


class PaperBroker(Broker):
    def __init__(self) -> None:
        self._svc = PaperTradingService()

    def place_order(self, symbol: str, side: str, qty: float, price: float | None = None) -> dict:
        if price is None:
            raise ValueError("paper broker requires price")
        return self._svc.execute(symbol=symbol, side=side, qty=qty, price=price)

    def cancel_order(self, order_id: str) -> dict:
        return {"cancelled": False, "reason": "paper broker executes immediately"}

    def positions(self) -> dict:
        return self._svc.positions()

    def balance(self) -> dict:
        return self._svc.account()
