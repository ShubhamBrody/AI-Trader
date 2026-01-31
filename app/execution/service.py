from __future__ import annotations

import hashlib

from app.broker.base import Broker


class ExecutionService:
    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._seen: set[str] = set()

    def _dedupe_key(self, symbol: str, side: str, qty: float, price: float | None) -> str:
        raw = f"{symbol}|{side}|{qty}|{price}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def place(self, symbol: str, side: str, qty: float, price: float | None = None) -> dict:
        if qty <= 0:
            raise ValueError("qty must be > 0")
        if side.upper() not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        key = self._dedupe_key(symbol, side, qty, price)
        if key in self._seen:
            return {"status": "ignored", "reason": "duplicate"}
        self._seen.add(key)

        return self._broker.place_order(symbol=symbol, side=side, qty=qty, price=price)
