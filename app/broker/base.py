from __future__ import annotations

from abc import ABC, abstractmethod


class Broker(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float, price: float | None = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def balance(self) -> dict:
        raise NotImplementedError
