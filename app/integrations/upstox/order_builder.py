from __future__ import annotations

from typing import Any

from app.core.settings import settings


def build_equity_intraday_market_order(*, instrument_token: str, side: str, qty: int, tag: str) -> dict[str, Any]:
    tx = side.upper()
    if tx not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if qty <= 0:
        raise ValueError("qty must be > 0")

    # NOTE: Field names follow Upstox V3 conventions as commonly used.
    # If your Upstox account/API expects different keys, you can adjust here.
    return {
        "instrument_token": instrument_token,
        "quantity": int(qty),
        "transaction_type": tx,
        "order_type": settings.UPSTOX_ENTRY_ORDER_TYPE,
        "product": settings.UPSTOX_EQ_PRODUCT,
        "validity": settings.UPSTOX_EQ_VALIDITY,
        "price": 0,
        "trigger_price": 0,
        "disclosed_quantity": 0,
        "is_amo": False,
        "tag": tag,
    }


def build_equity_intraday_stop_order(*, instrument_token: str, side: str, qty: int, trigger_price: float, tag: str) -> dict[str, Any]:
    tx = side.upper()
    if tx not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if qty <= 0:
        raise ValueError("qty must be > 0")
    if trigger_price <= 0:
        raise ValueError("trigger_price must be > 0")

    order_type = settings.UPSTOX_STOP_ORDER_TYPE
    body: dict[str, Any] = {
        "instrument_token": instrument_token,
        "quantity": int(qty),
        "transaction_type": tx,
        "order_type": order_type,
        "product": settings.UPSTOX_EQ_PRODUCT,
        "validity": settings.UPSTOX_EQ_VALIDITY,
        "trigger_price": float(trigger_price),
        "disclosed_quantity": 0,
        "is_amo": False,
        "tag": tag,
    }

    # Some stop order types require explicit price.
    if order_type.upper() in {"SL", "STOP_LIMIT"}:
        body["price"] = float(trigger_price)
    else:
        body["price"] = 0

    return body
