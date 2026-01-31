from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.guards import assert_trade_allowed
from app.core.settings import settings
from app.core.controls import assert_new_orders_allowed
from app.core.audit import log_event
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError
from app.orders.state import create_order
from app.paper_trading.service import PaperTradingService

router = APIRouter(prefix="/orders", tags=["orders"])

paper = PaperTradingService()


class OrderPlaceRequest(BaseModel):
    broker: Literal["paper", "upstox"] = "paper"

    # Optional idempotency key. If provided, repeating the same request will
    # return the existing order_state instead of placing a duplicate order.
    client_order_id: str | None = None

    # Paper broker fields
    symbol: str | None = None
    side: Literal["BUY", "SELL"] | None = None
    qty: float | None = Field(default=None, gt=0)
    price: float | None = Field(default=None, gt=0)

    # Upstox broker fields
    upstox_body: dict[str, Any] | None = None


@router.get("/brokers")
def brokers() -> dict:
    return {
        "safe_mode": settings.SAFE_MODE,
        "supported": ["paper", "upstox"],
        "upstox_configured": bool(settings.UPSTOX_ACCESS_TOKEN),
    }


@router.post("/place")
def place(req: OrderPlaceRequest, request: Request) -> dict:
    try:
        assert_new_orders_allowed()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    if req.broker == "paper":
        if not (req.symbol and req.side and req.qty and req.price):
            raise HTTPException(status_code=400, detail="paper orders require symbol, side, qty, price")

        client_order_id = (str(req.client_order_id).strip() if req.client_order_id else None)
        if client_order_id:
            from app.orders.state import get_order_by_client_order_id

            existing = get_order_by_client_order_id(client_order_id)
            if existing is not None:
                meta = existing.meta or {}
                res = meta.get("result") if isinstance(meta, dict) else None
                if not isinstance(res, dict):
                    res = {"ok": True, "detail": "idempotent replay"}
                return {**res, "order_state_id": existing.id, "idempotent_replay": True}

        res = paper.execute(symbol=req.symbol, side=req.side, qty=req.qty, price=req.price)
        order_state_id = create_order(
            broker="paper",
            instrument_key=str(req.symbol),
            side=str(req.side),
            qty=float(req.qty),
            order_kind="MANUAL",
            order_type="PAPER",
            price=float(req.price),
            client_order_id=client_order_id,
            status="FILLED",
            meta={"request": req.model_dump(), "result": res},
        )
        log_event("order.paper.place", {"request": req.model_dump(), "result": res}, request_id=getattr(request.state, "request_id", None))
        return {**res, "order_state_id": order_state_id}

    # Upstox (live)
    if settings.SAFE_MODE:
        raise HTTPException(status_code=403, detail="SAFE_MODE=true: live orders disabled")

    # Market-hours guard for live trading
    try:
        assert_trade_allowed()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    if not settings.UPSTOX_ACCESS_TOKEN:
        raise HTTPException(status_code=400, detail="UPSTOX_ACCESS_TOKEN is not configured")
    if not req.upstox_body:
        raise HTTPException(
            status_code=400,
            detail="upstox orders require upstox_body (pass-through payload for /v2/order/place)",
        )

    client_order_id = (str(req.client_order_id).strip() if req.client_order_id else None)
    if client_order_id:
        from app.orders.state import get_order_by_client_order_id

        existing = get_order_by_client_order_id(client_order_id)
        if existing is not None:
            return {"ok": True, "order_state_id": existing.id, "idempotent_replay": True, "order": {**existing.__dict__, "meta": existing.meta}}

    # Persist intent BEFORE calling the broker (crash-safe).
    order_state_id = create_order(
        broker="upstox",
        instrument_key=str((req.upstox_body or {}).get("instrument_token") or ""),
        side=str((req.upstox_body or {}).get("transaction_type") or (req.upstox_body or {}).get("side") or "").upper() or "BUY",
        qty=float((req.upstox_body or {}).get("quantity") or 1),
        order_kind="MANUAL",
        order_type=str((req.upstox_body or {}).get("order_type") or "UPSTOX"),
        client_order_id=client_order_id,
        status="NEW",
        meta={"body": req.upstox_body, "tag": (req.upstox_body or {}).get("tag"), "request": req.model_dump()},
    )

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.place_order_v2(req.upstox_body)
        data = res.get("data") if isinstance(res, dict) else None
        broker_order_id = None
        if isinstance(data, dict):
            broker_order_id = data.get("order_id") or data.get("orderId")

        from app.orders.state import update_order

        update_order(
            order_state_id,
            status="SUBMITTED",
            broker_order_id=(None if not broker_order_id else str(broker_order_id)),
            meta_patch={"result": res},
        )
        log_event("order.upstox.place_v2", {"request": req.upstox_body, "result": res}, request_id=getattr(request.state, "request_id", None))
        return {**res, "order_state_id": order_state_id}
    except UpstoxError as e:
        from app.orders.state import update_order

        update_order(order_state_id, status="ERROR", last_error=str(e)[:500], meta_patch={"error": str(e)})
        log_event("order.upstox.error", {"where": "place_v2", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.post("/upstox/place-v3")
def upstox_place_v3(body: dict[str, Any], request: Request) -> dict:
    if settings.SAFE_MODE:
        raise HTTPException(status_code=403, detail="SAFE_MODE=true: live orders disabled")
    try:
        assert_trade_allowed()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.place_order_v3(body)
        log_event("order.upstox.place_v3", {"request": body, "result": res}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("order.upstox.error", {"where": "place_v3", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.put("/upstox/modify-v3")
def upstox_modify_v3(body: dict[str, Any], request: Request) -> dict:
    if settings.SAFE_MODE:
        raise HTTPException(status_code=403, detail="SAFE_MODE=true: live orders disabled")
    try:
        assert_trade_allowed()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.modify_order_v3(body)
        log_event("order.upstox.modify_v3", {"request": body, "result": res}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("order.upstox.error", {"where": "modify_v3", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.delete("/upstox/cancel-v3")
def upstox_cancel_v3(order_id: str, request: Request) -> dict:
    if settings.SAFE_MODE:
        raise HTTPException(status_code=403, detail="SAFE_MODE=true: live orders disabled")
    try:
        assert_trade_allowed()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.cancel_order_v3(order_id)
        log_event("order.upstox.cancel_v3", {"order_id": order_id, "result": res}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("order.upstox.error", {"where": "cancel_v3", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.get("/upstox/book")
def upstox_order_book(request: Request) -> dict:
    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.order_book_v2()
        log_event("order.upstox.book", {"ok": True}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("order.upstox.error", {"where": "book_v2", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.get("/upstox/details")
def upstox_order_details(order_id: str, request: Request) -> dict:
    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.order_details_v2(order_id)
        log_event("order.upstox.details", {"order_id": order_id}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("order.upstox.error", {"where": "details_v2", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.get("/upstox/positions")
def upstox_positions(request: Request) -> dict:
    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.positions_v2()
        log_event("portfolio.upstox.positions", {"ok": True}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("portfolio.upstox.error", {"where": "positions_v2", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.get("/upstox/holdings")
def upstox_holdings(request: Request) -> dict:
    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.holdings_v2()
        log_event("portfolio.upstox.holdings", {"ok": True}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("portfolio.upstox.error", {"where": "holdings_v2", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.post("/upstox/margin")
def upstox_margin(body: dict[str, Any], request: Request) -> dict:
    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client = UpstoxClient(cfg)
    try:
        res = client.margin_v2(body)
        log_event("portfolio.upstox.margin", {"request": body}, request_id=getattr(request.state, "request_id", None))
        return res
    except UpstoxError as e:
        log_event("portfolio.upstox.error", {"where": "margin_v2", "error": str(e)}, request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        client.close()


@router.get("/upstox/funds")
def upstox_funds(request: Request, segment: str | None = "EQ") -> dict:
    cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
    client: UpstoxClient | None = None
    try:
        client = UpstoxClient(cfg)
        res = client.funds_and_margin_v2(segment=segment)
        log_event(
            "portfolio.upstox.funds",
            {"segment": segment, "ok": True},
            request_id=getattr(request.state, "request_id", None),
        )
        return res
    except UpstoxError as e:
        log_event(
            "portfolio.upstox.error",
            {"where": "funds_and_margin_v2", "segment": segment, "error": str(e)},
            request_id=getattr(request.state, "request_id", None),
        )
        status = 400 if "not configured" in str(e).lower() else 502
        raise HTTPException(status_code=status, detail=str(e))
    finally:
        if client is not None:
            client.close()
