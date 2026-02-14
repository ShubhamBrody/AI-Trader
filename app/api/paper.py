from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.broker.paper import PaperBroker
from app.candles.service import CandleService
from app.execution.service import ExecutionService
from app.paper_trading.service import PaperTradingService
from app.strategy.engine import StrategyEngine

router = APIRouter(prefix="/paper")
svc = PaperTradingService()
_candles = CandleService()
_execution = ExecutionService(PaperBroker())


class AmountRequest(BaseModel):
    amount: float = Field(..., gt=0)


class ExecuteRequest(BaseModel):
    symbol: str
    side: str = Field(..., pattern="^(BUY|SELL)$")
    qty: float = Field(..., gt=0)
    price: float = Field(..., gt=0)


class TradeDecisionRequest(BaseModel):
    instrument_key: str
    interval: str
    action: str
    quantity: int

    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None

    confidence: float | None = None
    confidence_raw: float | None = None

    risk_pct: float | None = None
    capital_used: float | None = None

    reason: str | None = None
    timestamp: datetime | None = None


@router.get("/account")
def account() -> dict:
    bal = float(svc.account()["balance"])
    # BackendComplete + frontend expect cash_balance.
    return {"cash_balance": bal, "balance": bal}


@router.post("/deposit")
def deposit(amount: float = Query(..., gt=0)) -> dict:
    out = svc.deposit(float(amount))
    bal = float(out["balance"])
    return {"status": "OK", "cash_balance": bal, "balance": bal}


@router.post("/withdraw")
def withdraw(amount: float = Query(..., gt=0)) -> dict:
    try:
        out = svc.withdraw(float(amount))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    bal = float(out["balance"])
    return {"status": "OK", "cash_balance": bal, "balance": bal}


@router.get("/positions")
def positions() -> dict:
    raw = (svc.positions() or {}).get("positions") or []
    mapped = [
        {
            "instrument_key": p.get("symbol"),
            "quantity": p.get("qty"),
            "average_price": p.get("avg_price"),
        }
        for p in raw
    ]
    return {"positions": mapped}


@router.post("/execute")
def execute_trade(
    instrument_key: str = Query(...),
    interval: str = Query(...),
    account_balance: float = Query(..., gt=0),
    lot_size: int = Query(1, gt=0),
) -> dict:
    """Contract-compatible paper execution endpoint.

    Strategy → sizing → place with PaperBroker using latest cached/polled price.
    """

    ik = str(instrument_key)
    interval = str(interval)

    # Get a recent price from cached candles (poll may be required first).
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)
    # Use a simple historical window; CandleService.get_historical reads DB and is cheap.
    series = _candles.get_historical(ik, interval, start, end)
    candles = list(series.candles or [])

    if not candles:
        # Try a lightweight poll to populate.
        try:
            series = _candles.poll_intraday(ik, interval, lookback_minutes=60)
            candles = list(series.candles or [])
        except Exception:
            candles = []

    last_price = float(candles[-1].close) if candles else 0.0
    if last_price <= 0:
        return {"status": "REJECTED", "reason": "no price available; run intraday poll first"}

    # Build a basic decision using StrategyEngine (same as /api/strategy/decision).
    svc_engine = StrategyEngine()
    highs = [float(c.high) for c in candles][-220:]
    lows = [float(c.low) for c in candles][-220:]
    closes = [float(c.close) for c in candles][-220:]
    volumes = [float(getattr(c, "volume", 0.0) or 0.0) for c in candles][-220:]
    idea = svc_engine.build_idea(symbol=ik, highs=highs, lows=lows, closes=closes, volumes=volumes)
    action = str(idea.side).upper()

    if action not in {"BUY", "SELL"}:
        return {"status": "REJECTED", "reason": "NO_TRADE", "action": action}

    # Position sizing: risk 1% of account, round to lot_size.
    entry = float(idea.entry)
    stop = float(idea.stop_loss)
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return {"status": "REJECTED", "reason": "invalid sizing", "action": action}

    risk_budget = float(account_balance) * 0.01
    raw_qty = int(risk_budget // risk_per_share)
    ls = max(1, int(lot_size))
    qty = (raw_qty // ls) * ls
    if qty <= 0:
        return {"status": "REJECTED", "reason": "qty=0", "action": action}

    result = _execution.place(symbol=ik, side=action, qty=float(qty), price=float(last_price))
    return {"status": "ACCEPTED", "result": result, "action": action, "quantity": qty, "price": last_price}


@router.post("/execute/manual")
def execute_manual(req: ExecuteRequest) -> dict:
    """Back-compat manual execution endpoint."""
    return svc.execute(symbol=req.symbol, side=req.side, qty=req.qty, price=req.price)


@router.post("/auto")
def auto_execute_trade(decision: TradeDecisionRequest) -> dict:
    action = str(decision.action or "").upper()
    if action not in {"BUY", "SELL"}:
        return {"status": "REJECTED", "reason": "NO_TRADE", "action": action}

    qty = int(decision.quantity)
    if qty <= 0:
        return {"status": "REJECTED", "reason": "qty=0", "action": action}

    price = float(decision.entry_price or 0.0)
    if price <= 0:
        # Fallback to latest close if not provided.
        try:
            series = _candles.poll_intraday(decision.instrument_key, decision.interval, lookback_minutes=60)
            candles = list(series.candles or [])
            price = float(candles[-1].close) if candles else 0.0
        except Exception:
            price = 0.0
    if price <= 0:
        return {"status": "REJECTED", "reason": "no price available"}

    result = _execution.place(symbol=decision.instrument_key, side=action, qty=float(qty), price=price)
    return {"status": "ACCEPTED", "result": result}


@router.get("/journal")
def journal(limit: int = 100) -> dict:
    raw = (svc.journal(limit=limit) or {}).get("journal") or []

    trades = []
    for r in raw:
        trades.append(
            {
                "id": r.get("id"),
                "timestamp": datetime.fromtimestamp(int(r.get("ts") or 0), tz=timezone.utc).isoformat() if r.get("ts") else None,
                "instrument_key": r.get("symbol"),
                "side": r.get("side"),
                "quantity": r.get("qty"),
                "entry_price": r.get("price"),
                "exit_price": None,
                "pnl": None,
                "raw": r,
            }
        )

    return {"trades": trades, "journal": raw}
