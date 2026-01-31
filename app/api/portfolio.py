from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.portfolio.service import PortfolioService
from app.paper_trading.service import PaperTradingService
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError
from app.auth import token_store
from app.utils.perf import perf_span

router = APIRouter(prefix="/portfolio")
svc = PortfolioService()
paper = PaperTradingService()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _not_logged_in(detail: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "detail": detail or "Upstox is not authenticated. Open /api/auth/upstox/login",
            "login_url": "/api/auth/upstox/login",
        },
    )


def _maybe_auth_401(e: Exception) -> JSONResponse | None:
    msg = str(e)
    if "Open /api/auth/upstox/login" in msg or "Upstox is not authenticated" in msg:
        return _not_logged_in(msg)
    if "Upstox HTTP 401" in msg:
        return _not_logged_in(msg)
    return None


def _coerce_live_balance(result: dict) -> dict | None:
    if not isinstance(result, dict):
        return None
    if result.get("error"):
        return None
    bal = result.get("balance")
    if bal is None:
        return None
    return {
        "total": bal,
        "segment": result.get("segment"),
        "raw": result.get("funds"),
    }


def _extract_list(payload: dict) -> list:
    if not isinstance(payload, dict):
        return []
    if payload.get("error"):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("holdings", "positions"):
            if isinstance(data.get(k), list):
                return list(data.get(k) or [])
    if isinstance(payload.get("holdings"), list):
        return list(payload.get("holdings") or [])
    if isinstance(payload.get("positions"), list):
        return list(payload.get("positions") or [])
    return []


@router.get("/balance")
def balance(source: str | None = None) -> dict:
    paper_balance = float(paper.account()["balance"])

    live_balance: dict | None = None
    try:
        live_balance = _coerce_live_balance(svc.balance(source="upstox"))
    except Exception:
        live_balance = None

    total = paper_balance + float((live_balance or {}).get("total", 0.0) or 0.0)
    return {
        "paper_balance": paper_balance,
        "live_balance": live_balance,
        "total_balance": total,
        "updated_at": _now_iso(),
    }


@router.get("/holdings")
def holdings(source: str | None = None) -> dict:
    paper_positions = (paper.positions() or {}).get("positions") or []
    paper_holdings: list[dict] = []
    for p in paper_positions:
        try:
            qty = float(p.get("qty") or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue
        paper_holdings.append(
            {
                "instrument_key": p.get("symbol"),
                "quantity": qty,
                "average_price": p.get("avg_price"),
                "last_price": None,
                "current_value": None,
                "pnl": None,
            }
        )

    live_holdings: list = []
    live_mf_holdings: list = []
    try:
        live_holdings = _extract_list(svc.holdings(source="upstox"))
    except Exception:
        live_holdings = []

    return {
        "paper_holdings": paper_holdings,
        "live_holdings": live_holdings,
        "live_mf_holdings": live_mf_holdings,
        "counts": {"paper": len(paper_holdings), "live": len(live_holdings), "live_mf": len(live_mf_holdings)},
        "updated_at": _now_iso(),
    }


@router.get("/positions")
def positions(source: str | None = None) -> dict:
    paper_positions_raw = (paper.positions() or {}).get("positions") or []
    paper_positions: list[dict] = []
    for p in paper_positions_raw:
        paper_positions.append(
            {
                "instrument_key": p.get("symbol"),
                "quantity": p.get("qty"),
                "average_price": p.get("avg_price"),
                "pnl": None,
            }
        )

    live_positions: list = []
    try:
        live_positions = _extract_list(svc.positions(source="upstox"))
    except Exception:
        live_positions = []

    return {
        "paper_positions": paper_positions,
        "live_positions": live_positions,
        "counts": {"paper": len(paper_positions), "live": len(live_positions)},
        "updated_at": _now_iso(),
    }


@router.get("/pnl")
def pnl(source: str | None = None) -> dict:
    return svc.pnl(source=source)


@router.get("/live/summary")
async def live_summary(source: str | None = None) -> dict:
    """Frontend compatibility endpoint for the sidebar widget."""

    # If Upstox is selected (explicitly or via auto) but not authenticated, return 401 so the UI can prompt login.
    if (source or "auto").lower() in {"auto", "upstox"} and not token_store.is_logged_in():
        return _not_logged_in("Upstox is not authenticated. Open /api/auth/upstox/login")

    errors: dict[str, object] = {}

    async def _call_balance() -> dict:
        try:
            with perf_span("portfolio.balance", source=(source or "auto")):
                res = await asyncio.to_thread(svc.balance, source)
            return res if isinstance(res, dict) else {"source": source or "auto", "balance": None, "error": "invalid response"}
        except Exception as e:
            return {"source": source or "auto", "balance": None, "error": str(e)}

    async def _call_holdings() -> dict:
        try:
            with perf_span("portfolio.holdings", source=(source or "auto")):
                res = await asyncio.to_thread(svc.holdings, source)
            return res if isinstance(res, dict) else {"source": source or "auto", "error": "invalid response"}
        except Exception as e:
            return {"source": source or "auto", "error": str(e)}

    async def _call_positions() -> dict:
        try:
            with perf_span("portfolio.positions", source=(source or "auto")):
                res = await asyncio.to_thread(svc.positions, source)
            return res if isinstance(res, dict) else {"source": source or "auto", "error": "invalid response"}
        except Exception as e:
            return {"source": source or "auto", "error": str(e)}

    bal, holds, pos = await asyncio.gather(_call_balance(), _call_holdings(), _call_positions())

    if isinstance(bal, dict) and bal.get("error"):
        errors["funds"] = bal.get("error")
    if isinstance(holds, dict) and holds.get("error"):
        errors["holdings"] = holds.get("error")
    if isinstance(pos, dict) and pos.get("error"):
        errors["positions"] = pos.get("error")

    # If any of these are auth-related, hard fail with 401 so frontend redirects to login.
    for k in ("funds", "holdings", "positions"):
        if k in errors and errors[k]:
            msg = str(errors[k])
            if "Open /api/auth/upstox/login" in msg or "not authenticated" in msg or "Upstox HTTP 401" in msg:
                token_store.clear_token()
                return _not_logged_in("Upstox token expired or missing. Open /api/auth/upstox/login")

    # Normalize shapes across paper/upstox.
    equity_holdings: list = _extract_list(holds if isinstance(holds, dict) else {})
    equity_positions: list = _extract_list(pos if isinstance(pos, dict) else {})

    # MF holdings: best-effort. Upstox MF endpoints are often unavailable for many accounts.
    mf_holdings: list = []
    try:
        mf = live_mf_holdings()
        if isinstance(mf, JSONResponse):
            # auth response
            return mf
        if isinstance(mf, dict):
            mf_holdings = list(mf.get("holdings") or []) if isinstance(mf.get("holdings"), list) else []
            if mf.get("detail"):
                errors["mf_holdings"] = mf.get("detail")
    except Exception as e:
        errors["mf_holdings"] = str(e)

    out = {
        "ok": True,
        "source": (bal.get("source") if isinstance(bal, dict) else None) or (source or "auto"),
        "funds": {
            "total": (bal.get("balance") if isinstance(bal, dict) else None),
            "equity_available_margin": (bal.get("balance") if isinstance(bal, dict) else None),
            "segment": (bal.get("segment") if isinstance(bal, dict) else None),
            "raw": (bal.get("funds") if isinstance(bal, dict) else None),
        },
        "equity_holdings": equity_holdings,
        "equity_positions": equity_positions,
        "mf_holdings": mf_holdings,
    }

    if errors:
        out["errors"] = errors
    return out


@router.get("/live/funds", response_model=None)
def live_funds():
    try:
        client = UpstoxClient(UpstoxConfig())
    except Exception as e:
        auth = _maybe_auth_401(e)
        return auth if auth is not None else JSONResponse(status_code=400, content={"detail": str(e)})

    try:
        try:
            # Upstox expects segment codes like SEC/COM/FO; many accounts reject EQ.
            payload = client.funds_and_margin_v2(segment="SEC")
        except Exception:
            # Some deployments reject the segment parameter entirely.
            payload = client.funds_and_margin_v2(segment=None)
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        equity = (data.get("equity") or {}) if isinstance(data, dict) else {}

        available = float(equity.get("available_margin", 0.0) or 0.0)
        used = float(equity.get("used_margin", 0.0) or 0.0)

        return {
            "funds": {
                "equity_available_margin": available,
                "equity_used_margin": used,
                "total": available,
                "raw": data,
            },
            "updated_at": _now_iso(),
        }
    except Exception as e:
        auth = _maybe_auth_401(e)
        return auth if auth is not None else JSONResponse(status_code=400, content={"detail": str(e)})
    finally:
        try:
            client.close()
        except Exception:
            pass


@router.get("/live/equity/holdings", response_model=None)
def live_equity_holdings():
    try:
        client = UpstoxClient(UpstoxConfig())
    except Exception as e:
        auth = _maybe_auth_401(e)
        return auth if auth is not None else JSONResponse(status_code=400, content={"detail": str(e)})

    try:
        payload = client.holdings_v2()
        holdings = payload.get("data", []) if isinstance(payload, dict) else []
        return {"holdings": list(holdings) if isinstance(holdings, list) else [], "updated_at": _now_iso()}
    except Exception as e:
        auth = _maybe_auth_401(e)
        return auth if auth is not None else JSONResponse(status_code=400, content={"detail": str(e)})
    finally:
        try:
            client.close()
        except Exception:
            pass


@router.get("/live/equity/positions", response_model=None)
def live_equity_positions():
    try:
        client = UpstoxClient(UpstoxConfig())
    except Exception as e:
        auth = _maybe_auth_401(e)
        return auth if auth is not None else JSONResponse(status_code=400, content={"detail": str(e)})

    try:
        payload = client.positions_v2()
        positions = payload.get("data", []) if isinstance(payload, dict) else []
        return {"positions": list(positions) if isinstance(positions, list) else [], "updated_at": _now_iso()}
    except Exception as e:
        auth = _maybe_auth_401(e)
        return auth if auth is not None else JSONResponse(status_code=400, content={"detail": str(e)})
    finally:
        try:
            client.close()
        except Exception:
            pass


@router.get("/live/mf/holdings", response_model=None)
def live_mf_holdings():
    token = token_store.get_access_token()
    if not token:
        return _not_logged_in()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    candidates = [
        "https://api.upstox.com/v2/mf/holdings",
        "https://api.upstox.com/v2/mf/portfolio/holdings",
        "https://api.upstox.com/v2/portfolio/mf/holdings",
    ]

    last_error: str | None = None
    for url in candidates:
        try:
            r = httpx.get(url, headers=headers, timeout=15)
            if r.status_code == 401:
                token_store.clear_token()
                return _not_logged_in("Upstox token expired. Open /api/auth/upstox/login")
            if r.status_code in (404, 400):
                last_error = r.text[:300]
                continue
            if r.status_code >= 400:
                last_error = r.text[:300]
                break
            payload = r.json() if r.content else {}
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if isinstance(data, list):
                return {"holdings": list(data), "updated_at": _now_iso()}
        except Exception as e:
            last_error = str(e)
            break

    # Best-effort: MF APIs often aren't enabled.
    return {"holdings": [], "updated_at": _now_iso(), "detail": last_error}
