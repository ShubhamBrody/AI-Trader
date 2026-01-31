from __future__ import annotations

import secrets
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from app.auth import token_store
from app.core.settings import settings

router = APIRouter(prefix="/auth/upstox", tags=["auth"])


AUTH_DIALOG_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def _frontend_default() -> str:
    return getattr(settings, "FRONTEND_URL", None) or "http://localhost:5173/"


def _redirect_uri_default() -> str:
    return getattr(settings, "UPSTOX_REDIRECT_URI", None) or "http://localhost:8000/api/auth/upstox/callback"


@router.get("/status")
def status() -> dict:
    """Frontend auth status endpoint.

    This backend currently uses env-based UPSTOX_ACCESS_TOKEN (no interactive OAuth storage).
    The UI only needs a boolean to show Connected/Not connected.
    """

    return {
        "logged_in": token_store.is_logged_in(),
        "login_url": "/api/auth/upstox/login",
        "redirect_uri": _redirect_uri_default(),
        "frontend_url": _frontend_default(),
        "has_client_id": bool(settings.UPSTOX_CLIENT_ID),
        "has_client_secret": bool(settings.UPSTOX_CLIENT_SECRET),
        "safe_mode": bool(settings.SAFE_MODE),
    }


@router.get("/login")
def login(next: str | None = Query(None, description="Where to redirect after successful login")):
    if not settings.UPSTOX_CLIENT_ID:
        raise HTTPException(status_code=500, detail="UPSTOX_CLIENT_ID is not configured")

    redirect_uri = _redirect_uri_default()
    state = secrets.token_urlsafe(24)

    params = {
        "response_type": "code",
        "client_id": settings.UPSTOX_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "state": state,
    }

    url = f"{AUTH_DIALOG_URL}?{urlencode(params)}"
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie(
        key="upstox_oauth_state",
        value=state,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=600,
        path="/",
    )
    if next:
        resp.set_cookie(
            key="upstox_oauth_next",
            value=next,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=600,
            path="/",
        )
    return resp


@router.get("/callback")
def callback(
    request: Request,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    error_description: str | None = Query(None),
):
    def _redirect_with_error(reason: str, description: str | None = None) -> RedirectResponse:
        next_url = request.cookies.get("upstox_oauth_next") or _frontend_default()
        parsed = urlparse(next_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.update({"upstox_login": "failed", "reason": reason})
        if description:
            query["detail"] = description
        out_url = urlunparse(parsed._replace(query=urlencode(query)))
        out = RedirectResponse(url=out_url, status_code=302)
        out.delete_cookie("upstox_oauth_state", path="/")
        out.delete_cookie("upstox_oauth_next", path="/")
        return out

    if error:
        return _redirect_with_error(reason=str(error), description=error_description)

    if not code:
        return _redirect_with_error(
            reason="missing_code",
            description="Open /api/auth/upstox/login to start authentication (do not open /callback directly).",
        )

    expected_state = request.cookies.get("upstox_oauth_state")
    if expected_state and not state:
        return _redirect_with_error(reason="missing_state", description="Missing OAuth state")
    if expected_state and state and expected_state != state:
        return _redirect_with_error(reason="invalid_state", description="Invalid OAuth state")

    if not settings.UPSTOX_CLIENT_ID or not settings.UPSTOX_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="UPSTOX_CLIENT_ID/UPSTOX_CLIENT_SECRET not configured")

    redirect_uri = _redirect_uri_default()
    form = {
        "code": code,
        "client_id": settings.UPSTOX_CLIENT_ID,
        "client_secret": settings.UPSTOX_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(TOKEN_URL, data=form, headers={"Accept": "application/json"})
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to reach Upstox token endpoint")

    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Upstox token exchange failed: {resp.text}")

    data = resp.json() if resp.content else {}
    access_token = data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Upstox response missing access_token")

    token_store.set_token(
        access_token=str(access_token),
        token_type=data.get("token_type"),
        expires_in=data.get("expires_in") if isinstance(data.get("expires_in"), int) else None,
        raw=data if isinstance(data, dict) else None,
    )

    next_url = request.cookies.get("upstox_oauth_next") or _frontend_default()
    out = RedirectResponse(url=next_url, status_code=302)
    out.delete_cookie("upstox_oauth_state", path="/")
    out.delete_cookie("upstox_oauth_next", path="/")
    return out


@router.post("/logout")
def logout() -> dict:
    token_store.clear_token()
    return {"ok": True}
