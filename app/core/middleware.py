from __future__ import annotations

import time
from collections import deque
import ipaddress
from typing import Deque
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from loguru import logger

from app.core.settings import settings
from app.utils.request_context import request_id_var


def _client_ip(request: Request) -> str:
    # Prefer X-Forwarded-For when behind a proxy.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = req_id
        token = request_id_var.set(req_id)
        try:
            resp = await call_next(request)
            resp.headers["x-request-id"] = req_id
            return resp
        finally:
            try:
                request_id_var.reset(token)
            except Exception:
                pass


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.REQUIRE_API_KEY:
            return await call_next(request)

        write_key = settings.API_KEY_WRITE or settings.API_KEY
        read_key = settings.API_KEY_READ or settings.API_KEY
        admin_key = settings.API_KEY_ADMIN or write_key
        if not write_key and not read_key:
            # Misconfigured; fail closed.
            return JSONResponse(status_code=500, content={"detail": "REQUIRE_API_KEY=true but no API key configured"})

        path = request.url.path
        method = request.method.upper()

        def _keys(s: str | None) -> set[str]:
            if not s:
                return set()
            # Support comma-separated lists.
            parts = [p.strip() for p in str(s).split(",")]
            return {p for p in parts if p}

        write_keys = _keys(write_key)
        read_keys = _keys(read_key)
        admin_keys = _keys(admin_key)

        # Protect known high-risk areas regardless of method.
        protected_prefixes = (
            "/api/orders",
            "/api/learning",
            "/api/agent",
            "/api/controls",
            "/api/positions",
            "/api/audit",
            "/api/alerts",
        )
        protected = path.startswith(protected_prefixes)

        # Also protect all API writes to reduce footguns.
        is_api = path.startswith("/api/")
        is_write = method in {"POST", "PUT", "PATCH", "DELETE"}
        if is_api and is_write:
            protected = True

        # Admin-only endpoints (even if user has write key).
        admin_only = False
        if path.startswith("/api/controls/emergency"):
            admin_only = True
        if path in {"/api/agent/start", "/api/agent/stop", "/api/agent/flatten", "/api/agent/cancel-open-orders"}:
            admin_only = True
        if path == "/api/prediction/retention/cleanup":
            admin_only = True

        if protected:
            key = request.headers.get("x-api-key")
            expected_keys = write_keys if is_write else read_keys
            if admin_only and is_write:
                expected_keys = admin_keys

            if not expected_keys or key not in expected_keys:
                return JSONResponse(status_code=401, content={"detail": "invalid api key"})

        # Optional IP allowlist for API writes (defense-in-depth).
        if is_api and is_write:
            cidrs = str(getattr(settings, "API_WRITE_ALLOWLIST_CIDRS", "") or "").strip()
            if cidrs:
                ip = _client_ip(request)
                try:
                    addr = ipaddress.ip_address(ip)
                    allowed = False
                    for c in [p.strip() for p in cidrs.split(",") if p.strip()]:
                        try:
                            net = ipaddress.ip_network(c, strict=False)
                        except Exception:
                            continue
                        if addr in net:
                            allowed = True
                            break
                    if not allowed:
                        return JSONResponse(status_code=403, content={"detail": "ip not allowed"})
                except Exception:
                    # If we cannot parse the client IP, fail closed.
                    return JSONResponse(status_code=403, content={"detail": "ip not allowed"})
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, Deque[float]] = {}

    async def dispatch(self, request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        ip = _client_ip(request)
        now = time.time()
        window = 60.0
        limit = max(1, int(settings.RATE_LIMIT_PER_MINUTE))

        q = self._hits.get(ip)
        if q is None:
            q = deque()
            self._hits[ip] = q

        # Evict old timestamps.
        cutoff = now - window
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= limit:
            return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})

        q.append(now)
        return await call_next(request)


class PerformanceLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not getattr(settings, "PERF_LOG_ENABLED", True):
            return await call_next(request)

        t0 = time.perf_counter()
        status_code: int | None = None
        try:
            resp = await call_next(request)
            status_code = int(getattr(resp, "status_code", 0) or 0)
            return resp
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            rid = None
            try:
                rid = getattr(request.state, "request_id", None)
            except Exception:
                rid = None
            if not rid:
                rid = request.headers.get("x-request-id")
            rid = rid or "-"

            method = str(getattr(request, "method", "?") or "?").upper()
            path = str(getattr(getattr(request, "url", None), "path", "?") or "?")

            slow_ms = int(getattr(settings, "PERF_LOG_SLOW_MS", 250) or 250)
            lvl = "WARNING" if dt_ms >= float(slow_ms) else "INFO"
            sc = status_code if status_code is not None else "?"
            logger.log(lvl, "HTTP {method} {path} -> {status} ({ms:.1f}ms) rid={rid}", method=method, path=path, status=sc, ms=dt_ms, rid=rid)
