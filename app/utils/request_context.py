from __future__ import annotations

from contextvars import ContextVar

# Best-effort request correlation for logs.
# HTTP middleware sets this; asyncio.to_thread propagates contextvars by default.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
