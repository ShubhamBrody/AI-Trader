from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, TypeVar, cast

from loguru import logger

from app.core.settings import settings
from app.utils.request_context import request_id_var

T = TypeVar("T")


def _enabled() -> bool:
    return bool(getattr(settings, "PERF_LOG_ENABLED", True)) and bool(getattr(settings, "PERF_LOG_INNER_ENABLED", True))


def _slow_ms() -> int:
    try:
        return int(getattr(settings, "PERF_LOG_SLOW_MS", 250) or 250)
    except Exception:
        return 250


def _always_inner() -> bool:
    return bool(getattr(settings, "PERF_LOG_INNER_ALWAYS", False))


@contextmanager
def perf_span(op: str, **tags: Any):
    """Measure a code block; logs ms.

    - Always logs if PERF_LOG_INNER_ALWAYS=true
    - Otherwise logs only when >= PERF_LOG_SLOW_MS
    """

    if not _enabled():
        yield
        return

    t0 = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        rid = request_id_var.get() or "-"
        slow_ms = _slow_ms()
        should = _always_inner() or (dt_ms >= float(slow_ms))
        if not should:
            return

        level = "WARNING" if dt_ms >= float(slow_ms) else "DEBUG"
        status = "ok" if ok else "err"
        # Keep tag payload compact.
        extra = {k: v for k, v in tags.items() if v is not None}
        logger.log(level, "PERF {op} {status}: {ms:.1f}ms rid={rid} tags={tags}", op=op, status=status, ms=dt_ms, rid=rid, tags=extra)


def timed(op: str | None = None, **fixed_tags: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for timing functions with perf_span."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        name = op or f"{fn.__module__}.{getattr(fn, '__name__', 'fn')}"

        def wrapped(*args: Any, **kwargs: Any) -> T:
            with perf_span(name, **fixed_tags):
                return fn(*args, **kwargs)

        return cast(Callable[..., T], wrapped)

    return deco
