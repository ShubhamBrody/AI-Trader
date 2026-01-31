from __future__ import annotations

import sys
from loguru import logger

from app.core.settings import settings


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        backtrace=False,
        diagnose=False,
        enqueue=True,
        serialize=bool(getattr(settings, "LOG_JSON", False)),
    )
